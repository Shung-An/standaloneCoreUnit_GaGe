# Correlation Kernel Update — Block Assignment & GPU Spec

Timestamp: 2025-10-09 09:34 (America/Chicago)
Version: v0.1 (First Edition)

##Summary

This update standardizes the block assignment strategy and documents the current GPU (Quadro GV100) capabilities as observed via deviceQuery. The kernel now uses a safe, scalable indexing pattern that eliminates out-of-bounds (OOB) accesses across segment tails and improves performance by favoring 256-thread blocks.

What Changed (Block Assignment)

Before: Launches occasionally assumed “nice” sizes (e.g., power-of-two buffer lengths) and used threadIdx.x to cover a full W×W tile without tail guards, causing illegal memory access when sizes didn’t divide cleanly.

Now:

Use grid-stride over segments and block-stride over the inner W×W work.

Add hard tail guards for the final (partial) segment.

Prefer block size = 256 on Volta (GV100) for higher occupancy vs. 1024.

Launch (host)
// Inputs
long long N_total     = /* total samples across A+B */;
int       W           = /* window length */;
int       segmentSize = 2 * W;
int       corrSize    = W * W;

// Grid/block
const int TPB = 256;                 // preferred on GV100
long long perChan = N_total >> 1;
int segs = (int)((perChan + segmentSize - 1) / segmentSize);
dim3 block(TPB);
dim3 grid(std::min(segs, 80 * 8));   // GV100 has 80 SMs; cap is a heuristic

corrSegments<<<grid, block>>>(
    dA, dB, N_total, dCorr,
    segmentSize, W, corrSize
);
CUDA_OK(cudaGetLastError());

Kernel (indexing core)
__global__ void corrSegments(
    const short* __restrict__ dataA,
    const short* __restrict__ dataB,
    long long    numElements,     // A+B length
    double* __restrict__ aggregatedCorrMatrix,
    int          segmentSize,     // == 2*W
    int          W,               // window length
    int          corrMatrixSize   // == W*W
){
    if (segmentSize != 2 * W || corrMatrixSize != W * W) return;

    const long long perChan = numElements >> 1;
    const int totalSegNum = (int)((perChan + segmentSize - 1) / segmentSize);

    const int t   = threadIdx.x;
    const int tpb = blockDim.x;
    const int G   = gridDim.x;

    // Grid-stride over segments
    for (int s = blockIdx.x; s < totalSegNum; s += G) {
        const long long base = 1LL * s * segmentSize;
        if (base >= perChan) continue;

        // Block-stride over W*W entries
        for (int k = t; k < W * W; k += tpb) {
            const int row = k / W;
            const int col = k % W;

            const long long iA0 = base + row;
            const long long iA1 = base + W + row;
            const long long iB0 = base + col;
            const long long iB1 = base + W + col;

            if (iA0 >= perChan || iA1 >= perChan || iB0 >= perChan || iB1 >= perChan) continue;

            const double a0 = (double)dataA[iA0];
            const double a1 = (double)dataA[iA1];
            const double b0 = (double)dataB[iB0];
            const double b1 = (double)dataB[iB1];

            const double corr = (a0 - a1) * (b0 - b1);
            const long long outBase = 1LL * s * corrMatrixSize;
            aggregatedCorrMatrix[outBase + row * W + col] = corr;  // row-major
        }
    }
}

Why 256 Threads/Block?

Occupancy: On GV100 (80 SMs, 2048 threads/SM), 256-thread blocks pack more concurrent blocks per SM than 1024, improving latency hiding.

Resource granularity: 1024-thread blocks can monopolize registers/shared memory, limiting concurrency; 256 usually balances throughput and resource usage.

Scheduling flexibility: 256 threads = 8 warps; schedulers can interleave more warps across memory stalls.

Current GPU Specification (from deviceQuery)

Model: NVIDIA Quadro GV100 (Volta, Compute Capability 7.0)

Driver / Runtime: 12.9 / 12.3

SMs / CUDA Cores: 80 SMs × 64 = 5120 CUDA cores

Global Memory: 32 GB HBM2

Clocks: GPU Max ~ 1627 MHz, Memory 850 MHz

Bus / Bandwidth: 4096-bit HBM2 (≈ 870 GB/s theoretical)

L2 Cache: 6 MB

Max Threads: 1024 per block, 2048 per SM

Grid Limits: (2,147,483,647; 65,535; 65,535)

Copy Engines: Concurrent copy/compute, 5 copy engines

ECC: Enabled

Mode: WDDM

NVLink: Supported

Notes & Next Steps

This edition focuses on indexing safety and block sizing.

Optional next optimization: shared-memory tiling for the 2×W window per segment to reduce global traffic (zero-pad tails).

End of README v0.1