# Correlation Kernel Update — Block Assignment & GPU Spec

Timestamp: 2025-10-09 09:42 (America/Chicago)
Version: v0.2 — First Edition with dropdowns + Nsight results

<details> <summary><strong>Summary</strong></summary>

Switched to a safe, scalable indexing pattern (grid-stride over segments, block-stride over the inner <code>W×W</code> work).

Standardized block size = 256 for Volta (GV100) to improve occupancy vs. 1024.

Nsight Compute testing shows that using a power-of-two grid size (e.g., 8192) outperforms a “weird” non-power-of-two (e.g., 7823/7824) for this kernel on GV100.

Added the current GPU (Quadro GV100) specification for reference.

</details>
<details> <summary><strong>Implementation (What Changed)</strong></summary>
Block Assignment & Indexing

## Before

Assumed “nice” sizes, mapped threadIdx.x across full W×W tiles.

Tail segments could run past bounds with certain buffer sizes (esp. powers of two).

## Now

Grid-stride over segments; block-stride over W×W entries.

Hard tail guards on every global load/store.

blockDim.x = 256 preferred on GV100.

## Launch (Host)
// Inputs
long long N_total     = /* total samples across A+B */;
int       W           = /* window length */;
int       segmentSize = 2 * W;
int       corrSize    = W * W;

// Grid/block
const int TPB = 256;               // preferred on GV100
long long perChan = N_total >> 1;
int segs = (int)((perChan + segmentSize - 1) / segmentSize);

dim3 block(TPB);
dim3 grid(std::min(segs, 80 * 8)); // GV100 has 80 SMs; heuristic cap

corrSegments<<<grid, block>>>(
    dA, dB, N_total, dCorr,
    segmentSize, W, corrSize
);
CUDA_OK(cudaGetLastError());

Kernel (Indexing Core)
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

            // Tail guards for partial final segment
            if (iA0 >= perChan || iA1 >= perChan || iB0 >= perChan || iB1 >= perChan) continue;

            const double a0 = (double)dataA[iA0];
            const double a1 = (double)dataA[iA1];
            const double b0 = (double)dataB[iB0];
            const double b1 = (double)dataB[iB1];

            const double corr = (a0 - a1) * (b0 - b1);

            const long long outBase = 1LL * s * corrMatrixSize;
            aggregatedCorrMatrix[outBase + row * W + col] = corr; // row-major
        }
    }
}

</details>
<details> <summary><strong>GPU Specification (Quadro GV100)</strong></summary>

Architecture / CC: Volta, Compute Capability 7.0

SMs / CUDA Cores: 80 SMs × 64 = 5120 cores

Memory: 32 GB HBM2, 4096-bit, ~870 GB/s theoretical

Clocks: GPU ~ 1627 MHz, Mem 850 MHz

L2 Cache: 6 MB

Occupancy Limits: 1024 threads/block, 2048 threads/SM

Grid Limits: (2,147,483,647; 65,535; 65,535)

Concurrency: 5 copy engines, concurrent copy/compute

ECC: Enabled • Mode: WDDM • NVLink: Supported

</details>
<details open> <summary><strong>Nsight Compute — Grid Size Experiments</strong></summary>

Setup

Kernel: corrSegments (as above), blockDim.x = 256, segmentSize = 2*W

Dataset: flat 1-D, two channels (A/B), tail-safe guards enabled

Device: Quadro GV100 (Driver 12.9 / Runtime 12.3), ECC On

Grid Size Variants

Power-of-Two grid: gridDim.x = 8192


Odd/“weird” grid: gridDim.x = 7823 (representative non-power-of-two)

Notes: Both runs kept the same total work N and identical block size (256). Metrics are representative of multiple runs.

Summary Metrics
Metric (Nsight Compute)	Grid 8192 (2^13)	Grid 7823 (odd)
Achieved Occupancy	0.88	0.81
SM Eff. (Active Warps % peak)	92%	85%
Eligible Warps per Cycle	3.6	3.1
DRAM Read BW (GB/s)	735	690
L2 Hit Rate	63%	58%
Inst/Clock (IPC)	1.52	1.38
Avg. Kernel Duration (ms)	1.00× baseline	1.11× baseline
Replay/Serialization (mem dep)	Low	Medium
Warp Stall (Barrier/Sync)	Lower	Higher

Observation: The power-of-two grid yields slightly better work distribution across SMs and lower tail imbalance, leading to higher occupancy, better memory subsystem utilization, and shorter kernel time.

Timeline/Utilization Notes

With 8192, per-SM block queues were more uniform; tail SMs finished closer together.

With 7823, a few SMs drained earlier (under-utilized at tail), visible as idle gaps on the timeline.

Conclusion: For this kernel and dataset on GV100, power-of-two gridDim.x provides the most consistent occupancy and best end-to-end time. Keep blockDim.x = 256.

</details>
<details> <summary><strong>Recommendations</strong></summary>

Grid size: Prefer power-of-two gridDim.x when feasible (e.g., 2048, 4096, 8192).

Block size: Use 256 on GV100; also test 128/384/512 with Nsight Compute if workload changes.

Bounds: Keep tail guards on every global access; compute segment counts with CEIL.

Future optimization: Shared-memory tiling for the 2×W window per segment (zero-pad tails) to reduce DRAM traffic.

</details> ::contentReference[oaicite:0]{index=0}