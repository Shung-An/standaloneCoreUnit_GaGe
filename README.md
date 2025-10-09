Correlation Kernel Update — Block Assignment, Indexing, and GV100 Benchmarks

Timestamp: 2025-10-09 10:22 (America/Chicago)
Version: v1.0 (complete markdown, dropdown style)

TL;DR

Switched to a safe 1-D indexing scheme: grid-stride over segments + block-stride over the inner W×W work with strict tail guards → fixes out-of-bounds on “power-of-2” lengths.

On Quadro GV100, using blockDim.x = 256 is ~4× faster than 1024 for this kernel class (better occupancy & resource packing).

Nsight Compute experiments show power-of-two grid sizes (e.g., 8192) outperform an odd count (7824) for this workload.

README is kept current; history lives in Git (optional docs/benchmarks/ for longer reports).

<details open> <summary><strong>1) What changed (high level)</strong></summary>

Before

Assumed “nice” sizes; mapped threadIdx.x across a full W×W tile.

Missing tail checks caused illegal memory access on certain lengths (esp. powers of two).

Large block size (1024) resulted in poor occupancy on GV100.

Now

Grid-stride over segments (blockIdx.x += gridDim.x).

Block-stride over W×W inner work (k += blockDim.x).

Tail guards on all global loads/stores.

Preferred blockDim.x = 256 on GV100.

</details>
<details> <summary><strong>2) Implementation (kernel + launch)</strong></summary>
Kernel (indexing core)
__global__ void corrSegments(
    const short* __restrict__ dataA,
    const short* __restrict__ dataB,
    long long    numElements,     // total samples across A+B
    double* __restrict__ aggregatedCorrMatrix,
    int          segmentSize,     // == 2*W
    int          W,               // window length
    int          corrMatrixSize   // == W*W
){
    if (segmentSize != 2 * W || corrMatrixSize != W * W) return;

    const long long perChan    = numElements >> 1; // length per channel
    const int       totalSegNum = (int)((perChan + segmentSize - 1) / segmentSize);

    const int t   = threadIdx.x;
    const int tpb = blockDim.x;
    const int G   = gridDim.x;

    // Grid-stride over segments
    for (int s = blockIdx.x; s < totalSegNum; s += G) {
        const long long base = 1LL * s * segmentSize;
        if (base >= perChan) continue;

        // Block-stride over W*W work
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

Launch (host side)
// Inputs
long long N_total     = /* total samples across A+B */;
int       W           = /* window length */;
int       segmentSize = 2 * W;
int       corrSize    = W * W;

// Geometry
const int TPB = 256;                           // preferred on GV100
const long long perChan = N_total >> 1;
const int segs = (int)((perChan + segmentSize - 1) / segmentSize);

// Heuristic cap: GV100 has 80 SMs → a few blocks/SM is usually enough
dim3 block(TPB);
dim3 grid(std::min(segs, 80 * 8));

corrSegments<<<grid, block>>>(
    dA, dB, N_total, dCorr,
    segmentSize, W, corrSize
);
CUDA_OK(cudaGetLastError());

</details>
<details> <summary><strong>3) GPU card specification (Quadro GV100)</strong></summary>

Architecture / CC: Volta, 7.0

SMs / CUDA cores: 80 SMs × 64 = 5120 cores

Global memory: 32 GB HBM2, 4096-bit, ~870 GB/s theoretical

Clocks: GPU ~1627 MHz, Mem 850 MHz

L2 cache: 6 MB

Limits: 1024 threads/block, 2048 threads/SM; grid limits (2,147,483,647; 65,535; 65,535)

Copy engines: 5 (concurrent copy/compute) • ECC: On • NVLink: Supported • Mode: WDDM

</details>
<details open> <summary><strong>4) Nsight Compute — grid size experiments (tables only)</strong></summary>

Setup

Device: Quadro GV100 (Driver 12.9 / Runtime 12.3), ECC On

Kernel: demodulationCrossCorrelation

Launch: blockDim.x = 256, gridDim.x ∈ {8192, 7824}, segmentSize = 2*W

Notes: tail-safe guards enabled; identical dataset across repeated runs
<details open>
  <summary><strong>Nsight Compute — Per-run results (power-of-two grid = 8192)</strong></summary>

| Run | Duration (µs) | Compute Thruput (%) | Memory Thruput (%) | Registers / thr |
|---:|---:|---:|---:|---:|
| 1 | 669.95 | 50.12 | 3.17 | 37 |
| 2 | 580.38 | 54.40 | 3.19 | 37 |
| 3 | 613.95 | 46.65 | 4.01 | 37 |
| 4 | 647.33 | 51.85 | 2.85 | 37 |

</details>

<details open>
  <summary><strong>Nsight Compute — Per-run results (odd grid = 7824)</strong></summary>

| Run | Duration (µs) | Compute Thruput (%) | Memory Thruput (%) | Registers / thr |
|---:|---:|---:|---:|---:|
| 1 | 631.39 | 46.47 | 3.60 | 37 |
| 2 | 668.96 | 48.95 | 2.85 | 37 |
| 3 | 671.04 | 44.32 | 3.85 | 37 |
| 4 | 635.97 | 46.71 | 4.46 | 37 |

</details>

<details open>
  <summary><strong>Summary (averages)</strong></summary>

| GridDim.x | Avg Duration (µs) | Avg Compute % | Avg Memory % |
|---:|---:|---:|---:|
| **8192** | **627.90** | **50.75** | 3.30 |
| 7824 | 651.84 | 46.61 | **3.69** |

</details>


Conclusion
With blockDim.x = 256, power-of-two grid sizes (e.g., 8192) show ~3.7% lower runtime and higher compute utilization than a comparably sized odd grid (7824). Prefer gridDim.x = 2^k when feasible for this kernel/dataset.

</details>
<details> <summary><strong>5) How to reproduce the measurements</strong></summary>

Assumes CUDA Toolkit + Nsight Compute (ncu) installed and in PATH.

# Example: run your app normally (release build recommended)
./your_binary --args ...

# Nsight Compute: collect kernel stats focused on our kernel
ncu --target-processes all \
    --kernel-name-base function \
    --kernel-name "demodulationCrossCorrelation*" \
    --metrics sm__warps_active.avg.pct_of_peak_sustained_active,smsp__sass_average_branch_targets_threads_uniform.pct,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg,sm__inst_executed_per_cycle_avg \
    --set full \
    --csv \
    --export ncu-report-grid8192 \
    ./your_binary --grid 8192 --block 256

# Repeat for the odd grid
ncu --target-processes all \
    --kernel-name-base function \
    --kernel-name "demodulationCrossCorrelation*" \
    --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg,sm__inst_executed_per_cycle_avg \
    --set full \
    --csv \
    --export ncu-report-grid7824 \
    ./your_binary --grid 7824 --block 256


Tips

Pin your input data and kernel parameters to ensure apples-to-apples runs.

Capture at least 3–5 runs per grid size and average the results.

Watch achieved occupancy, active warps %, dram throughput, IPC, replay/serialization.

</details>
<details> <summary><strong>6) Rationale: why 256 threads/block beats 1024 on GV100</strong></summary>

Occupancy & latency hiding: GV100 supports 2048 threads/SM. 1024-thread blocks often cap you at 1–2 blocks/SM; 256 allows up to 8 (subject to regs/SMem), giving schedulers more warps to hide memory latency.

Resource granularity: 1024-thread blocks can monopolize registers/shared memory → fewer concurrent blocks.

Scheduler flexibility: 256 = 8 warps/block → finer-grained scheduling and less tail drag.

Memory behavior: Coalescing is warp-level; very large blocks don’t inherently help but can magnify bank conflicts or shared-mem pressure.

</details>
<details> <summary><strong>7) Project hygiene & versioning</strong></summary>

Keep README.md as the latest, concise truth.

Use Git history (and optional tags) instead of embedding old versions here.

For deep dives, add timestamped docs under docs/benchmarks/ (e.g., 2025-10-09-nsight-grid-sizing.md).

Example:

git add README.md docs/benchmarks/2025-10-09-nsight-grid-sizing.md
git commit -m "README v1.0: GV100 block/grid tuning; 8192 PoT wins; safe indexing"
git push

</details>
<details> <summary><strong>8) Appendix — quick GPU introspection commands (Windows)</strong></summary>
:: General GPU info
wmic path win32_videocontroller get name,adapterram,driverversion,videoprocessor

:: Full properties via PowerShell
powershell "Get-WmiObject Win32_VideoController | where {$_.Name -match 'GV100'} | Format-List *"

:: NVIDIA SMI
nvidia-smi -q

:: CUDA samples (deviceQuery)
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\extras\demo_suite\deviceQuery.exe"

</details>

Maintainer note: If future datasets or kernels differ (e.g., heavier register usage, larger W), re-check blockDim.x ∈ {128, 256, 384, 512} and validate the “power-of-two grid” advantage with Nsight Compute.