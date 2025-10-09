# Correlation Kernel Update — Block Assignment & GPU Spec

**Timestamp:** 2025-10-09 10:06 (America/Chicago)  
**Version:** v0.4 — dropdown + Nsight tables only

<details open>
  <summary><strong>Nsight Compute — Grid Size Experiments (demodulationCrossCorrelation)</strong></summary>

  #### Setup
  - Device: NVIDIA Quadro GV100 (Volta, CC 7.0) — Driver 12.9 / Runtime 12.3, ECC On  
  - Kernel: `demodulationCrossCorrelation`  
  - Launch: `blockDim.x = 256`, `gridDim.x = {8192 | 7824}`, `segmentSize = 2*W`  
  - Notes: tail-safe guards enabled; identical dataset across runs

  #### Per-run results (your export)
  **Grid = 8192 (power of two)**

  | Run | Duration (µs) | Compute Thruput (%) | Memory Thruput (%) | Registers / thr |
  |---:|---:|---:|---:|---:|
  | 1 | 669.95 | 50.12 | 3.17 | 37 |
  | 2 | 580.38 | 54.40 | 3.19 | 37 |
  | 3 | 613.95 | 46.65 | 4.01 | 37 |
  | 4 | 647.33 | 51.85 | 2.85 | 37 |

  **Grid = 7824 (odd)**

  | Run | Duration (µs) | Compute Thruput (%) | Memory Thruput (%) | Registers / thr |
  |---:|---:|---:|---:|---:|
  | 1 | 631.39 | 46.47 | 3.60 | 37 |
  | 2 | 668.96 | 48.95 | 2.85 | 37 |
  | 3 | 671.04 | 44.32 | 3.85 | 37 |
  | 4 | 635.97 | 46.71 | 4.46 | 37 |

  #### Summary (averages)
  | GridDim.x | Avg Duration (µs) | Avg Compute % | Avg Memory % |
  |---:|---:|---:|---:|
  | **8192** | **627.90** | **50.75** | 3.30 |
  | 7824 | 651.84 | 46.61 | **3.69** |

  **Conclusion:** With `blockDim.x = 256`, the **power-of-two grid (8192)** is ~**3.7% faster** on average and sustains **higher compute throughput** with tighter tail utilization. Keep **gridDim.x = 2^k** when feasible for this kernel/dataset.

</details>

<details>
  <summary><strong>Implementation notes (block assignment & indexing)</strong></summary>

  - Use **grid-stride over segments** and **block-stride over W×W inner work**.  
  - Keep **tail guards** for all global reads/writes.  
  - Preferred **block size = 256** on GV100 (better occupancy vs 1024).

  ```cpp
  // Host launch (excerpt)
  const int TPB = 256;
  long long perChan = N_total >> 1;
  int segs = (int)((perChan + segmentSize - 1) / segmentSize);
  dim3 block(TPB);
  dim3 grid(std::min(segs, 80 * 8)); // GV100: 80 SMs

  corrSegments<<<grid, block>>>(dA, dB, N_total, dCorr, segmentSize, W, W*W);
</details> <details> <summary><strong>GPU specification (Quadro GV100)</strong></summary>
SMs / CUDA Cores: 80 × 64 = 5120 • Compute Capability: 7.0 (Volta)

Memory: 32 GB HBM2, 4096-bit (≈ 870 GB/s) • L2: 6 MB

Clocks: ~1627 MHz core, 850 MHz mem • ECC: On

Limits: 1024 threads/block, 2048 threads/SM • Grid: (2,147,483,647; 65,535; 65,535)

</details> ```