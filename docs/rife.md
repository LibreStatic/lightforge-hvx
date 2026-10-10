# RIFE v4.6 flow network on the Hexagon cDSP (HVX)

Computes `flow` (4xHxW, pixels at input resolution, [0:2] towards img0 and [2:4] towards img1) and
`mask` (1xHxW, after sigmoid) for two RGBA8 frames at a fixed size (multiples of 32; tested 320x192) plus
timestep t. The high-resolution warp stays on the CPU (Lightforge). One RPC per frame pair.

## Layout

| path | purpose |
| --- | --- |
| `tools/export_weights.py` | PyTorch/ncnn weights -> `rife46_f32.bin` / `rife46_hvx.bin` + goldens in `tests/golden` |
| `inc/lfrife_weights.h` | blob format |
| `ref/` | portable C11 fp32 reference (host and arm64) |
| `dsp/lfrife_net.c` | the network (HVX; builds on the host with `-DLF_HOST` and `dsp/hvx_emu.h`) |
| `dsp/lfrife_skel.c` | FastRPC skel (SETUP, RUN, probes), `dsp/lfpool.h` fork/join on qurt threads |
| `inc/lfrife.h`, `apps/lfrife.c` | arm64 API (dlopens `libcdsprpc.so`) |
| `apps/lfrife_bench.c` | accuracy and timing driver |

## Design

* NHWC fp16 activations with a 1 px zero halo; output channels on the 64 halfword lanes.
* IEEE hf/sf HVX ops return zeros on this device; only the qfloat ops work. A convolution broadcasts
  each input activation (rt form `vmpy(Vu.hf, Rt.hf)` -> qf16, scalar duplicated in both halves) against a
  64-lane weight vector and accumulates with `vadd_qf16` into 1-16 pixel accumulators (about 2.2-2.6
  cycles per 64-MAC vector per thread, measured). Bias initialises the accumulator; LeakyReLU(0.2) is
  `max(x, 0.2x)`; ResConv adds the residual in qf16 before the activation.
* ResConv (stride-1 3x3, 8 per block) uses Winograd F(2x2,3x3): 16 multiplications per 2x2 output tile instead of
  36. Weights are transformed at create time (U = G g G^T, fp16, `lay.uw`, [ob][16][ci][64], 1.8x the size
  of the direct weights, about 18 MB for the whole net). Per row of tiles the input is transformed with adds only
  (V = B^T d B, 64 channels per vector, qf16), then for each of the 16 positions the usual broadcast-MAC loop
  runs with up to 16 tiles as accumulators (the weight vector is reused 16 times, like the pixel tiles of
  `tile_n`), and Y = A^T M A plus bias, residual and LeakyReLU is applied on the qf16 products. Odd heights and
  widths use partial tiles (the spare row at the end of every tensor is read, never written). The direct
  kernel is still there (`-DLF_WINO=0`) for A/B checks. Stride-2 conv0/conv1 and the deconv stay direct.
* Weights stream from DRAM once per layer and the DSP does not hide the latency by itself: a software
  prefetch 16 weight vectors ahead (`LF_PREFETCH`, `__builtin_prefetch`) took the small blocks (few pixels per
  weight vector) from latency-bound to compute-bound, and sped up conv0/conv1/deconv too.
* ConvTranspose 4x4 s2 + PixelShuffle is two passes of 6 taps. Weights and bias are permuted at create
  time (`deconv_lane`) so the output lanes feed the vector scatter directly. The result is scattered
  (`scatter_rows`) into per-block fp16 planes, then bilinearly upsampled (`upsample_row`) and accumulated
  into the fp32 flow/mask planes. The last block writes flow_out/mask_out directly (sigmoid in `final_row`).
* The former scalar fp32 stages are HVX now: build_input (warp taps, 2x2 means), scatter, upsample, the final
  copy and a vector sigmoid. Only qf32 add/sub/mul, sf<->w conversions, integer ops and permutes are used
  (IEEE sf/hf arithmetic stays off the table). Offsets are exact fp32 integers below 2^24; the warp's
  per-pixel RGBA gather is scalar. `-fno-strict-aliasing` plus `LF_MEMBAR()` order vector stores
  against scalar loads. Block 0 assigns instead of accumulating, so there is no plane clear.
* Per block (scales 8,4,2,1): build input (downsample by the mean of the central 2x2, or warp in fp32
  then downsample for blocks 1-3), conv0, conv1, 8 ResConv, deconv, upsample. Warp coordinates are fp32
  (border clamp, align_corners). No integer or float division on the DSP (no libcalls).
* Threads: per-run qurt thread spawn, spin barriers, atomic work counter (rows x output blocks). At setup the
  skel reads `qurt_hvx_get_units() >> 8` (0x600 = 6 contexts here) and clamps nthreads to it and to
  `LFP_MAX`. Each worker must lock HVX and join through a CAS gate to get a dense tid; workers that fail
  or arrive late are left out, so the pool degrades instead of deadlocking. The real count is reported in
  `stats->nthreads` (`lfrife_threads()`). Timing is split into 24 groups (6 per block).

## Build and test

```
make weights                      # export blobs + goldens (needs the rife-npu venv)
make test-host                    # fp32 reference vs golden (max err 2e-3 px)
make test-hostemu                 # DSP network source on the host with the HVX emulation (slow)
make rife                         # liblfrife_skel.so + lfrife_bench
make push-rife push-data ADB="adb -s <serial>"
adb shell 'cd /data/local/tmp/lightforge-hvx && ./lfrife_bench run 6 30 pair000_320x192 pair002_320x192'
make build/test_ref_arm64         # CPU baseline, push and run test_ref_arm64 with --bench
```

## Results (razr fold, SM8845, v81, 320x192, fp16 weights, qf16 accumulation)

Accuracy against the fp32 PyTorch golden:

| pair | t | flow max err px | flow mean err px | pixels > 0.5 px | mask max err |
| --- | --- | --- | --- | --- | --- |
| 0 | .75 | 0.64 | 0.0135 | 0.02 % | 0.16 |
| 2 | .50 | 0.67 | 0.0142 | 0.02 % | 0.57 |
| 3 | .25 | 0.36 | 0.0159 | 0 | 0.21 |
| 11 | .875 | 0.22 | 0.0144 | 0 | 0.09 |

The mask maximum is the sigmoid of a small logit error in a few saturated edge pixels; mean mask error is 1e-3 to 7e-3.
Calibration flows are small (max about 3-4 px); larger motion has not been measured.

Timing (DSP-side p50, 30 iterations; wall time is about 1 ms more):

| path | ms |
| --- | --- |
| HVX, 1 thread (first version) | 150 |
| HVX, 4 threads (first version) | 56 |
| HVX, 6 threads | 21.5 (p90 22) |
| plain C fp32 on the phone CPU (1 thread, arm64) | 615 |
| Vulkan ncnn GPU (reference quoted by the task) | about 20-40 |

Vectorised scalar stages, then Winograd ResConv + weight prefetch (p50 over 30 iterations, 6 threads, DSP-side;
run-to-run noise about 1 ms, another process may use the phone):

| size | original ms | vectorised stages ms | + Winograd + prefetch ms |
| --- | --- | --- | --- |
| 320x192 | 48.3 | 37.6 | 21.5 |
| 512x288 | 111.8 | 83.5 | 49.5 |

Accuracy is unchanged by Winograd (flow mean 0.0133/0.0172/0.0167/0.0124 px for pairs 0/2/3/11, mask mean 1.6e-3 /
6.4e-3 / 2.3e-3 / 7.2e-4).

6-thread breakdown per block at 320x192, ms (build / conv0+conv1 / 8 ResConv / deconv / scatter / upsample):
block0 0.13/0.12/1.49/0.08/0.00/0.05, block1 0.41/0.23/1.77/0.20/0.01/0.17, block2 0.77/0.71/4.69/0.55/0.03/0.21,
block3 1.08/1.77/5.35/1.49/0.11/0.20. At 512x288: block0 0.20/0.26/2.71/0.20/0.01/0.09, block1
0.98/0.50/3.60/0.43/0.02/0.34, block2 1.84/1.84/10.04/1.35/0.07/0.46, block3 2.58/4.60/12.40/3.67/0.30/0.49.
Before Winograd the ResConv column was 3.8/4.6/8.9/11.0 (320x192) and 7.8/8.1/19.8/25.6 (512x288).
`lfrife_bench time <w> <h> <threads> <iters>` runs a synthetic pair for timing only.

Large motion (golden image 320x192 against the same image moved by (dx,dy), DSP vs the fp32 reference,
`lfrife_bench shift` + `test_ref --shift`):

| pair | shift px | ref mean flow0 | flow max | flow mean | mask mean | direct kernel: flow mean / max |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | (20,10) | -9.1 | 0.96 | 0.054 | 1.4e-2 | 0.054 / 1.16 |
| 2 | (40,20) | -18.6 | 2.40 | 0.109 | 2.1e-2 | 0.114 / 1.90 |
| 3 | (-30,15) | 13.4 | 1.20 | 0.085 | 1.9e-2 | 0.089 / 1.25 |
| 11 | (12,-6) | -5.6 | 0.96 | 0.035 | 1.2e-2 | 0.037 / 1.27 |

The error is larger than on the calibration pairs and grows with the motion, but the direct fp16 kernel shows
the same numbers, so it comes from the fp16 pipeline in general and not from Winograd.

Why not int16 or int8 MACs. `vmpyacc` int16 x int16 -> int32 pairs (k_floor mode 3, `probe11` kernel 4) cost the
same 2.0 pcycles per op as the qf16 `vmpy`+`vadd` pair (which dual-issue in one packet), and the 6-thread aggregate
is 102.8 against 99.6 MAC/pcycle, so there is no gain. `Wsf_vmpyacc_WsfVhfVhf` returns zeros (probe1: pair.lo/hi
all 0). `vrmpy` u8 x s8 is 3.5x faster (kernel 5), but fake-quantised in the fp32 reference it fails the accuracy
gate: int8 per-output-channel weights alone give flow mean 0.05 px (0.03 even when only the block-3 ResConvs are
quantised), int8 activations on top 0.10 px; 10-bit weights 0.011, 12-bit 0.003, 16 bit 4e-4. Hence
only 16-bit paths are accurate and those are as slow as qf16.

Remaining steps: (a) conv0/conv1 (stride 2) and the deconv at block 3, (b) the 96-channel layers of block 2 waste
25 % of the lanes (two pixels per vector would recover it), (c) the warp's scalar RGBA gather.

### Clock vote (`power_corner`)

`lfrife_config.power_corner` pins a DCVS voltage corner through `HAP_power_set` (DCVS v2 request, DCVS off,
100 us latency) for as long as the session is open; close hands the clocks back to DCVS. Corners measured on
the SM8845: 4 = 1248 MHz, 6 = 1594 MHz, 7 = 1709 MHz, 8 and above = 1862 MHz (the core maximum). Corners
above 8 (255 = maximum) also raise the bus clocks.

| 512x288, tight loop (bench) | ms/call | W busy | mJ/call total |
|---|---|---|---|
| DCVS (0) | 48.2 | 2.2 | 106-108 |
| 6 | 54.1 | 1.9 | 100 |
| 8 | 46.1 | 2.5 | 113 |
| 255 | 45.0 | 2.9-3.7 | 131-167 |

In a tight loop DCVS already reaches 1862 MHz. Inside an app, where the DSP idles between calls while the CPU
composes, DCVS settles near 1085 MHz: the hold-to-slow-motion preview (320-px net, 1280 output) went from
27-30 to 35 frames/s with corner 8 at about the same energy per frame. Corner 255 buys ~2 % more for 25-55 %
more energy, so the app uses 8.

## Using it from an app

```c
lfrife_config cfg = {.skel_dir = nativeLibraryDir, .weights_path = ".../rife46_hvx.bin",
                     .width = 320, .height = 192, .nthreads = 6, .qos = -1};
lfrife *r; int rc = lfrife_open(&cfg, &r);       // < 0: fall back to the GPU path
rc = lfrife_flow(r, rgba0, rgba1, stride, 320, 192, t, flow /*4*h*w*/, mask /*h*w*/);
lfrife_close(r);
```

`lfrife_open` dlopens `libcdsprpc.so`, requests an unsigned PD and returns an error code (never crashes)
when anything is missing. One handle per thread of use.

Vendoring into an Android Gradle module:

* CMake: `add_subdirectory(lightforge-hvx)` is not provided; compile `apps/lfrife.c` with `-I inc` into a
  shared or static library (links `-ldl -lm` only) in the module's `CMakeLists.txt`.
* Put `liblfrife_skel.so` (built by `make build/liblfrife_skel.so`, a Hexagon ELF) in
  `src/main/jniLibs/arm64-v8a/` with `android:extractNativeLibs="true"` so it exists on disk; pass the
  app's `nativeLibraryDir` as `skel_dir`. Set `DSP_LIBRARY_PATH=<nativeLibraryDir>` with `setenv` before
  `lfrife_open` so the DSP loader finds it.
* Ship `rife46_hvx.bin` (11.5 MB) as an asset and copy it to the cache, or pass it via `weights_blob`.
* Remaining for integration: CMake wiring, the CPU warp from flow+mask, 512x288/640x384 sizes (the code
  is size-generic but only 320x192 was validated), and a thread-count policy while the app uses the GPU.
