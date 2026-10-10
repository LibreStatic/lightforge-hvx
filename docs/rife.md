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
* ConvTranspose 4x4 s2 + PixelShuffle is two passes of 6 taps (lane = ((pv*2+i)*2+j)*5+ch), scattered
  to fp32 planes, then bilinear upsampled and accumulated into the fp32 flow/mask planes.
* Per block (scales 8,4,2,1): build input (downsample by the mean of the central 2x2, or warp in fp32
  then downsample for blocks 1-3), conv0, conv1, 8 ResConv, deconv, upsample. Warp coordinates are fp32
  (border clamp, align_corners). No integer or float division on the DSP (no libcalls).
* Threads: per-run qurt thread spawn, spin barriers, atomic work counter (rows x output blocks).

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
| 0 | .75 | 0.57 | 0.013 | 0.01 % | 0.19 |
| 2 | .50 | 1.07 | 0.016 | 0.47 % | 0.50 |
| 3 | .25 | 0.39 | 0.016 | 0 | 0.27 |
| 11 | .875 | 0.24 | 0.014 | 0 | 0.07 |

The mask maximum is the sigmoid of a small logit error in a few saturated edge pixels; mean mask error is 1e-3 to 7e-3.
Calibration flows are small (max about 3-4 px); larger motion has not been measured.

Timing (DSP-side p50, 30 iterations; wall time is about 1 ms more):

| path | ms |
| --- | --- |
| HVX, 1 thread | 150 |
| HVX, 4 threads | 56 |
| HVX, 6 threads | 48 (p90 53; first call 50-67) |
| plain C fp32 on the phone CPU (1 thread, arm64) | 615 |
| Vulkan ncnn GPU (reference quoted by the task) | about 20-40 |

6-thread breakdown per block, ms (build X / conv0+conv1 / 8 ResConv / deconv+upsample):
block0 0.12/0.28/3.7/1.6, block1 0.65/0.35/4.4/1.9, block2 2.3/0.80/8.7/2.5, block3 3.9/1.8/11.0/3.3.
The ResConv stages are 58 % of the time and run at the measured qf16 mpy+add rate, so the next steps are
(a) packing two pixels per vector for the 32/48/64-channel layers, (b) vectorised warp/downsample and
upsample (about 14 ms of scalar fp32 work today), (c) better balance across 6 threads.

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
