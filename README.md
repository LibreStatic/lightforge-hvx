# lightforge-hvx

RIFE frame interpolation on the Qualcomm Hexagon compute DSP (cDSP), built only with open-source
tools.

`lightforge-hvx` runs your own HVX vector code on the cDSP of a retail, non-rooted Snapdragon phone
from an ordinary installed app. It does not use the Hexagon SDK, QAIC, QNN or SNPE. The DSP code is
compiled with upstream `clang --target=hexagon` and `ld.lld`, loaded into an unsigned protection
domain through the phone's public `libcdsprpc.so`, and driven over FastRPC with a hand-written
protocol.

On top of that, it implements the flow network of [RIFE](https://github.com/hzwer/Practical-RIFE)
v4.6 in fp16 HVX. It was written for [Lightforge](https://play.google.com/store/apps/details?id=com.librestatic.lightforge),
which interpolates frames for slow-motion video.

> **Status: experimental.** Everything was validated on one device, a motorola razr fold (SM8845,
> Android 16). Lightforge does not use this backend yet; it ships the Vulkan (ncnn) path.

## Results

The flow network matches the PyTorch reference to within 2e-3 px. A 320x192 frame pair takes these
times on the DSP (p50):

| Path | ms |
| --- | --- |
| HVX, 6 threads | 48 |
| HVX, 1 thread | 150 |
| Plain C fp32 on one arm64 core | 615 |

The cDSP feasibility spike also found:

- A skel built with clang 22 loads into an unsigned user PD, both from `adb shell` and from a
  non-debuggable `untrusted_app` with targetSdk 36.
- HVX results are bit-exact. Multithreaded HVX (6 contexts) and 8 MiB of VTCM work from an app.
- A FastRPC call costs about 400 µs at p50, and the first open takes 50–230 ms.

## Repository layout

| Path | What |
| --- | --- |
| `dsp/` | Hexagon side: the skels (`lfhvx_skel.c` probes and benchmarks, `lfrife_skel.c` RIFE), the RIFE network (`lfrife_net.c`), a thread pool on QuRT threads and a host emulation of the HVX intrinsics (`hvx_emu.h`). |
| `inc/` | Wire protocols, the public arm64 API (`lfrife.h`) and the weight blob format. |
| `apps/` | arm64 side: the `lfrife` library and command-line benchmarks. |
| `ref/` | A portable C11 fp32 reference of the RIFE flow network. |
| `tests/` | Golden inputs and outputs (synthetic test patterns) and the host emulation test. |
| `tools/export_weights.py` | Converts the RIFE weights to the fp32 and HVX blobs and regenerates the goldens. |
| `android-test/` | A minimal test APK, built without Gradle, that proves loading from an installed app. |
| `docs/` | [RIFE port design and numbers](docs/rife.md), the [cDSP spike report](docs/cdsp-spike.md) and the [research notes](docs/research.md). |

## Requirements

- A Snapdragon phone whose `/vendor/etc/public.libraries.txt` lists `libcdsprpc.so` and whose cDSP
  has an unsigned PD shell (`fastrpc_shell_unsigned_3`). Only the SM8845 was tested.
- LLVM with the Hexagon target (`clang --target=hexagon`, `ld.lld`). Tested with LLVM 22.
- Android NDK r28 or newer, with `NDK` pointing to it.
- `adb`. Set `ADB="adb -s <serial>"` when several devices are connected.
- Optional: Python with PyTorch, to export weights and regenerate goldens.

## Quick start

```sh
make test-host        # fp32 reference against the goldens, on the host
make test-hostemu     # the DSP network source on the host with emulated HVX (slow)

make                  # probe skel + benchmark
make push run         # copy to /data/local/tmp/lightforge-hvx and run

make weights          # export the RIFE blobs (needs the weights and PyTorch)
make rife             # liblfrife_skel.so + lfrife_bench
make push-rife push-data
```

`make HEX_CPU=hexagonv81` targets another HVX version; v68, v73 (the default), v79 and v81 all ran
on the test device. See [docs/rife.md](docs/rife.md) for the benchmark options and for how to use
`lfrife` from an Android app.

## Using it from an app

```c
lfrife_config cfg = {.skel_dir = nativeLibraryDir, .weights_path = ".../rife46_hvx.bin",
                     .width = 320, .height = 192, .nthreads = 6, .qos = -1};
lfrife *r;
if (lfrife_open(&cfg, &r) < 0) { /* no usable cDSP: fall back to the GPU or CPU */ }
lfrife_flow(r, rgba0, rgba1, stride, 320, 192, t, flow, mask);
lfrife_close(r);
```

`lfrife_open` returns an error code instead of crashing when the DSP, the unsigned PD or the skel
is not available, so callers can always fall back.

## Clean-room rules

No Qualcomm-licensed SDK header, library or tool was used, read or linked. The OS symbols the DSP
code needs are declared by hand and resolved by the user-PD shell at load time. FastRPC declarations
on the arm64 side come from the BSD-3 [quic/fastrpc](https://github.com/quic/fastrpc) sources. The
[spike report](docs/cdsp-spike.md#clean-room-rules-followed) lists every source consulted, and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) holds their notices.

## License

[MIT](LICENSE). Third-party notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
