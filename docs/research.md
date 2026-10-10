# Running RIFE v4.6 on the Hexagon cDSP with HVX: a clean-room feasibility study

Date: 2026-10-10. Target: motorola razr fold, SM8845 "canoe", Hexagon v81, Android 16 (SDK 36), app installed by the user (no root).
Constraint: open source only. No QNN, SNPE, Hexagon SDK headers or libraries, QAIC or hexagon-clang.

Labels used below:
- **[D]** documented in public source or docs.
- **[V]** verified on this host or on the phone during this research.
- **[I]** inferred. Treat it as a hypothesis until tested on the device.

---

## TL;DR

- **Toolchain (OK).** Upstream clang 22 builds v73/v75/v79/v81 HVX-128B code. It also links a Hexagon ELF shared object with lld, using `-shared -nostdlib`. The HVX intrinsic headers ship with clang under Apache-2.0 WITH LLVM-exception. [V]
- **FastRPC ABI (OK).** quic/fastrpc is BSD-3-Clause. It documents everything the APPS side needs: `remote_handle64_open/invoke`, `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE)`, `REMOTE_SCALARS_MAKEX`, `remote_arg`, the URI format, and the skel open/close convention. That convention is method 0 = open, method 1 = close, and user methods start at 2. [D]
- **Device (OK so far).** `libcdsprpc.so` is listed in `/vendor/etc/public.libraries.txt`. `/dev/fastrpc-cdsp` is `crw-rw-r--` with the label `vendor_qdsp_device`. [V] Normal apps already reach the cDSP this way: ONNX Runtime QNN EP and llama.cpp both use unsigned PD. [I/D]
- **HMX (NOT available to us).** Upstream LLVM 22 has no `-mhmx` flag and rejects `mxmem` instructions. [V] Every open project that uses HMX, including llama.cpp ggml-hexagon and hexagon-mlir, builds with Qualcomm's proprietary toolchain or the Hexagon SDK. Under our constraint we get **HVX only**.
- **Biggest risk.** Nobody has publicly shown a skel built with upstream LLVM and no SDK running on v73+. tinygrad did it on v65 (SD845). Halide's notes say that devices newer than sm8350 only run skels "built with SDK ≥ 4.2.02", and nobody has said why. **Run the "hello skel" test on the phone first.**
- **Performance.** HVX alone probably gives int8 RIFE at 320 px in about 10–25 ms, and fp16 in about 25–60 ms. That is roughly the same as the Adreno path (20–40 ms), not a clear win. The only published NPU number is QNN with HMX, which is off-limits: RIFE at 360p INT8 takes 14.7 ms on v75, and FP16 is over 33 ms. RIFE is about 95 % memory-bound operations (resize, grid-sample, elementwise), which favours neither HVX nor HMX.

---

## 1. Upstream LLVM and the Hexagon target

| Item | Finding | Evidence |
|---|---|---|
| CPU versions | clang 22.1.3 (system) accepts `--target=hexagon -mv73/-mv75/-mv79/-mv81 -mhvx -mhvx-length=128b` | [V] all four return rc=0 |
| HVX intrinsics | `hexagon_types.h`, `hexagon_protos.h`, `hvx_hexagon_protos.h` and `hexagon_circ_brev_intrinsics.h` are in clang's resource dir. Their headers say "Part of the LLVM Project … Apache-2.0 WITH LLVM-exception". `hvx_hexagon_protos.h` has `__HVX_ARCH__ >= 73/79/81` sections and IEEE-fp parts gated by `__HVX_IEEE_FP__` (`-mhvx-ieee-fp`) | [V] |
| Codegen check | `Q6_Vw_vrmpyacc_VwVubRb` gives `v0.w += vrmpy(v0.ub,r0.b)`. `__builtin_HEXAGON_V6_vmpy_hf_hf_128B` gives `v0.hf = vmpy(v0.hf,v1.hf)` with `-mhvx-ieee-fp` | [V] llvm-objdump |
| Shared object | `clang --target=hexagon -mv81 -mhvx -mhvx-length=128b -fPIC -shared -nostdlib -fuse-ld=lld` produces "ELF 32-bit LSB shared object, QUALCOMM DSP6" | [V] |
| HMX | `-mhmx` is "unknown argument". The inline asm `activation.hf = mxmem(...)` is "unrecognized instruction" | [V] |
| Open libc / runtime | **quic/toolchain_for_hexagon** (BSD-3) builds clang/lld "from exclusively public open source repos". It ships musl for `hexagon-unknown-linux-musl` and picolibc for `hexagon-unknown-none-elf`. LLVM compiler-rt builtins support Hexagon (`__hexagon_divsf3` and similar) | [D] https://github.com/quic/toolchain_for_hexagon |

### What a skel links against inside `fastrpc_shell`

**[I] Strong evidence:**
- The shell (`fastrpc_shell_3` for a signed PD, `fastrpc_shell_unsigned_3` for an unsigned PD; names from `fastrpc_apps_user.c`) contains the QuRT user runtime, an rtld (`dlopen`/`dlsym`/`dlopenbuf`), libc, and the HAP_* and qurt_* APIs.
- When the skel is `dlopen`ed, its undefined symbols are resolved against the shell.
- tinygrad builds its DSP kernels with `clang --target=hexagon -mcpu=hexagonv65 -fuse-ld=lld -nostdlib -mhvx=v65 -mhvx-length=128b -shared -ffreestanding`. It uses a linker script that 4 KiB-aligns every section; tinygrad's comment says this "fixes invoke problem". It calls `HAP_power_set`, `HAP_mmap`, `HAP_munmap` and `HAP_perf_get_time_us` without linking any library. The prototypes and struct layouts are re-declared inline in tinygrad's MIT-licensed source (`tinygrad/runtime/ops_dsp.py`).
- Halide's `known_symbols.cpp` also expects `qurt_hvx_lock/unlock` and `__hexagon_{muldf3,divdf3,adddf3,divsf3,udivdi3,udivsi3}` to come from the environment.

**Recommendation:**
- Use `-nostdlib -ffreestanding`.
- Link compiler-rt builtins statically.
- Write your own `memcpy`/`memset`.
- Import only a small, clean-room list of shell symbols. Write their prototypes from public docs and observed ABI, never from SDK headers:
  - `qurt_thread_create/join`, `qurt_hvx_lock/unlock`, `qurt_hvx_get_units`
  - `HAP_power_set`
  - optionally `HAP_compute_res_*` for VTCM
  - `HAP_perf_get_time_us`
  - `HAP_debug_v2`/FARF for logging, if needed.

Copying SDK headers would break the clean-room rule. The struct layouts (`HAP_power_request_t`, `compute_res_attr_t`) have to be rebuilt from public documentation or from tinygrad's MIT re-declaration.

---

## 2. quic/fastrpc: license and ABI coverage

- **License: BSD-3-Clause.** `LICENSE.txt` is "Copyright (c) 2021, Qualcomm Innovation Center". 138 files carry `SPDX: BSD-3-Clause` and 4 carry `BSD-3-Clause-Clear`. [V] https://github.com/quic/fastrpc
- **APPS side** (`inc/remote.h`) [D]:
  - `remote_handle64_open(const char* uri, remote_handle64*)`, `remote_handle64_invoke(h, uint32_t sc, remote_arg*)`, `remote_handle64_close`, `remote_session_control(req, data, len)`, `remote_handle_control(DSPRPC_GET_DSP_INFO, …)`.
  - `REMOTE_SCALARS_MAKEX(attr,method,nIn,nOut,noIn,noOut)` packs to `attr<<29 | (method&0x1f)<<24 | (nIn&0xff)<<16 | (nOut&0xff)<<8 | (noIn&0xf)<<4 | (noOut&0xf)`.
  - 32-bit `remote_arg` is a union of `{void* pv; size_t nLen}`, `remote_handle`, `remote_handle64` and `{int32 fd; uint32 offset}`. On the 32-bit DSP each entry is 8 bytes.
- **URI** (`idl/remote.idl` and the generated headers) [D]: `"file:///<lib>.so?<iface>_skel_handle_invoke&_modver=1.0" "&_dom=cdsp"` and optionally `"&_session=N"`.
  - The first unnamed key is the skel entry symbol.
  - Keys starting with `_` are reserved. Unknown keys are forwarded.
  - llama.cpp logs exactly this: `file:///libggml-htp-v79.so?htp_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp&_session=0`.
- **Skel entry point and open/close** [D, from `src/mod_table.c`, the APPS-side twin of the DSP module table]:
  - Signature: `int <iface>_skel_handle_invoke(remote_handle64 h, uint32_t sc, remote_arg* pra)`.
  - **open**: `handle_invoke(0, MAKEX(0, 0, 2,0, 0,1), args)`, where `args[0]={&len,4}`, `args[1]={uri,len}` (NUL included), and `args[2].h64` is the out handle.
  - **close**: `handle_invoke(0, MAKEX(0, 1, 0,0, 1,0), {h64})`.
  - The generated stubs number user methods from `_mid = 2` (for example `adsp_current_process1_stub.c`).
  - Generated skels dispatch with `switch (REMOTE_SCALARS_METHOD(sc))` (`apps_mem_skel.c`, `adspmsgd_apps_skel.c`).
  - tinygrad's hand-written DSP entry is `int entry(unsigned long long handle, unsigned int sc, remote_arg* pra)`.
  - [I] The DSP-side shell uses the same module-table logic. tinygrad's raw-ioctl open/close (method 0 with ins=2, method 1 on handle 0) agrees.
- **What is missing:** QAIC itself. The marshalling conventions that QAIC generates for complex types (sequences, structs, in/rout ordering) are only visible through the generated `*_stub.c`/`*_skel.c` files in the repo. That is enough for a hand-written flat ABI: one input buffer with the parameters plus a few `dma` fd args, or one rout buffer.
- **Zero-copy:** `rpcmem_alloc`/`fastrpc_mmap(domain, fd, va, 0, size, FASTRPC_MAP_FD…)` (llama.cpp uses these) or dma-buf fds passed as `remote_arg.dma`. On the DSP side, `HAP_mmap(fd)` (tinygrad) or fd args that FastRPC maps automatically. The phone has `/dev/dma_heap/system` and `qcom,system` [V].

---

## 3. Unsigned PD from a regular Android app

- **Requesting it** [D, `remote.h` and `fastrpc_apps_user.c`]: call `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &(struct remote_rpc_control_unsigned_module){.domain=CDSP_DOMAIN_ID(3), .enable=1}, sizeof …)` **before** the first `remote_handle64_open`.
  - The library then sets `FASTRPC_MODE_UNSIGNED_MODULE (0x8)` in the process attributes and loads `fastrpc_shell_unsigned_3`.
  - Without it, an untrusted app gets `ECONNREFUSED` and this log line: *"untrusted app trying to offload to signed remote process … Try offloading to unsignedPD using remote_session_control"*.
  - `_session=` is only needed for multiple sessions (`FASTRPC_RESERVE_NEW_SESSION`).
  - Check capability first with `remote_handle_control(DSPRPC_GET_DSP_INFO)`. Useful attributes: `UNSIGNED_PD_SUPPORT`, `HVX_SUPPORT_128B` (the count), `VTCM_PAGE`, `VTCM_COUNT`, `ARCH_VER`, `HMX_SUPPORT_*`.
- **Precedent** [D]: llama.cpp `ggml-hexagon.cpp:4838` always enables unsigned PD. In it, the DSP code runs these calls and uses the results: `HAP_compute_res_acquire` (VTCM, `attr_set_hmx_param(1)`), `qurt_hvx_get_units()`, `qurt_sysenv_get_max_hw_threads`, `HAP_mmap2` and DMA.
  - [I] On v73+ an unsigned PD can therefore get VTCM, multiple HVX threads, and even HMX through the compute-resource manager.
  - That contradicts older SDK-era advice that "VTCM via CRM requires signed PD" (some Qualcomm forum and community posts say this). Treat that advice as obsolete on v73+, but verify on the device.
  - The other documented unsigned-PD limit is that the initial PD memory size cannot be changed. `fastrpc_apps_user.c` says "Setting user PD initial memory length is not supported for unsigned PD".
  - [I] Common folklore adds: no cache locking, capped thread priority, no access to some system services. Not verified here.
- **SELinux and node** [V]: `/dev/fastrpc-cdsp` is `crw-rw-r-- system system u:object_r:vendor_qdsp_device:s0`. The `-secure` nodes are `vendor_xdsp_device` with mode 0644.
  - [I] Qualcomm's vendor sepolicy grants `appdomain` access to `vendor_qdsp_device`. This is the same path ONNX Runtime QNN EP and the Qualcomm AI Hub apps use from normal apps. Not yet confirmed with an `untrusted_app` open on this phone.
- **Library access** [V]: `libcdsprpc.so` and `libadsprpc.so` are both in `/vendor/etc/public.libraries.txt` and `/vendor/lib64`. Apps targeting SDK 31+ must declare `<uses-native-library android:name="libcdsprpc.so" android:required="false"/>` and `dlopen("libcdsprpc.so")`, as Halide's `libadsprpc_shim.cpp` does. Do not link against the library at build time.
- **Where the DSP loads the skel from** [D, `apps_std_imp.c`]:
  - The DSP asks the **APPS process** to open the file through reverse RPC (`apps_std_fopen_with_env(ADSP_LIBRARY_PATH, ";", name…)`).
  - The search path is `DSP_LIBRARY_PATH` (preferred), or else `ADSP_LIBRARY_PATH`. The `;`-separated user list gets the default `DSP_SEARCH_PATH` (`/vendor/dsp/cdsp`, …) appended.
  - [I] So any path that **our app** can read works. Call `setenv("ADSP_LIBRARY_PATH", "<nativeLibraryDir>;…")` before the first open. That requires extracted native libs (`android:extractNativeLibs="true"` / `packaging.jniLibs.useLegacyPackaging = true`). Alternatively, copy the `.so` to `filesDir` at startup and point the variable there.
  - tinygrad serves the file through its own listener in the same way.
  - `/vendor/dsp` is not readable by `shell` on this phone [V], which does not matter for us.

---

## 4. Precedents

| Project | Open? | Without SDK? | HVX/HMX | Unsigned PD / app |
|---|---|---|---|---|
| **llama.cpp ggml-hexagon** (`ggml/src/ggml-hexagon`) | MIT | **No.** CMake `FATAL_ERROR` without `HEXAGON_SDK_ROOT`. It uses `hexagon_fun.cmake`, `build_idl` (QAIC), SDK `incs/`, QuRT and HAP headers, and the Qualcomm Hexagon tools. Docker image `ghcr.io/snapdragon-toolchain/arm64-android` includes Hexagon SDK 6.6 | Hand-written HVX (`hvx-*.h`) plus HMX through inline asm `mxmem`/`acc` (`hmx-utils.h`). Skels for v73/v75/v79/v81 | Yes, unsigned PD. Run from `adb shell` with `ADSP_LIBRARY_PATH=./lib`, not shown inside an app |
| **qualcomm/hexagon-mlir** | BSD-3 | **No.** The user guide requires Hexagon SDK 6.4.0.2, Hexagon Tools and HexKL | Triton/PyTorch to HVX, plus convs/matmuls on "a dedicated engine" (HMX) | Device runs go through SDK tooling |
| **Halide Hexagon backend** | MIT | **Partly.** Generated pipelines are compiled by upstream LLVM. The DSP-side glue `libhalide_hexagon_remote_skel.so` is a prebuilt binary checked into `src/runtime/hexagon_remote/bin/v65`, built with the SDK (Makefile needs `HEXAGON_SDK_ROOT`, `HEXAGON_QAIC`, hexagon-clang). At run time it `dlopenbuf`s the LLVM-built pipeline inside the PD | HVX only | Loads libcdsprpc through a `dlopen` shim. The Makefile warns "Devices >sm8350 will only run skel shared objects that have been built with SDK version 4.2.02 or newer" |
| **tinygrad `ops_dsp.py`** | MIT | **Yes.** Upstream clang/lld, no SDK, raw `/dev/adsprpc-smd` ioctls, its own reverse-RPC listener | Plain C / HVX (v65) | Ran on comma 3/3X (SD845, v65) as a privileged process. Loads `fastrpc_shell_3` itself. Not an untrusted app, not v73+ |
| **geohot/openhexagon** | — | Old (2010s) attempt at an open Hexagon toolchain. Historical | — | — |

**Is HMX usable without QNN?**
- Yes, technically. llama.cpp drives HMX directly from a FastRPC skel with no QNN.
- But it needs the **proprietary assembler/compiler** for the `mxmem`/`acc` instructions.
- [D] HMX is absent from upstream LLVM. [V] No public ISA document for HMX was found. The encodings could in principle be emitted as raw `.word` values, but that would mean reverse-engineering Qualcomm's binaries, which falls outside a clean-room effort.

**Conclusion: HVX only.**

---

## 5. Throughput estimate

Hardware facts:
- HVX: 128-byte vectors, 32 vector registers per context.
- The v73 PRM's example has **4 HVX contexts** [D] (chipsandcheese, citing the v73 PRM).
- The real count for our PD comes from `qurt_hvx_get_units()` or the `HVX_SUPPORT_128B` capability at run time.
- The cDSP clock is not published for SM8845. Older parts ran at 576 MHz–1 GHz. Assume about 1.0–1.4 GHz [I].

Per-instruction work [I, from instruction semantics]:
- `vrmpy` (u8×s8 → 32 × int32, 4-way dot) = 128 int8 MACs per instruction. At 1–2 multiply-class instructions per packet, that is **128–256 int8 MAC/cycle/context**.
- int16 `vmpa`/`vdmpy` ≈ 64–128 MAC/cycle.
- fp16 `vmpy hf` = 64 lanes, but the multiply and add are separate on qf16/IEEE paths. That gives **≈32–64 fp16 MAC/cycle/context**.

Peak, with 4 contexts at 1.2 GHz:

| Precision | Peak | Realistic for convs (30–50 %, VLIW scheduling, padding, im2col/border handling, DMA stalls) |
|---|---|---|
| int8 | 0.6–1.2 TMAC/s | ≈0.2–0.5 TMAC/s |
| fp16 | 0.15–0.3 TMAC/s | ≈0.06–0.15 TMAC/s |

RIFE v4.6 cost:
- Compute scales with pixel count. 6–10 GMAC at 512×288–640×384 works out to roughly **2.5–3.5 GMAC at 320×180–320×192**.
- Convs:
  - int8: 2.5–3.5 GMAC ÷ 0.2–0.5 TMAC/s ≈ **5–18 ms**
  - fp16: ≈ **17–60 ms**
- Then add the memory-bound share. ANVIL profiled RIFE on v75 and found convs are only 5 % of cycles; resize, grid-sample, div, add, mul, concat and PReLU make up 95 %.
  - On HVX, bilinear warp needs `vgather`, which requires VTCM. [I] That makes VTCM availability in unsigned PD critical.
- FastRPC round trip: about 0.3–1 ms per invoke [I]. Use one invoke per pass and keep buffers mapped persistently.

**Realistic end-to-end per RIFE pass at about 320 px:**

| Path | Time |
|---|---|
| HVX int8 | ≈10–25 ms |
| HVX fp16 | ≈25–60 ms |
| Current Adreno 840 | 20–40 ms |
| Published QNN+HMX (forbidden) | 14.7 ms (int8, 360p, v75); over 33 ms in fp16 |

So the DSP mainly offers offload and power savings plus a modest speedup, and only with int8. With fp16 it is unlikely to beat the GPU. Source for the QNN numbers: ANVIL, https://arxiv.org/html/2603.26835v2, Table I and the per-op profile.

---

## Uncertainties and the first experiments to run

1. **Hello-skel on v81 (go/no-go).** Build an upstream-LLVM skel (`-nostdlib`, sections 4 KiB-aligned like tinygrad) that exports `x_skel_handle_invoke` and returns `42` from method 2. Load it from the app with unsigned PD. If it fails, the skel probably needs ELF properties that SDK ≥ 4.2.02 adds (the Halide warning). Diff the ELF headers, dynamic tags and notes against llama.cpp's published `libggml-htp-v81.so`, without copying it.
2. Confirm that `untrusted_app` can open `/dev/fastrpc-cdsp`. Read `DSPRPC_GET_DSP_INFO` (unsigned support, HVX count, VTCM page and count).
3. Confirm that loading through `ADSP_LIBRARY_PATH=nativeLibraryDir` works from an app.
4. Measure `HAP_compute_res_acquire` for VTCM in unsigned PD, `qurt_hvx_get_units`, and the clock via `HAP_power_set` DCVS.
5. Micro-benchmark `vrmpy` and hf `vmpy` loops to replace the inferred MAC/cycle figures.

## Sources

- quic/fastrpc (BSD-3): https://github.com/quic/fastrpc. Files: `inc/remote.h`, `src/mod_table.c`, `src/fastrpc_apps_user.c`, `src/apps_std_imp.c`, `idl/remote.idl`
- LLVM clang Hexagon headers: clang resource dir `include/hvx_hexagon_protos.h` (Apache-2.0 WITH LLVM-exception)
- quic/toolchain_for_hexagon (BSD-3, musl + picolibc): https://github.com/quic/toolchain_for_hexagon
- llama.cpp ggml-hexagon: https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-hexagon and `docs/backend/snapdragon`
- qualcomm/hexagon-mlir: https://github.com/qualcomm/hexagon-mlir (`docs/user-guide.md`), paper https://arxiv.org/abs/2602.19762
- Halide hexagon_remote: https://github.com/halide/Halide/tree/main/src/runtime/hexagon_remote
- tinygrad DSP runtime: https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/ops_dsp.py
- geohot/openhexagon: https://github.com/geohot/openhexagon
- Chips and Cheese, "Qualcomm's Hexagon DSP, and now, NPU": https://chipsandcheese.com/p/qualcomms-hexagon-dsp-and-now-npu
- ANVIL (RIFE on HTP v73/v75 latency and op profile): https://arxiv.org/html/2603.26835v2
- Qualcomm HVX docs (memory/VTCM): https://docs.qualcomm.com/bundle/publicresource/80-N2040-61/topics/memory.md
