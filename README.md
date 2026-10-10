# lightforge-hvx: our own HVX code on the cDSP, open-source tooling only

Feasibility spike. The question was whether we can run our own Hexagon HVX code
on the Qualcomm compute DSP (cDSP) of a retail, non-rooted phone, and time it,
using only open-source tools.

**Result: yes.** A skel built with upstream clang 22 and ld.lld loads into an
unsigned user PD on the cDSP. It is driven from an arm64 binary run from
`adb shell`, its results are bit-exact, and it reports DSP cycle counts.

## Clean-room rules followed

- Not used: Hexagon SDK, QAIC, QNN, SNPE, or any Qualcomm-licensed header,
  library or tool.
- DSP side: system `clang --target=hexagon` (LLVM 22.1.3) and `ld.lld`. The
  code is freestanding (`-nostdlib`). We declare every OS symbol ourselves
  (`qurt_hvx_lock`, `qurt_hvx_unlock`, `qurt_hvx_get_mode`, `HAP_debug_v2`),
  and the user-PD shell resolves them at `dlopen` time.
- APPS side: Android NDK r28 clang. The binary `dlopen`s the phone's own
  `/vendor/lib64/libcdsprpc.so`, which is a public vendor library listed in
  `/vendor/etc/public.libraries.txt`. Its symbols (`remote_session_control`,
  `remote_handle64_open/invoke/close/control`, `rpcmem_alloc/free`) and the
  structs/constants are re-declared from the BSD-3
  [quic/fastrpc](https://github.com/quic/fastrpc) sources (commit `a56e9d4`,
  `LICENSE.txt` checked: BSD 3-Clause, Qualcomm Innovation Center):
  `inc/remote.h`, `inc/rpcmem.h`, `inc/HAP_debug.h`, `src/mod_table.c`.
- Halide's MIT `src/runtime/qurt_hvx.cpp` and `mini_qurt.h` (commit `ee9f17a`)
  were used as a reference for the `qurt_hvx_lock(QURT_HVX_MODE_128B = 1)`
  convention. Halide's committed prebuilt skel was not used. We only compared
  its ELF program headers with ours while debugging the loader (see below).

## Device

motorola razr fold (`blanc`), SM8845 (`ro.board.platform=canoe`), Android 16
(`W3WB36.36-123-2`, user/release-keys), not rooted. Only `/dev/fastrpc-cdsp` is
relevant here (`crw-rw-r-- system system`). The shell domain cannot open it
directly. libcdsprpc falls back to the `vendor.qti.hardware.dsp.IDspService`
HAL ("open thru HAL"), and the HAL hands back an fd. `/vendor/dsp` cannot be
listed from shell (SELinux `adsprpcd_file`), but `cdsprpcd` reads
`/vendor/dsp/cdsp/fastrpc_shell_unsigned_3` on our behalf.

## Layout

| Path | What |
|---|---|
| `inc/lfhvx_proto.h` | Hand-written wire protocol (scalars macros, method ids, arg structs). Replaces a QAIC IDL. |
| `dsp/lfhvx_skel.c` | Hexagon skel: `lfhvx_skel_handle_invoke`, open/close, `add`, HVX/scalar u8 saturating add, HVX int16 `vdmpy` MAC, info. |
| `apps/lfhvx_bench.c` | arm64 driver: unsigned-PD request, open, correctness checks, latency and throughput benchmarks. |
| `Makefile` | Build, push, run. |

## Build and run

```sh
make                 # build/liblfhvx_skel.so (hexagon) + build/lfhvx_bench (arm64)
make push            # adb push to /data/local/tmp/lightforge-hvx
make run             # run the benchmark from adb shell
make HEX_CPU=hexagonv81 ...   # v68, v73 (default), v79 and v81 all load and run
make clean-device    # rm -rf /data/local/tmp/lightforge-hvx on the phone
```

These are the exact commands that `make` runs:

```sh
clang --target=hexagon -mcpu=hexagonv73 -mhvx -mhvx-length=128b -O2 -fPIC -G0 \
  -ffreestanding -fno-builtin -fno-stack-protector -nostdlib -fvisibility=hidden \
  -Wall -Wextra -Wno-unused-parameter -c dsp/lfhvx_skel.c -o build/lfhvx_skel.o
ld.lld -shared -soname liblfhvx_skel.so --hash-style=sysv -z max-page-size=4096 \
  -z norelro --no-rosegment -z separate-loadable-segments \
  -o build/liblfhvx_skel.so build/lfhvx_skel.o
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang \
  -O2 -Wall -Wextra -o build/lfhvx_bench apps/lfhvx_bench.c -ldl
```

Driver options: `--qos 0|1|2|3` (`DSPRPC_CONTROL_LATENCY`: off, PM QoS,
adaptive, poll), `--malloc` (plain heap buffers instead of `rpcmem_alloc`),
`--size N`, `--dom cdsp|cdsp1`, `--signed` (skip the unsigned-module request),
`--libdir D`. Set `LFHVX_LOG_WARMUP=1` when debugging loader failures (see
"DSP logs").

### How it works

1. `setenv("ADSP_LIBRARY_PATH", "/data/local/tmp/lightforge-hvx;/vendor/dsp/cdsp;...")`
   is called before libcdsprpc is loaded. The DSP loader fetches the skel
   through reverse-RPC `apps_std` file reads from these paths.
2. `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE /*2*/, {domain=3, enable=1})`.
3. `remote_handle64_open("file:///liblfhvx_skel.so?lfhvx_skel_handle_invoke&_modver=1.0&_dom=cdsp")`.
   The DSP-side module table calls our `handle_invoke` with method 0
   (`in[0]=int32 len`, `in[1]=uri`, out-handle `pra[2].h64`). Close is method 1.
   User methods start at 2. This protocol comes from `open_mod_handle_open/close`
   in quic/fastrpc `src/mod_table.c`.
4. `remote_handle64_invoke(h, REMOTE_SCALARS_MAKE(method, nIn, nOut), pra)`.
   On the DSP, `remote_arg` is 8 bytes (32-bit `pv` + `nLen`, or `h64`). On
   arm64 it is 16 bytes. The kernel driver converts between the two. Scalars
   are packed into a small in-buffer and the results come back in an
   out-buffer, as QAIC does it.
5. HVX: `qurt_hvx_lock(1)` (128-byte mode) around each kernel. It returned 0,
   and `qurt_hvx_get_mode()` reports 1. No `HAP_power_set` was needed for HVX
   to work.
6. Timing: `r1:0 = c15:14` (UPCYCLE) and `c31:30` (UTIMER, 19.2 MHz), read
   with inline asm. Their ratio gives the DSP clock, about 1248 MHz.

## Measured numbers (v73 build, rpcmem buffers, 1 MiB, single DSP thread)

| Measurement | Result |
|---|---|
| First `remote_handle64_open` (spawns the unsigned PD, loads the skel) | 40–90 ms |
| `add()` warm round trip, default | min 84–97 µs, **p50 ≈ 395 µs**, p99 0.6–3.7 ms |
| `add()` warm round trip, `--qos 1` (PM QoS, 100 µs) | min 71 µs, **p50 ≈ 83 µs**, p90 ≈ 396 µs (bimodal) |
| `add()` warm round trip, `--qos 3` (poll) | min 56 µs, p50 ≈ 392 µs |
| DSP clock (UPCYCLE / UTIMER) | ≈ 1248 MHz |
| u8 sat-add HVX, 1 MiB in to 1 MiB out, 1 rep, wall including RPC | 1.1–1.2 ms (first call), steady p50 0.79 ms, min 0.53–0.68 ms |
| u8 sat-add HVX, 1 rep, DSP only | 0.21 ms, ≈ 3.7–4.0 B/cycle |
| u8 sat-add HVX, 32 reps (buffer warm) | **5.7–5.9 B/cycle ≈ 7.2–7.4 GB/s** |
| u8 sat-add scalar (same op, `vectorize(disable)`) | 0.19 B/cycle, 4.5 ms per MiB, **about 30× slower than HVX** |
| int16 `vdmpy` MAC HVX, 32 reps (read-only, compute) | **30–34 B/cycle ≈ 38–42 GB/s** |
| int16 MAC, 1 rep, wall including RPC | 0.7–0.8 ms |

Run-to-run variation is about 10%. All HVX and scalar outputs are verified
bit-exact against an APPS-side reference. The MAC checksum is checked against
a saturating per-lane reference.

Buffer-size sweep (32 reps, buffer warm):

| Size | u8 sat-add HVX | int16 MAC HVX |
|---|---|---|
| 64 KiB | 14.2 B/cycle | 19.5 B/cycle (unaligned build) |
| 256 KiB | 14.3 B/cycle | 19.3 B/cycle (unaligned build) |
| 1 MiB | 5.7–5.9 B/cycle | 30–34 B/cycle (aligned) |
| 4 MiB | 2.8 B/cycle | 4.4 B/cycle (unaligned build) |

The sweep shows that 1 MiB in + 1 MiB out spills the L2. Above about 1 MiB of
working set the code is DDR-bound unless it prefetches with `l2fetch`, tiles,
or uses VTCM. Aligned `vmem` loads instead of `vmemu` sped up the MAC 2.7× at
1 MiB (12 to 33 B/cycle). The 64 KiB to 4 MiB sweep rows were measured with the
earlier unaligned build. Only the 1 MiB rows use the aligned build.

`--malloc` (heap buffers that the driver copies) costs about 2.4 ms on the
first 1 MiB call, against 1.2 ms with rpcmem. The steady-state p50 is the same,
about 0.8 ms.

**Takeaway:** fixed RPC overhead is 80–400 µs per call, depending on CPU
power/QoS state. Per-pixel work only pays off when it is batched, at roughly
1 ms or more of DSP work per call.

## Failures hit and how they were diagnosed

1. **`remote_handle64_open` returned `0x80000406`.** That is
   `DSP_AEE_EOFFSET (0x80000400) + 6` = `AEE_EUNABLETOLOAD`. The logcat from
   the APPS-side library showed the PD was created
   (`Created user PD ... Unsigned:Y`) and that `apps_std` successfully served
   `/data/local/tmp/lightforge-hvx/./liblfhvx_skel.so`. So the problem was
   inside the DSP loader.
2. **No DSP log lines.** DSP-side messages (`CDSP0:[DU]: ...`, tag `adsprpc`)
   only appear if a `<process>.farf` file (here `lfhvx_bench.farf` containing
   `0x1f`) sits in `ADSP_LIBRARY_PATH`. libcdsprpc's file-watcher thread picks
   it up asynchronously, after the first load attempt has already failed
   ("Enabled adspmsgd with mask 1f001f" was logged after the error). Fix:
   `LFHVX_LOG_WARMUP=1` does a throw-away open of a nonexistent skel (which
   spawns the PD) and sleeps 800 ms. The driver also sleeps 2 s before exiting
   on a failed open so the log thread can flush.
3. **The real cause: `Cannot protect segment 1 ./liblfhvx_skel.so with err:-1`
   then `dlopen_ex failed`.** lld's default layout had 4 PT_LOADs (R, RX, RW
   RELRO, RW) with 64 KiB alignment and segments that start mid-page
   (`vaddr 0x10300`). The QuRT rtld `mprotect`s each segment. Comparing with a
   known-good skel's program headers (two LOADs, RX then RW, both starting on
   4 KiB boundaries in file and memory, 0x1000 alignment) gave the fix:
   `-z max-page-size=4096 -z norelro --no-rosegment -z separate-loadable-segments`.
   Merging the segments alone (`--no-rosegment -z norelro`) was not enough,
   because the RW segment still began at `0x20c0`. Padding each segment to a
   page boundary made it load.
4. **Smaller issues:**
   - The distro clang enables `-fstack-protector-strong` by default, which
     added `__stack_chk_guard`/`__stack_chk_fail` imports. Fixed with
     `-fno-stack-protector`.
   - `-z notext` hid a TEXTREL. The final build has none.
   - Small data (GP-relative addressing) is disabled with `-G0`.
5. **`--dom cdsp1` fails with `0xe` "Invalid domain -1 in uri".** This SoC
   exposes one cDSP to apps. `/dev/fastrpc-nsp1000` exists, but libcdsprpc does
   not map a `cdsp1` URI to it.
6. **`--signed`** (no unsigned-module request) still works. libcdsprpc creates
   an unsigned PD anyway ("Unsigned:Y, Signed:N") because the caller has no
   signed-PD privilege.

## Notes for an app (not tested here)

- The manifest needs `<uses-native-library android:name="libcdsprpc.so" android:required="false"/>`
  so that the app's linker namespace can `dlopen` the public vendor library.
- The skel must be readable by the app process at a path in
  `ADSP_LIBRARY_PATH`. The likely place is the app's `nativeLibraryDir` with
  `android:extractNativeLibs="true"` (or `useLegacyPackaging`), so that the
  `.so` exists as a real file. The skel is a Hexagon ELF, so packaging must be
  checked: AGP may strip or refuse a non-ABI-matching `.so` under `lib/arm64-v8a`.
  The alternative is shipping it as an asset and copying it to `filesDir`.
- Set `ADSP_LIBRARY_PATH` in the process before libcdsprpc initializes.
- The untrusted_app SELinux domain may differ from shell for the
  `IDspService` HAL path. This is the main thing to verify next.

## Open questions

- **App domain:** does `untrusted_app` get the HAL fallback and the unsigned
  PD the way `shell` does? This is the most important follow-up.
- **RPC latency:** the 395 µs p50 is suspiciously constant. Is it a CPU
  wake-up or DSP DCVS effect? PM QoS halves it half of the time. Things to
  try: `HAP_power_set` (DCVS v3 / apptype compute) declared ourselves,
  `remote_handle_control(DSPRPC_CONTROL_WAKELOCK)`, and dspqueue (the
  open-source `dspqueue` in quic/fastrpc) for lower-latency queued work.
- **VTCM and `l2fetch`:** VTCM via `HAP_compute_res_*` (structs would need to
  be re-derived from open sources) and `l2fetch` prefetch should lift the
  DDR-bound 1–4 MiB cases. Multi-threading over the 4–6 HVX contexts is also
  untested.
- **libc in the shell:** what the user-PD shell actually exports (libc, `qurt_*`
  beyond the HVX calls, `HAP_perf_*`). Weak declarations let us probe at runtime.
- **Code coverage:** whether code using libc++/C++ exceptions or TLS can be
  linked without SDK runtime libraries. For kernels, freestanding C plus HVX
  builtins is sufficient.
- **Dynamic loader and lld output:** whether the QuRT rtld needs
  `DT_HEXAGON_VER` (the SDK linker emits it, lld does not). It was not needed
  here.
