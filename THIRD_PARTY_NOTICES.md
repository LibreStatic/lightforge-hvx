# Third-party notices

This project contains no code copied from the Hexagon SDK, QAIC, QNN, SNPE or any other
Qualcomm-licensed header, library or tool.

## quic/fastrpc (BSD 3-Clause)

`apps/lfhvx_bench.c`, `apps/lfrife.c`, `android-test/jni/lfhvx_jni.c`, `inc/lfhvx_proto.h` and
`inc/lfrife_proto.h` re-declare FastRPC symbols, structures and constants (`remote_handle64_*`,
`remote_session_control`, `rpcmem_*`, `remote_arg`, the scalars macros and the module-table
open/close protocol) based on [quic/fastrpc](https://github.com/quic/fastrpc) (commit `a56e9d4`):
`inc/remote.h`, `inc/rpcmem.h`, `inc/HAP_debug.h` and `src/mod_table.c`.

```
Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Used as references only (no code copied)

- [Halide](https://github.com/halide/Halide) (MIT): `src/runtime/qurt_hvx.cpp`, `mini_qurt.h` and
  `mini_qurt_vtcm.h`, for the `qurt_hvx_lock` mode convention, the `qurt_thread_*` prototypes and
  the VTCM request calls, and the `qurt_hvx_get_units() >> 8` context count.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT): `ggml/src/ggml-hexagon/htp/main.c`, for
  the argument lists of the VTCM compute-resource calls and the `qurt_hvx_get_units() >> 8` use.
- [tinygrad](https://github.com/tinygrad/tinygrad) (MIT): the DSP build flags and linker-script
  alignment, and (`tinygrad/runtime/ops_dsp.py`) the `HAP_power_set` prototype and the field layout
  of its DCVS v2 request, re-declared in `dsp/lfrife_skel.c`. The voltage-corner values were measured
  on the device.

## RIFE

The network implements [RIFE](https://github.com/hzwer/Practical-RIFE) v4.6 (MIT, Copyright (c) 2021
hzwer). The weights are not part of this repository; `tools/export_weights.py` converts them from a
local copy.
