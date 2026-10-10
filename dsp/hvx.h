/* SPDX-License-Identifier: MIT
 * HVX 128B intrinsics from LLVM's own headers (Apache-2.0 with LLVM exception);
 * on the host (LF_HOST) a small emulation is used instead. */
#ifndef LF_HVX_H
#define LF_HVX_H
#ifdef LF_HOST
#include "hvx_emu.h"
#else
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
static inline HVX_Vector lf_vload(const void *p) { return *(const HVX_Vector *)p; }
static inline void lf_vstore(void *p, HVX_Vector v) { *(HVX_Vector *)p = v; }
static inline HVX_Vector lf_vloadu(const void *p) { return *(const HVX_UVector *)p; }
static inline void lf_vstoreu(void *p, HVX_Vector v) { *(HVX_UVector *)p = v; }
#endif
#endif
