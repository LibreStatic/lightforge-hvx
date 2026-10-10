/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Minimal FastRPC skel for the Hexagon cDSP, built with upstream clang/lld only.
 * No Hexagon SDK, no QAIC: the marshalling is hand-written against the
 * protocol documented in inc/lfhvx_proto.h.
 *
 * Every OS symbol used here (qurt_*, HAP_*) is declared by us and resolved at
 * load time by the DSP's dynamic loader against the FastRPC user PD shell.
 */
#include <stddef.h>
#include <stdint.h>

#include "../inc/lfhvx_proto.h"

#define EXPORT __attribute__((visibility("default")))

/* ---- DSP-side remote_arg layout (32-bit pointers) ---------------------- */
typedef struct {
    void *pv;
    uint32_t nLen;
} dsp_remote_buf;

typedef union {
    dsp_remote_buf buf;
    uint32_t h;
    uint64_t h64;
    struct {
        int32_t fd;
        uint32_t offset;
    } dma;
} dsp_remote_arg;

_Static_assert(sizeof(dsp_remote_arg) == 8, "remote_arg must be 8 bytes on hexagon");

/* ---- OS symbols provided by the user PD (declared weak so a missing one
 *      shows up as a NULL instead of a load failure) ----------------------- */
extern int qurt_hvx_lock(int mode) __attribute__((weak));
extern int qurt_hvx_unlock(void) __attribute__((weak));
extern int qurt_hvx_get_mode(void) __attribute__((weak));
extern void HAP_debug_v2(int level, const char *file, int line, const char *fmt, ...)
    __attribute__((weak));

#define LOG(...)                                                         \
    do {                                                                 \
        if (HAP_debug_v2) HAP_debug_v2(2 /*HIGH*/, "lfhvx_skel.c", __LINE__, __VA_ARGS__); \
    } while (0)

#define QURT_HVX_MODE_128B 1

/* ---- freestanding helpers --------------------------------------------- */
/* The compiler may emit calls to these; provide hidden local copies. */
__attribute__((visibility("hidden"))) void *memcpy(void *d, const void *s, size_t n) {
    uint8_t *dd = d;
    const uint8_t *ss = s;
    while (n--) *dd++ = *ss++;
    return d;
}
__attribute__((visibility("hidden"))) void *memset(void *d, int c, size_t n) {
    uint8_t *dd = d;
    while (n--) *dd++ = (uint8_t)c;
    return d;
}

static inline uint64_t read_pcycle(void) {
    uint64_t v;
    __asm__ volatile("%0 = upcycle" : "=r"(v));
    return v;
}
static inline uint64_t read_utimer(void) {
    uint64_t v;
    __asm__ volatile("%0 = utimer" : "=r"(v));
    return v;
}

/* ---- HVX kernels -------------------------------------------------------- */
typedef int32_t HVX_Vector __attribute__((vector_size(128), aligned(128)));
typedef int32_t HVX_UVector __attribute__((vector_size(128), aligned(4)));

#define U8ADD_BODY(VT)                                                    \
    for (uint32_t r = 0; r < reps; r++) {                                 \
        const VT *s = (const VT *)src;                                    \
        VT *d = (VT *)dst;                                                \
        for (uint32_t i = 0; i < n; i++) {                                \
            HVX_Vector v = s[i];                                          \
            d[i] = __builtin_HEXAGON_V6_vaddubsat_128B(v, v);             \
        }                                                                 \
    }

static int64_t u8add_hvx(const uint8_t *src, uint8_t *dst, uint32_t len, uint32_t reps) {
    const uint32_t n = len / 128;
    if ((((uintptr_t)src | (uintptr_t)dst) & 127) == 0) {
        U8ADD_BODY(HVX_Vector) /* aligned vmem */
    } else {
        U8ADD_BODY(HVX_UVector) /* unaligned vmemu */
    }
    return dst[0] + dst[len - 1];
}

static int64_t u8add_scalar(const uint8_t *src, uint8_t *dst, uint32_t len, uint32_t reps) {
    for (uint32_t r = 0; r < reps; r++) {
#pragma clang loop vectorize(disable)
        for (uint32_t i = 0; i < len; i++) {
            uint32_t v = (uint32_t)src[i] * 2u;
            dst[i] = v > 255u ? 255u : (uint8_t)v;
        }
    }
    return dst[0] + dst[len - 1];
}

#define MAC_BODY(VT)                                                      \
    for (uint32_t r = 0; r < reps; r++) {                                 \
        const VT *s = (const VT *)src;                                    \
        uint32_t i = 0;                                                   \
        for (; i + 4 <= n; i += 4) {                                      \
            HVX_Vector a = s[i], b = s[i + 1], c = s[i + 2], d = s[i + 3]; \
            acc0 = __builtin_HEXAGON_V6_vdmpyhvsat_acc_128B(acc0, a, a);  \
            acc1 = __builtin_HEXAGON_V6_vdmpyhvsat_acc_128B(acc1, b, b);  \
            acc2 = __builtin_HEXAGON_V6_vdmpyhvsat_acc_128B(acc2, c, c);  \
            acc3 = __builtin_HEXAGON_V6_vdmpyhvsat_acc_128B(acc3, d, d);  \
        }                                                                 \
        for (; i < n; i++) {                                              \
            HVX_Vector a = s[i];                                          \
            acc0 = __builtin_HEXAGON_V6_vdmpyhvsat_acc_128B(acc0, a, a);  \
        }                                                                 \
    }

static int64_t mac_hvx(const int16_t *src, uint32_t len, uint32_t reps) {
    const uint32_t n = len / 128;
    HVX_Vector acc0 = __builtin_HEXAGON_V6_vd0_128B();
    HVX_Vector acc1 = acc0, acc2 = acc0, acc3 = acc0;
    if (((uintptr_t)src & 127) == 0) {
        MAC_BODY(HVX_Vector)
    } else {
        MAC_BODY(HVX_UVector)
    }
    acc0 = __builtin_HEXAGON_V6_vaddw_128B(acc0, acc1);
    acc2 = __builtin_HEXAGON_V6_vaddw_128B(acc2, acc3);
    acc0 = __builtin_HEXAGON_V6_vaddw_128B(acc0, acc2);
    int64_t sum = 0;
    for (int k = 0; k < 32; k++) sum += acc0[k];
    return sum;
}

/* ---- method implementations -------------------------------------------- */
static int do_kernel(uint32_t method, uint32_t sc, dsp_remote_arg *pra) {
    const uint32_t nin = LF_SCALARS_INBUFS(sc), nout = LF_SCALARS_OUTBUFS(sc);
    const int is_mac = method == LFHVX_M_MAC_HVX;
    if (nin != 2 || nout != (is_mac ? 1u : 2u)) return 0x10; /* AEE_EBADPARM-ish */
    if (pra[0].buf.nLen < sizeof(lfhvx_kernel_args)) return 0x11;
    lfhvx_kernel_args args;
    memcpy(&args, pra[0].buf.pv, sizeof(args));
    const uint8_t *src = pra[1].buf.pv;
    if (args.len == 0 || (args.len & 127) || pra[1].buf.nLen < args.len) return 0x12;
    uint8_t *dst = 0;
    dsp_remote_arg *res_arg;
    if (is_mac) {
        res_arg = &pra[2];
    } else {
        dst = pra[2].buf.pv;
        if (pra[2].buf.nLen < args.len) return 0x13;
        res_arg = &pra[3];
    }
    if (res_arg->buf.nLen < sizeof(lfhvx_kernel_result)) return 0x14;

    lfhvx_kernel_result res = {0};
    int use_hvx = method != LFHVX_M_U8ADD_SCALAR;
    if (use_hvx) {
        res.hvx_lock = qurt_hvx_lock ? qurt_hvx_lock(QURT_HVX_MODE_128B) : -1000;
        res.hvx_mode = qurt_hvx_get_mode ? qurt_hvx_get_mode() : -1000;
        if (res.hvx_lock != 0) {
            LOG("lfhvx: qurt_hvx_lock failed %d", res.hvx_lock);
            memcpy(res_arg->buf.pv, &res, sizeof(res));
            return 0x15;
        }
    }
    uint64_t t0 = read_utimer(), c0 = read_pcycle();
    if (method == LFHVX_M_U8ADD_HVX)
        res.checksum = u8add_hvx(src, dst, args.len, args.reps);
    else if (method == LFHVX_M_U8ADD_SCALAR)
        res.checksum = u8add_scalar(src, dst, args.len, args.reps);
    else
        res.checksum = mac_hvx((const int16_t *)src, args.len, args.reps);
    uint64_t c1 = read_pcycle(), t1 = read_utimer();
    if (use_hvx && qurt_hvx_unlock) qurt_hvx_unlock();
    res.pcycles = c1 - c0;
    res.ticks = t1 - t0;
    memcpy(res_arg->buf.pv, &res, sizeof(res));
    return 0;
}

static int g_open_count;

EXPORT int lfhvx_skel_handle_invoke(uint64_t h, uint32_t sc, dsp_remote_arg *pra) {
    const uint32_t method = LF_SCALARS_METHOD(sc);
    (void)h;
    switch (method) {
    case LFHVX_M_OPEN:
        /* in[0] = int32 len, in[1] = uri; out handle = pra[2].h64 */
        if (LF_SCALARS_INBUFS(sc) != 2 || LF_SCALARS_OUTHANDLES(sc) != 1) return 0x20;
        g_open_count++;
        pra[2].h64 = 0x4c460000u + (uint32_t)g_open_count; /* opaque, non-zero */
        LOG("lfhvx: open #%d", g_open_count);
        return 0;
    case LFHVX_M_CLOSE:
        LOG("lfhvx: close");
        return 0;
    case LFHVX_M_ADD: {
        if (LF_SCALARS_INBUFS(sc) != 1 || LF_SCALARS_OUTBUFS(sc) != 1) return 0x21;
        if (pra[0].buf.nLen < 8 || pra[1].buf.nLen < 4) return 0x22;
        int32_t ab[2];
        memcpy(ab, pra[0].buf.pv, 8);
        int32_t sum = ab[0] + ab[1];
        memcpy(pra[1].buf.pv, &sum, 4);
        return 0;
    }
    case LFHVX_M_U8ADD_HVX:
    case LFHVX_M_U8ADD_SCALAR:
    case LFHVX_M_MAC_HVX:
        return do_kernel(method, sc, pra);
    case LFHVX_M_INFO: {
        if (LF_SCALARS_OUTBUFS(sc) != 1 || pra[0].buf.nLen < sizeof(lfhvx_info)) return 0x23;
        lfhvx_info info = {0};
        info.pcycle = read_pcycle();
        info.utimer = read_utimer();
        info.hvx_mode_before = qurt_hvx_get_mode ? (uint32_t)qurt_hvx_get_mode() : 0xffffffffu;
        info.hvx_lock_rc = qurt_hvx_lock ? qurt_hvx_lock(QURT_HVX_MODE_128B) : -1000;
        info.hvx_mode_locked = qurt_hvx_get_mode ? (uint32_t)qurt_hvx_get_mode() : 0xffffffffu;
        if (info.hvx_lock_rc == 0 && qurt_hvx_unlock) qurt_hvx_unlock();
        info.skel_version = 1;
        memcpy(pra[0].buf.pv, &info, sizeof(info));
        return 0;
    }
    default:
        return 0x2f;
    }
}
