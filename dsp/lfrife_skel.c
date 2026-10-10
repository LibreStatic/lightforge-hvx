/*
 * SPDX-License-Identifier: MIT
 *
 * RIFE v4.6 flow network on the Hexagon cDSP (HVX). Freestanding, built with
 * upstream clang + ld.lld only. Protocol: inc/lfrife_proto.h.
 */
#include "dsp_common.h"
#include "../inc/lfrife_proto.h"
#include "hvx.h"
#include "lfpool.h"

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

extern void *malloc(size_t) __attribute__((weak));
extern void *memalign(size_t, size_t) __attribute__((weak));
extern void *calloc(size_t, size_t) __attribute__((weak));
extern void free(void *) __attribute__((weak));
extern void *HAP_malloc(size_t) __attribute__((weak));
extern int posix_memalign(void **, size_t, size_t) __attribute__((weak));

#define LF_UNITY 1
#include "lfrife_net.c"
#define DUP(x) Q6_R_combine_RlRl((x), (x))
static uint8_t g_scratch[16384] __attribute__((aligned(128)));
static volatile uint32_t g_sink;

/* K1: qf16 accumulate, vector x scalar (rt) form, P pixels in flight */
__attribute__((noinline)) static void k_qf16_rt(const HVX_Vector *wv, const uint16_t *act, uint32_t iters, HVX_Vector *outv) {
    HVX_Vector a0 = Q6_V_vzero(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    for (uint32_t i = 0; i < iters; i++) {
        HVX_Vector w = wv[i & 15];
        const uint16_t *ap = act + (i & 63) * 8;
        uint32_t r0 = DUP(ap[0]), r1 = DUP(ap[1]), r2 = DUP(ap[2]), r3 = DUP(ap[3]);
        uint32_t r4 = DUP(ap[4]), r5 = DUP(ap[5]), r6 = DUP(ap[6]), r7 = DUP(ap[7]);
        a0 = Q6_Vqf16_vadd_Vqf16Vqf16(a0, Q6_Vqf16_vmpy_VhfRhf(w, r0));
        a1 = Q6_Vqf16_vadd_Vqf16Vqf16(a1, Q6_Vqf16_vmpy_VhfRhf(w, r1));
        a2 = Q6_Vqf16_vadd_Vqf16Vqf16(a2, Q6_Vqf16_vmpy_VhfRhf(w, r2));
        a3 = Q6_Vqf16_vadd_Vqf16Vqf16(a3, Q6_Vqf16_vmpy_VhfRhf(w, r3));
        a4 = Q6_Vqf16_vadd_Vqf16Vqf16(a4, Q6_Vqf16_vmpy_VhfRhf(w, r4));
        a5 = Q6_Vqf16_vadd_Vqf16Vqf16(a5, Q6_Vqf16_vmpy_VhfRhf(w, r5));
        a6 = Q6_Vqf16_vadd_Vqf16Vqf16(a6, Q6_Vqf16_vmpy_VhfRhf(w, r6));
        a7 = Q6_Vqf16_vadd_Vqf16Vqf16(a7, Q6_Vqf16_vmpy_VhfRhf(w, r7));
    }
    outv[0] = a0; outv[1] = a1; outv[2] = a2; outv[3] = a3; outv[4] = a4; outv[5] = a5; outv[6] = a6; outv[7] = a7;
}

/* K2: widening hf x hf -> qf32 pair with splatted activation, P=4 pixels */
__attribute__((noinline)) static void k_qf32_w(const HVX_Vector *wv, const uint16_t *act, uint32_t iters, HVX_Vector *outv) {
    HVX_Vector l0 = Q6_V_vzero(), h0 = l0, l1 = l0, h1 = l0, l2 = l0, h2 = l0, l3 = l0, h3 = l0;
    for (uint32_t i = 0; i < iters; i++) {
        HVX_Vector w = wv[i & 15];
        const uint16_t *ap = act + (i & 63) * 8;
        HVX_Vector s0 = Q6_Vh_vsplat_R(ap[0]), s1 = Q6_Vh_vsplat_R(ap[1]), s2 = Q6_Vh_vsplat_R(ap[2]), s3 = Q6_Vh_vsplat_R(ap[3]);
        HVX_VectorPair p0 = Q6_Wqf32_vmpy_VhfVhf(w, s0), p1 = Q6_Wqf32_vmpy_VhfVhf(w, s1);
        HVX_VectorPair p2 = Q6_Wqf32_vmpy_VhfVhf(w, s2), p3 = Q6_Wqf32_vmpy_VhfVhf(w, s3);
        l0 = Q6_Vqf32_vadd_Vqf32Vqf32(l0, Q6_V_lo_W(p0)); h0 = Q6_Vqf32_vadd_Vqf32Vqf32(h0, Q6_V_hi_W(p0));
        l1 = Q6_Vqf32_vadd_Vqf32Vqf32(l1, Q6_V_lo_W(p1)); h1 = Q6_Vqf32_vadd_Vqf32Vqf32(h1, Q6_V_hi_W(p1));
        l2 = Q6_Vqf32_vadd_Vqf32Vqf32(l2, Q6_V_lo_W(p2)); h2 = Q6_Vqf32_vadd_Vqf32Vqf32(h2, Q6_V_hi_W(p2));
        l3 = Q6_Vqf32_vadd_Vqf32Vqf32(l3, Q6_V_lo_W(p3)); h3 = Q6_Vqf32_vadd_Vqf32Vqf32(h3, Q6_V_hi_W(p3));
    }
    outv[0] = l0; outv[1] = h0; outv[2] = l1; outv[3] = h1; outv[4] = l2; outv[5] = h2; outv[6] = l3; outv[7] = h3;
}

/* K3: qf16 accumulate of vector x vector with splat */
__attribute__((noinline)) static void k_qf16_v(const HVX_Vector *wv, const uint16_t *act, uint32_t iters, HVX_Vector *outv) {
    HVX_Vector a0 = Q6_V_vzero(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    for (uint32_t i = 0; i < iters; i++) {
        HVX_Vector w = wv[i & 15];
        const uint16_t *ap = act + (i & 63) * 8;
        a0 = Q6_Vqf16_vadd_Vqf16Vqf16(a0, Q6_Vqf16_vmpy_VhfVhf(w, Q6_Vh_vsplat_R(ap[0])));
        a1 = Q6_Vqf16_vadd_Vqf16Vqf16(a1, Q6_Vqf16_vmpy_VhfVhf(w, Q6_Vh_vsplat_R(ap[1])));
        a2 = Q6_Vqf16_vadd_Vqf16Vqf16(a2, Q6_Vqf16_vmpy_VhfVhf(w, Q6_Vh_vsplat_R(ap[2])));
        a3 = Q6_Vqf16_vadd_Vqf16Vqf16(a3, Q6_Vqf16_vmpy_VhfVhf(w, Q6_Vh_vsplat_R(ap[3])));
        a4 = Q6_Vqf16_vadd_Vqf16Vqf16(a4, Q6_Vqf16_vmpy_VhfVhf(w, Q6_Vh_vsplat_R(ap[4])));
        a5 = Q6_Vqf16_vadd_Vqf16Vqf16(a5, Q6_Vqf16_vmpy_VhfVhf(w, Q6_Vh_vsplat_R(ap[5])));
        a6 = Q6_Vqf16_vadd_Vqf16Vqf16(a6, Q6_Vqf16_vmpy_VhfVhf(w, Q6_Vh_vsplat_R(ap[6])));
        a7 = Q6_Vqf16_vadd_Vqf16Vqf16(a7, Q6_Vqf16_vmpy_VhfVhf(w, Q6_Vh_vsplat_R(ap[7])));
    }
    outv[0] = a0; outv[1] = a1; outv[2] = a2; outv[3] = a3; outv[4] = a4; outv[5] = a5; outv[6] = a6; outv[7] = a7;
}

/* floor tests: only vmpy (rt) or only vadd, no loads */
__attribute__((noinline)) static void k_floor(HVX_Vector *outv, uint32_t iters, uint32_t mode, uint32_t r) {
    HVX_Vector a0 = outv[0], a1 = outv[1], a2 = outv[2], a3 = outv[3], a4 = outv[4], a5 = outv[5], a6 = outv[6], a7 = outv[7];
    HVX_Vector w = outv[8];
    if (mode == 0) {
        for (uint32_t i = 0; i < iters; i++) {
            a0 = Q6_Vqf16_vmpy_VhfRhf(w, r); a1 = Q6_Vqf16_vmpy_VhfRhf(w, r + 1); a2 = Q6_Vqf16_vmpy_VhfRhf(w, r + 2); a3 = Q6_Vqf16_vmpy_VhfRhf(w, r + 3);
            a4 = Q6_Vqf16_vmpy_VhfRhf(w, r + 4); a5 = Q6_Vqf16_vmpy_VhfRhf(w, r + 5); a6 = Q6_Vqf16_vmpy_VhfRhf(w, r + 6); a7 = Q6_Vqf16_vmpy_VhfRhf(w, r + 7);
        }
    } else if (mode == 1) {
        for (uint32_t i = 0; i < iters; i++) {
            a0 = Q6_Vqf16_vadd_Vqf16Vqf16(a0, w); a1 = Q6_Vqf16_vadd_Vqf16Vqf16(a1, w); a2 = Q6_Vqf16_vadd_Vqf16Vqf16(a2, w); a3 = Q6_Vqf16_vadd_Vqf16Vqf16(a3, w);
            a4 = Q6_Vqf16_vadd_Vqf16Vqf16(a4, w); a5 = Q6_Vqf16_vadd_Vqf16Vqf16(a5, w); a6 = Q6_Vqf16_vadd_Vqf16Vqf16(a6, w); a7 = Q6_Vqf16_vadd_Vqf16Vqf16(a7, w);
        }
    } else if (mode == 2) { /* both, vmpy independent of adds */
        for (uint32_t i = 0; i < iters; i++) {
            HVX_Vector m0 = Q6_Vqf16_vmpy_VhfRhf(w, r), m1 = Q6_Vqf16_vmpy_VhfRhf(w, r + 1), m2 = Q6_Vqf16_vmpy_VhfRhf(w, r + 2), m3 = Q6_Vqf16_vmpy_VhfRhf(w, r + 3);
            a0 = Q6_Vqf16_vadd_Vqf16Vqf16(a0, m0); a1 = Q6_Vqf16_vadd_Vqf16Vqf16(a1, m1); a2 = Q6_Vqf16_vadd_Vqf16Vqf16(a2, m2); a3 = Q6_Vqf16_vadd_Vqf16Vqf16(a3, m3);
            m0 = Q6_Vqf16_vmpy_VhfRhf(w, r + 4); m1 = Q6_Vqf16_vmpy_VhfRhf(w, r + 5); m2 = Q6_Vqf16_vmpy_VhfRhf(w, r + 6); m3 = Q6_Vqf16_vmpy_VhfRhf(w, r + 7);
            a4 = Q6_Vqf16_vadd_Vqf16Vqf16(a4, m0); a5 = Q6_Vqf16_vadd_Vqf16Vqf16(a5, m1); a6 = Q6_Vqf16_vadd_Vqf16Vqf16(a6, m2); a7 = Q6_Vqf16_vadd_Vqf16Vqf16(a7, m3);
        }
    } else { /* int16 vmpy acc pair, rt form */
        HVX_VectorPair p0 = Q6_W_vcombine_VV(a1, a0), p1 = Q6_W_vcombine_VV(a3, a2), p2 = Q6_W_vcombine_VV(a5, a4), p3 = Q6_W_vcombine_VV(a7, a6);
        for (uint32_t i = 0; i < iters; i++) {
            p0 = Q6_Ww_vmpyacc_WwVhRh(p0, w, r); p1 = Q6_Ww_vmpyacc_WwVhRh(p1, w, r + 1); p2 = Q6_Ww_vmpyacc_WwVhRh(p2, w, r + 2); p3 = Q6_Ww_vmpyacc_WwVhRh(p3, w, r + 3);
            p0 = Q6_Ww_vmpyacc_WwVhRh(p0, w, r + 4); p1 = Q6_Ww_vmpyacc_WwVhRh(p1, w, r + 5); p2 = Q6_Ww_vmpyacc_WwVhRh(p2, w, r + 6); p3 = Q6_Ww_vmpyacc_WwVhRh(p3, w, r + 7);
        }
        a0 = Q6_V_lo_W(p0); a1 = Q6_V_hi_W(p0); a2 = Q6_V_lo_W(p1); a3 = Q6_V_hi_W(p1); a4 = Q6_V_lo_W(p2); a5 = Q6_V_hi_W(p2); a6 = Q6_V_lo_W(p3); a7 = Q6_V_hi_W(p3);
    }
    outv[0] = a0; outv[1] = a1; outv[2] = a2; outv[3] = a3; outv[4] = a4; outv[5] = a5; outv[6] = a6; outv[7] = a7;
}

/* K4: int16 x int16 -> int32 pair accumulate (vmpyacc rt form), 8 px */
__attribute__((noinline)) static void k_i16(const HVX_Vector *wv, const uint16_t *act, uint32_t iters, HVX_Vector *outv) {
    HVX_VectorPair p0 = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero()), p1 = p0, p2 = p0, p3 = p0, p4 = p0, p5 = p0, p6 = p0, p7 = p0;
    for (uint32_t i = 0; i < iters; i++) {
        HVX_Vector w = wv[i & 15];
        const uint16_t *ap = act + (i & 63) * 8;
        p0 = Q6_Ww_vmpyacc_WwVhRh(p0, w, DUP(ap[0])); p1 = Q6_Ww_vmpyacc_WwVhRh(p1, w, DUP(ap[1]));
        p2 = Q6_Ww_vmpyacc_WwVhRh(p2, w, DUP(ap[2])); p3 = Q6_Ww_vmpyacc_WwVhRh(p3, w, DUP(ap[3]));
        p4 = Q6_Ww_vmpyacc_WwVhRh(p4, w, DUP(ap[4])); p5 = Q6_Ww_vmpyacc_WwVhRh(p5, w, DUP(ap[5]));
        p6 = Q6_Ww_vmpyacc_WwVhRh(p6, w, DUP(ap[6])); p7 = Q6_Ww_vmpyacc_WwVhRh(p7, w, DUP(ap[7]));
    }
    outv[0] = Q6_V_lo_W(p0); outv[1] = Q6_V_lo_W(p1); outv[2] = Q6_V_lo_W(p2); outv[3] = Q6_V_lo_W(p3);
    outv[4] = Q6_V_hi_W(p4); outv[5] = Q6_V_hi_W(p5); outv[6] = Q6_V_hi_W(p6); outv[7] = Q6_V_hi_W(p7);
    outv[8] = Q6_V_hi_W(p0); outv[9] = Q6_V_hi_W(p1); outv[10] = Q6_V_hi_W(p2); outv[11] = Q6_V_hi_W(p3);
    outv[12] = Q6_V_lo_W(p4); outv[13] = Q6_V_lo_W(p5); outv[14] = Q6_V_lo_W(p6); outv[15] = Q6_V_lo_W(p7);
}

/* K5: u8 x s8 -> int32, 4 MACs per word lane (vrmpy rt form), 8 px, one scalar = 4 input channels */
__attribute__((noinline)) static void k_rmpy(const HVX_Vector *wv, const uint16_t *act, uint32_t iters, HVX_Vector *outv) {
    HVX_Vector a0 = Q6_V_vzero(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    const uint32_t *au = (const uint32_t *)act;
    for (uint32_t i = 0; i < iters; i++) {
        HVX_Vector w = wv[i & 15];
        const uint32_t *ap = au + (i & 63) * 4;
        a0 = Q6_Vw_vrmpyacc_VwVubRb(a0, w, ap[0]); a1 = Q6_Vw_vrmpyacc_VwVubRb(a1, w, ap[1]);
        a2 = Q6_Vw_vrmpyacc_VwVubRb(a2, w, ap[2]); a3 = Q6_Vw_vrmpyacc_VwVubRb(a3, w, ap[3]);
        a4 = Q6_Vw_vrmpyacc_VwVubRb(a4, w, ap[0] + 1); a5 = Q6_Vw_vrmpyacc_VwVubRb(a5, w, ap[1] + 1);
        a6 = Q6_Vw_vrmpyacc_VwVubRb(a6, w, ap[2] + 1); a7 = Q6_Vw_vrmpyacc_VwVubRb(a7, w, ap[3] + 1);
    }
    outv[0] = a0; outv[1] = a1; outv[2] = a2; outv[3] = a3; outv[4] = a4; outv[5] = a5; outv[6] = a6; outv[7] = a7;
}

static lfpool g_pool;
static uint8_t g_mt_scratch[LFP_MAX][16384] __attribute__((aligned(128)));
static uint32_t g_mt_iters, g_mt_kernel;
static uint64_t g_mt_pc[LFP_MAX];
static void mt_fn(lfpool *p, int tid, void *arg) {
    uint8_t *sc = g_mt_scratch[tid];
    HVX_Vector *wv = (HVX_Vector *)sc;
    for (int i = 0; i < 16 * 32; i++) ((uint16_t *)sc)[i] = 0x3800 + (i & 15);
    uint16_t *act = (uint16_t *)(sc + 4096);
    for (int i = 0; i < 64 * 8 + 8; i++) act[i] = 0x3c00 + (i & 7);
    HVX_Vector *ov = (HVX_Vector *)(sc + 8192);
    lfpool_barrier(p);
    uint64_t c0 = read_pcycle();
    if (g_mt_kernel == 1) k_qf16_rt(wv, act, g_mt_iters, ov);
    else if (g_mt_kernel == 2) k_qf32_w(wv, act, g_mt_iters, ov);
    else if (g_mt_kernel == 4) k_i16(wv, act, g_mt_iters, ov);
    else if (g_mt_kernel == 5) k_rmpy(wv, act, g_mt_iters, ov);
    else k_qf16_v(wv, act, g_mt_iters, ov);
    g_mt_pc[tid] = read_pcycle() - c0;
    lfpool_barrier(p);
}

static int probe(const lfrife_probe_args *pa, uint8_t *in, uint32_t inlen, uint8_t *out, uint32_t outlen) {
    switch (pa->exp) {
    case 1: { /* hf lane semantics. in: A (128B hf), B (128B hf); a = Rt */
        if (inlen < 256 || outlen < 1280) return 0x50;
        HVX_Vector A = *(HVX_UVector *)in, B = *(HVX_UVector *)(in + 128);
        if (qurt_hvx_lock(QURT_HVX_MODE_128B)) return 0x51;
        HVX_VectorPair z = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
        HVX_VectorPair p = Q6_Wsf_vmpyacc_WsfVhfVhf(z, A, B);
        *(HVX_UVector *)(out + 0) = Q6_V_lo_W(p);
        *(HVX_UVector *)(out + 128) = Q6_V_hi_W(p);
        *(HVX_UVector *)(out + 256) = Q6_Vhf_vcvt_VsfVsf(Q6_V_hi_W(p), Q6_V_lo_W(p));
        *(HVX_UVector *)(out + 384) = Q6_Vhf_vcvt_VsfVsf(Q6_V_lo_W(p), Q6_V_hi_W(p));
        HVX_VectorPair c = Q6_Wsf_vcvt_Vhf(A);
        *(HVX_UVector *)(out + 512) = Q6_V_lo_W(c);
        *(HVX_UVector *)(out + 640) = Q6_V_hi_W(c);
        *(HVX_UVector *)(out + 768) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfRhf(A, pa->a));
        *(HVX_UVector *)(out + 896) = Q6_Vh_vsplat_R(pa->a);
        *(HVX_UVector *)(out + 1024) = Q6_Vhf_vmpy_VhfVhf(A, B);
        *(HVX_UVector *)(out + 1152) = Q6_Vhf_vmax_VhfVhf(A, B);
        qurt_hvx_unlock();
        return 0;
    }
    case 3: { /* IEEE vs qfloat variants; zeros mean the instruction produced nothing */
        if (inlen < 256 || outlen < 1280) return 0x50;
        HVX_Vector A = *(HVX_UVector *)in, B = *(HVX_UVector *)(in + 128);
        if (qurt_hvx_lock(QURT_HVX_MODE_128B)) return 0x51;
        *(HVX_UVector *)(out + 0) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(A, B));
        *(HVX_UVector *)(out + 128) = Q6_Vhf_vadd_VhfVhf(A, B);
        HVX_VectorPair q = Q6_Wqf32_vmpy_VhfVhf(A, B);
        *(HVX_UVector *)(out + 256) = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(q));
        *(HVX_UVector *)(out + 384) = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(q));
        *(HVX_UVector *)(out + 512) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(A, B));
        HVX_VectorPair w = Q6_Wsf_vadd_VhfVhf(A, B);
        *(HVX_UVector *)(out + 640) = Q6_V_lo_W(w);
        *(HVX_UVector *)(out + 768) = Q6_V_hi_W(w);
        HVX_VectorPair z = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
        HVX_VectorPair p = Q6_Wsf_vmpyacc_WsfVhfVhf(z, A, B);
        *(HVX_UVector *)(out + 896) = Q6_V_lo_W(p);
        *(HVX_UVector *)(out + 1024) = Q6_V_hi_W(p);
        *(HVX_UVector *)(out + 1152) = Q6_Vhf_vmpy_VhfVhf(A, B);
        qurt_hvx_unlock();
        return 0;
    }
    case 10: { /* microbench: a = kernel id, b = iterations. out: u64 ticks, u64 pcycles */
        if (outlen < 16) return 0x50;
        if (qurt_hvx_lock(QURT_HVX_MODE_128B)) return 0x51;
        HVX_Vector *wv = (HVX_Vector *)g_scratch;
        for (int i = 0; i < 16 * 32; i++) ((uint16_t *)g_scratch)[i] = 0x3800 + (i & 15);
        uint16_t *act = (uint16_t *)(g_scratch + 4096);
        for (int i = 0; i < 64 * 8 + 8; i++) act[i] = 0x3c00 + (i & 7);
        HVX_Vector *ov = (HVX_Vector *)(g_scratch + 8192);
        uint64_t t0 = read_utimer(), c0 = read_pcycle();
        if (pa->a >= 10) k_floor(ov, pa->b, pa->a - 10, 0x3c003c00);
        else if (pa->a == 1) k_qf16_rt(wv, act, pa->b, ov);
        else if (pa->a == 2) k_qf32_w(wv, act, pa->b, ov);
        else k_qf16_v(wv, act, pa->b, ov);
        uint64_t c1 = read_pcycle(), t1 = read_utimer();
        qurt_hvx_unlock();
        uint64_t *o = (uint64_t *)out;
        o[0] = t1 - t0; o[1] = c1 - c0;
        g_sink = ((uint32_t *)ov)[0];
        return 0;
    }
    case 11: { /* MT microbench: a = kernel, b = iterations, c = threads. out: u64 ticks, u64 max pcycles, u64 thread-create ticks */
        if (outlen < 24) return 0x50;
        if (qurt_hvx_lock(QURT_HVX_MODE_128B)) return 0x51;
        g_mt_iters = pa->b; g_mt_kernel = pa->a;
        uint64_t t0 = read_utimer();
        int rc = lfpool_run(&g_pool, (int)pa->c, mt_fn, 0);
        uint64_t t1 = read_utimer();
        qurt_hvx_unlock();
        uint64_t *o = (uint64_t *)out;
        o[0] = t1 - t0; o[1] = 0;
        for (uint32_t i = 0; i < pa->c; i++) if (g_mt_pc[i] > o[1]) o[1] = g_mt_pc[i];
        o[2] = (uint64_t)rc;
        return 0;
    }
    case 20: { /* semantics of the intrinsics used by the vector stages. in: A sf, B sf, H hf, W3 words, W4 words (5 x 128 B) */
        if (inlen < 640 || outlen < 40 * 128) return 0x50;
        HVX_Vector A = *(HVX_UVector *)in, B = *(HVX_UVector *)(in + 128), H = *(HVX_UVector *)(in + 256);
        HVX_Vector W3 = *(HVX_UVector *)(in + 384), W4 = *(HVX_UVector *)(in + 512);
        if (qurt_hvx_lock(QURT_HVX_MODE_128B)) return 0x51;
        HVX_Vector *o = (HVX_Vector *)out;
        int k = 0;
        HVX_Vector qa = Q6_Vqf32_vadd_VsfVsf(A, B), qs = Q6_Vqf32_vsub_VsfVsf(A, B), qm = Q6_Vqf32_vmpy_VsfVsf(A, B);
        o[k++] = Q6_Vsf_equals_Vqf32(qa);
        o[k++] = Q6_Vsf_equals_Vqf32(qs);
        o[k++] = Q6_Vsf_equals_Vqf32(qm);
        o[k++] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfRsf(A, 0x40200000u)); /* 2.5 */
        o[k++] = Q6_Vsf_equals_Vw(W3);
        o[k++] = Q6_Vw_equals_Vsf(A);
        o[k++] = Q6_Vsf_vmax_VsfVsf(A, B);
        o[k++] = Q6_Vsf_vmin_VsfVsf(A, B);
        o[k++] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(qm, qa));
        o[k++] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(qa, qs));
        HVX_VectorPair one = Q6_Wqf32_vmpy_VhfVhf(H, Q6_Vh_vsplat_R(0x3c00));
        o[k++] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(one));
        o[k++] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(one));
        o[k++] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(qm, qa)); /* hi=qm, lo=qa */
        o[k++] = Q6_Vhf_vcvt_VsfVsf(B, A);
        HVX_VectorPair cv = Q6_Wsf_vcvt_Vhf(H);
        o[k++] = Q6_V_lo_W(cv);
        o[k++] = Q6_V_hi_W(cv);
        HVX_VectorPair sh = Q6_W_vshuff_VVR(W3, W4, -4);
        o[k++] = Q6_V_lo_W(sh);
        o[k++] = Q6_V_hi_W(sh);
        HVX_VectorPair dl = Q6_W_vdeal_VVR(W3, W4, -4);
        o[k++] = Q6_V_lo_W(dl);
        o[k++] = Q6_V_hi_W(dl);
        o[k++] = Q6_V_valign_VVR(W3, W4, 16);
        o[k++] = Q6_V_vror_VR(W3, 4);
        HVX_VectorPred q = Q6_Q_vcmp_gt_VwVw(W3, Q6_V_vsplat_R(10));
        o[k++] = Q6_V_vmux_QVV(q, W3, W4);
        HVX_Vector qh = Q6_Vqf16_vmpy_VhfRhf(H, 0x41004100u); /* x8 */
        HVX_VectorPair w16 = Q6_Wqf32_vmpy_Vqf16Vhf(qh, Q6_Vh_vsplat_R(0x3c00));
        o[k++] = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(w16));
        o[k++] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(w16));
        o[k++] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(A, Q6_V_vsplat_R(0x4b000000)));
        HVX_VectorPair sh2 = Q6_W_vshuff_VVR(W3, W3, -8);
        o[k++] = Q6_V_lo_W(sh2);
        o[k++] = Q6_V_hi_W(sh2);
        HVX_VectorPair sh3 = Q6_W_vshuff_VVR(W3, W4, -2);
        o[k++] = Q6_V_lo_W(sh3);
        o[k++] = Q6_V_hi_W(sh3);
        HVX_VectorPair dl2 = Q6_W_vdeal_VVR(W3, W4, -8);
        o[k++] = Q6_V_lo_W(dl2);
        o[k++] = Q6_V_hi_W(dl2);
        HVX_VectorPair un = Q6_Wuw_vzxt_Vuh(H);
        o[k++] = Q6_V_lo_W(un);
        o[k++] = Q6_V_hi_W(un);
        o[k++] = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_Vqf16Vhf(qh, Q6_Vh_vsplat_R(0x3c00)));
        o[k++] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(qm, A));
        o[k++] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vsf(qm, A));
        qurt_hvx_unlock();
        return 0;
    }
    case 2: { /* which libc symbols does the shell export */
        uint32_t *o = (uint32_t *)out;
        o[0] = malloc != 0; o[1] = memalign != 0; o[2] = calloc != 0; o[3] = free != 0;
        o[7] = qurt_hvx_get_units ? (uint32_t)qurt_hvx_get_units() : 0;
        if (pa->a && malloc) {
            void *p = memalign ? memalign(128, pa->a) : malloc(pa->a);
            o[8] = (uint32_t)(uintptr_t)p;
            if (p) { memset(p, 1, pa->a); if (free) free(p); }
        }
        return 0;
    }
    default:
        return 0x5f;
    }
}

/*
 * DCVS v2 request for HAP_power_set (request type 7). The layout follows the re-declaration in tinygrad's
 * MIT-licensed tinygrad/runtime/ops_dsp.py (see THIRD_PARTY_NOTICES.md); no SDK header is used.
 */
typedef struct {
    int type;
    int pad0;
    _Bool dcvs_enable;
    char dcvs_option;
    _Bool set_latency;
    int latency;
    _Bool set_dcvs_params;
    short pad1;
    char target_corner;
    char min_corner;
    char max_corner;
    int pad2[3];
} lf_dcvs_v2_req;
extern int HAP_power_set(void *ctx, void *req) __attribute__((weak));
static int g_power_ctx; /* its address identifies this client to the power manager */
static uint32_t g_power_corner;

/* Pins the core and bus clocks at [corner] with DCVS off, or hands them back to DCVS when corner is 0. */
static int power_vote(uint32_t corner) {
    if (!HAP_power_set) return -1;
    lf_dcvs_v2_req q;
    memset(&q, 0, sizeof q);
    q.type = 7;
    q.dcvs_enable = corner == 0;
    q.set_latency = 1;
    q.latency = corner ? 100 : 65535; /* us of wake-up latency the client tolerates; large = deep sleep allowed */
    q.set_dcvs_params = 1;
    q.target_corner = (char)corner;
    const int rc = HAP_power_set(&g_power_ctx, &q);
    if (rc == 0) g_power_corner = corner;
    return rc;
}

static int g_open_count;
static lfnet *g_net;
static uint8_t *g_blob;
static uint32_t g_blob_size;

static int do_setup(dsp_remote_arg *pra) {
    lfrife_setup_args sa;
    memcpy(&sa, pra[0].buf.pv, sizeof sa);
    lfrife_setup_res res;
    memset(&res, 0, sizeof res);
    uint32_t bsz = pra[1].buf.nLen;
    if (sa.version != LFRIFE_PROTO_VERSION) { res.rc = -1; goto out; }
    if (!g_blob || g_blob_size != bsz || (sa.flags & 1)) {
        if (g_net) { lfnet_destroy(g_net); g_net = 0; }
        if (g_blob) { free(g_blob); g_blob = 0; }
        g_blob = (uint8_t *)memalign(128, bsz);
        if (!g_blob) { res.rc = -3; goto out; }
        memcpy(g_blob, pra[1].buf.pv, bsz);
        g_blob_size = bsz;
    }
    if (g_net) { lfnet_destroy(g_net); g_net = 0; }
    res.hvx_units = qurt_hvx_get_units ? (uint32_t)qurt_hvx_get_units() : 0;
    {
        /* qurt_hvx_get_units() >> 8 = number of 128-byte HVX contexts (0x600 = 6 on the SM8845); lfnet_create then
         * confirms by really starting that many workers, so a wrong or missing answer cannot hang the barriers */
        uint32_t nt = sa.nthreads < 1 ? 1 : sa.nthreads > LFP_MAX ? LFP_MAX : sa.nthreads;
        const uint32_t units = (res.hvx_units >> 8) & 0xff;
        if (units && nt > units) nt = units;
        if (qurt_hvx_lock(QURT_HVX_MODE_128B)) { res.rc = -4; goto out; }
        g_net = lfnet_create(g_blob, bsz, (int)sa.w, (int)sa.h, (int)nt);
        qurt_hvx_unlock();
    }
    if (!g_net) { res.rc = -2; goto out; }
    res.nthreads = (uint32_t)g_net->nt;
    res.power_rc = (sa.power || g_power_corner) ? power_vote(sa.power) : 0;
    res.scratch_bytes = (uint32_t)lfnet_scratch_bytes(g_net);
out:
    memcpy(pra[2].buf.pv, &res, sizeof res);
    return 0;
}

static int do_run(dsp_remote_arg *pra) {
    lfrife_run_args ra;
    memcpy(&ra, pra[0].buf.pv, sizeof ra);
    lfrife_run_stats st;
    memset(&st, 0, sizeof st);
    int rc = 0;
    if (!g_net || ra.w != (uint32_t)g_net->W || ra.h != (uint32_t)g_net->H) {
        st.rc = -2;
    } else if (qurt_hvx_lock(QURT_HVX_MODE_128B)) {
        st.rc = -4;
    } else {
        float *flow = (float *)pra[3].buf.pv;
        float *mask = flow + (size_t)4 * ra.w * ra.h;
        st.rc = lfnet_run(g_net, (const uint8_t *)pra[1].buf.pv, (const uint8_t *)pra[2].buf.pv, (int)ra.stride, ra.t,
                          flow, mask, (ra.flags & 1) ? &st : 0);
        st.nthreads = (uint32_t)lfpool_nthreads(&g_net->pool);
        qurt_hvx_unlock();
    }
    memcpy(pra[4].buf.pv, &st, sizeof st);
    return rc;
}

EXPORT int lfrife_skel_handle_invoke(uint64_t h, uint32_t sc, dsp_remote_arg *pra) {
    const uint32_t method = LF_SCALARS_METHOD(sc);
    (void)h;
    switch (method) {
    case LFRIFE_M_OPEN:
        if (LF_SCALARS_INBUFS(sc) != 2 || LF_SCALARS_OUTHANDLES(sc) != 1) return 0x20;
        g_open_count++;
        pra[2].h64 = 0x4c520000u + (uint32_t)g_open_count;
        return 0;
    case LFRIFE_M_CLOSE:
        if (g_power_corner) power_vote(0);
        return 0;
    case LFRIFE_M_PROBE: {
        if (LF_SCALARS_INBUFS(sc) != 2 || LF_SCALARS_OUTBUFS(sc) != 1) return 0x21;
        lfrife_probe_args pa;
        memcpy(&pa, pra[0].buf.pv, sizeof pa);
        return probe(&pa, pra[1].buf.pv, pra[1].buf.nLen, pra[2].buf.pv, pra[2].buf.nLen);
    }
    case LFRIFE_M_SETUP:
        if (LF_SCALARS_INBUFS(sc) != 2 || LF_SCALARS_OUTBUFS(sc) != 1) return 0x22;
        return do_setup(pra);
    case LFRIFE_M_RUN:
        if (LF_SCALARS_INBUFS(sc) != 3 || LF_SCALARS_OUTBUFS(sc) != 2) return 0x23;
        return do_run(pra);
    default:
        return 0x2f;
    }
}
