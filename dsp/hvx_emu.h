/*
 * SPDX-License-Identifier: MIT
 *
 * Host (x86) emulation of the handful of HVX intrinsics used by dsp/lfrife_net.c,
 * so the network logic and tensor layouts can be tested against the golden
 * outputs without a Hexagon target. qf16 is modelled as a float rounded to a
 * 10-bit significand after every operation (the hardware format has an 11-bit
 * signed mantissa); the real rounding behaviour is only measurable on the device.
 */
#ifndef LF_HVX_EMU_H
#define LF_HVX_EMU_H
#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    union {
        uint16_t h[64]; /* hf / halfword view */
        uint32_t w[32]; /* word / sf bits view (same bytes) */
        uint8_t b[128];
    };
    float q[64];  /* qf16 value (model) */
    float qs[32]; /* qf32 value (model); sf/word results live in w, not here */
} HVX_Vector;
typedef HVX_Vector HVX_UVector;
typedef struct { uint8_t m[128]; } HVX_VectorPred;
typedef struct {
    HVX_Vector v[2];
} HVX_VectorPair;

#ifndef LF_EMU_QBITS
#define LF_EMU_QBITS 10
#endif

static inline float lfe_h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 31, m = h & 0x3ff;
    union { float f; uint32_t u; } v;
    if (e == 0) {
        float f = ldexpf((float)m, -24);
        return s ? -f : f;
    }
    if (e == 31) { v.u = s | 0x7f800000u | (m << 13); return v.f; }
    v.u = s | ((e + 112) << 23) | (m << 13);
    return v.f;
}
static inline uint16_t lfe_f2h(float f) {
    union { float f; uint32_t u; } v = {f};
    uint32_t s = (v.u >> 16) & 0x8000u;
    float a = fabsf(f);
    if (a != a) return (uint16_t)(s | 0x7e00);
    if (a >= 65520.0f) return (uint16_t)(s | 0x7c00);
    if (a < 6.103515625e-05f) { /* subnormal range: multiples of 2^-24 */
        float r = rintf(ldexpf(a, 24));
        return (uint16_t)(s | (uint16_t)r);
    }
    int e;
    float m = frexpf(a, &e); /* a = m * 2^e, m in [0.5,1) */
    float r = rintf(ldexpf(m, 11)); /* 11-bit significand */
    if (r >= 2048.0f) { r = 1024.0f; e++; }
    int he = e - 1 + 15;
    if (he >= 31) return (uint16_t)(s | 0x7c00);
    return (uint16_t)(s | (he << 10) | ((int)r - 1024));
}
static inline float lfe_qround(float x) {
    if (x == 0.0f || x != x || isinf(x)) return x;
    int e;
    float m = frexpf(x, &e);
    return ldexpf(rintf(ldexpf(m, LF_EMU_QBITS + 1)), e - (LF_EMU_QBITS + 1));
}

static inline HVX_Vector Q6_V_vzero(void) { HVX_Vector v; memset(&v, 0, sizeof v); return v; }
static inline uint32_t Q6_R_combine_RlRl(uint32_t a, uint32_t b) { return (a << 16) | (b & 0xffffu); }
static inline HVX_Vector Q6_Vqf16_vmpy_VhfRhf(HVX_Vector a, uint32_t rt) {
    HVX_Vector r = Q6_V_vzero();
    for (int i = 0; i < 64; i++)
        r.q[i] = lfe_qround(lfe_h2f(a.h[i]) * lfe_h2f((uint16_t)((i & 1) ? (rt >> 16) : (rt & 0xffff))));
    return r;
}
static inline HVX_Vector Q6_Vqf16_vmpy_VhfVhf(HVX_Vector a, HVX_Vector b) {
    HVX_Vector r = Q6_V_vzero();
    for (int i = 0; i < 64; i++) r.q[i] = lfe_qround(lfe_h2f(a.h[i]) * lfe_h2f(b.h[i]));
    return r;
}
static inline HVX_Vector Q6_Vqf16_vadd_Vqf16Vqf16(HVX_Vector a, HVX_Vector b) {
    HVX_Vector r = Q6_V_vzero();
    for (int i = 0; i < 64; i++) r.q[i] = lfe_qround(a.q[i] + b.q[i]);
    return r;
}
static inline HVX_Vector Q6_Vqf16_vadd_Vqf16Vhf(HVX_Vector a, HVX_Vector b) {
    HVX_Vector r = Q6_V_vzero();
    for (int i = 0; i < 64; i++) r.q[i] = lfe_qround(a.q[i] + lfe_h2f(b.h[i]));
    return r;
}
static inline HVX_Vector Q6_Vqf16_vadd_VhfVhf(HVX_Vector a, HVX_Vector b) {
    HVX_Vector r = Q6_V_vzero();
    for (int i = 0; i < 64; i++) r.q[i] = lfe_qround(lfe_h2f(a.h[i]) + lfe_h2f(b.h[i]));
    return r;
}
static inline HVX_Vector Q6_Vqf16_vsub_Vqf16Vqf16(HVX_Vector a, HVX_Vector b) {
    HVX_Vector r = Q6_V_vzero();
    for (int i = 0; i < 64; i++) r.q[i] = lfe_qround(a.q[i] - b.q[i]);
    return r;
}
static inline HVX_Vector Q6_Vqf16_vsub_VhfVhf(HVX_Vector a, HVX_Vector b) {
    HVX_Vector r = Q6_V_vzero();
    for (int i = 0; i < 64; i++) r.q[i] = lfe_qround(lfe_h2f(a.h[i]) - lfe_h2f(b.h[i]));
    return r;
}
static inline HVX_Vector Q6_Vhf_equals_Vqf16(HVX_Vector a) {
    HVX_Vector r = Q6_V_vzero();
    for (int i = 0; i < 64; i++) r.h[i] = lfe_f2h(a.q[i]);
    return r;
}
static inline HVX_Vector Q6_Vhf_vmax_VhfVhf(HVX_Vector a, HVX_Vector b) {
    HVX_Vector r = Q6_V_vzero();
    for (int i = 0; i < 64; i++) r.h[i] = lfe_h2f(a.h[i]) >= lfe_h2f(b.h[i]) ? a.h[i] : b.h[i];
    return r;
}
/* ---- 32-bit lanes: qf32, sf bits, integer, permutes (semantics measured on the device, see probe 20) ---- */
static inline float lfe_w2f(uint32_t u) { union { float f; uint32_t u; } v = {.u = u}; return v.f; }
static inline uint32_t lfe_f2w(float f) { union { float f; uint32_t u; } v = {.f = f}; return v.u; }
#define LFE_MAP(name, expr) \
    static inline HVX_Vector name(HVX_Vector a, HVX_Vector b) { HVX_Vector r = Q6_V_vzero(); for (int i = 0; i < 32; i++) r.qs[i] = (expr); (void)b; return r; }
LFE_MAP(Q6_Vqf32_vadd_VsfVsf, lfe_w2f(a.w[i]) + lfe_w2f(b.w[i]))
LFE_MAP(Q6_Vqf32_vsub_VsfVsf, lfe_w2f(a.w[i]) - lfe_w2f(b.w[i]))
LFE_MAP(Q6_Vqf32_vmpy_VsfVsf, lfe_w2f(a.w[i]) * lfe_w2f(b.w[i]))
LFE_MAP(Q6_Vqf32_vadd_Vqf32Vqf32, a.qs[i] + b.qs[i])
LFE_MAP(Q6_Vqf32_vsub_Vqf32Vqf32, a.qs[i] - b.qs[i])
LFE_MAP(Q6_Vqf32_vmpy_Vqf32Vqf32, a.qs[i] * b.qs[i])
LFE_MAP(Q6_Vqf32_vadd_Vqf32Vsf, a.qs[i] + lfe_w2f(b.w[i]))
LFE_MAP(Q6_Vqf32_vsub_Vqf32Vsf, a.qs[i] - lfe_w2f(b.w[i]))
LFE_MAP(Q6_Vqf32_vsub_VsfVqf32, lfe_w2f(a.w[i]) - b.qs[i])
#undef LFE_MAP
static inline HVX_Vector Q6_Vqf32_vmpy_VsfRsf(HVX_Vector a, uint32_t rt) {
    HVX_Vector r = Q6_V_vzero();
    for (int i = 0; i < 32; i++) r.qs[i] = lfe_w2f(a.w[i]) * lfe_w2f(rt);
    return r;
}
static inline HVX_Vector Q6_Vsf_equals_Vqf32(HVX_Vector a) { HVX_Vector r = Q6_V_vzero(); for (int i = 0; i < 32; i++) r.w[i] = lfe_f2w(a.qs[i]); return r; }
static inline HVX_Vector Q6_Vsf_equals_Vw(HVX_Vector a) { HVX_Vector r = Q6_V_vzero(); for (int i = 0; i < 32; i++) r.w[i] = lfe_f2w((float)(int32_t)a.w[i]); return r; }
static inline HVX_Vector Q6_Vw_equals_Vsf(HVX_Vector a) { /* truncates toward zero, saturates */
    HVX_Vector r = Q6_V_vzero();
    for (int i = 0; i < 32; i++) {
        float f = lfe_w2f(a.w[i]);
        r.w[i] = f >= 2147483520.0f ? 0x7fffffffu : f <= -2147483648.0f ? 0x80000000u : (uint32_t)(int32_t)f;
    }
    return r;
}
static inline HVX_Vector Q6_Vsf_vmax_VsfVsf(HVX_Vector a, HVX_Vector b) { HVX_Vector r = Q6_V_vzero(); for (int i = 0; i < 32; i++) r.w[i] = lfe_w2f(a.w[i]) >= lfe_w2f(b.w[i]) ? a.w[i] : b.w[i]; return r; }
static inline HVX_Vector Q6_Vsf_vmin_VsfVsf(HVX_Vector a, HVX_Vector b) { HVX_Vector r = Q6_V_vzero(); for (int i = 0; i < 32; i++) r.w[i] = lfe_w2f(a.w[i]) <= lfe_w2f(b.w[i]) ? a.w[i] : b.w[i]; return r; }
static inline HVX_Vector Q6_V_vsplat_R(uint32_t x) { HVX_Vector r = Q6_V_vzero(); for (int i = 0; i < 32; i++) r.w[i] = x; return r; }
static inline HVX_Vector Q6_Vh_vsplat_R(uint32_t x) { HVX_Vector r = Q6_V_vzero(); for (int i = 0; i < 64; i++) r.h[i] = (uint16_t)x; return r; }
static inline HVX_Vector Q6_V_vand_VV(HVX_Vector a, HVX_Vector b) { HVX_Vector r = Q6_V_vzero(); for (int i = 0; i < 32; i++) r.w[i] = a.w[i] & b.w[i]; return r; }
static inline HVX_Vector Q6_V_vor_VV(HVX_Vector a, HVX_Vector b) { HVX_Vector r = Q6_V_vzero(); for (int i = 0; i < 32; i++) r.w[i] = a.w[i] | b.w[i]; return r; }
static inline HVX_Vector Q6_Vw_vadd_VwVw(HVX_Vector a, HVX_Vector b) { HVX_Vector r = Q6_V_vzero(); for (int i = 0; i < 32; i++) r.w[i] = a.w[i] + b.w[i]; return r; }
static inline HVX_Vector Q6_Vw_vsub_VwVw(HVX_Vector a, HVX_Vector b) { HVX_Vector r = Q6_V_vzero(); for (int i = 0; i < 32; i++) r.w[i] = a.w[i] - b.w[i]; return r; }
static inline HVX_Vector Q6_Vw_vmin_VwVw(HVX_Vector a, HVX_Vector b) { HVX_Vector r = Q6_V_vzero(); for (int i = 0; i < 32; i++) r.w[i] = (int32_t)a.w[i] < (int32_t)b.w[i] ? a.w[i] : b.w[i]; return r; }
static inline HVX_Vector Q6_Vw_vmax_VwVw(HVX_Vector a, HVX_Vector b) { HVX_Vector r = Q6_V_vzero(); for (int i = 0; i < 32; i++) r.w[i] = (int32_t)a.w[i] > (int32_t)b.w[i] ? a.w[i] : b.w[i]; return r; }
static inline HVX_Vector Q6_Vw_vasl_VwR(HVX_Vector a, uint32_t n) { HVX_Vector r = Q6_V_vzero(); for (int i = 0; i < 32; i++) r.w[i] = a.w[i] << (n & 31); return r; }
static inline HVX_Vector Q6_Vuw_vlsr_VuwR(HVX_Vector a, uint32_t n) { HVX_Vector r = Q6_V_vzero(); for (int i = 0; i < 32; i++) r.w[i] = a.w[i] >> (n & 31); return r; }
static inline HVX_VectorPair Q6_W_vcombine_VV(HVX_Vector hi, HVX_Vector lo) { HVX_VectorPair p; p.v[0] = lo; p.v[1] = hi; return p; }
static inline HVX_Vector Q6_V_lo_W(HVX_VectorPair p) { return p.v[0]; }
static inline HVX_Vector Q6_V_hi_W(HVX_VectorPair p) { return p.v[1]; }
/* widening hf*hf -> qf32: lo = even halfword lanes, hi = odd lanes */
static inline HVX_VectorPair Q6_Wqf32_vmpy_VhfVhf(HVX_Vector a, HVX_Vector b) {
    HVX_VectorPair r;
    r.v[0] = Q6_V_vzero();
    r.v[1] = Q6_V_vzero();
    for (int i = 0; i < 32; i++) {
        r.v[0].qs[i] = lfe_h2f(a.h[2 * i]) * lfe_h2f(b.h[2 * i]);
        r.v[1].qs[i] = lfe_h2f(a.h[2 * i + 1]) * lfe_h2f(b.h[2 * i + 1]);
    }
    return r;
}
/* pair of qf32 -> hf, re-interleaved: h[2i] = lo[i], h[2i+1] = hi[i] */
static inline HVX_Vector Q6_Vhf_equals_Wqf32(HVX_VectorPair p) {
    HVX_Vector r = Q6_V_vzero();
    for (int i = 0; i < 32; i++) { r.h[2 * i] = lfe_f2h(p.v[0].qs[i]); r.h[2 * i + 1] = lfe_f2h(p.v[1].qs[i]); }
    return r;
}
/* vshuff(Vu,Vv,-n): stream = Vv[0],Vu[0],Vv[1],Vu[1],... in n-byte elements; lo = first 128 bytes */
static inline HVX_VectorPair Q6_W_vshuff_VVR(HVX_Vector u, HVX_Vector v, int32_t rt) {
    const int n = -rt, ne = 128 / n;
    uint8_t st[256];
    for (int i = 0; i < ne; i++) { memcpy(st + (2 * i) * n, v.b + i * n, (size_t)n); memcpy(st + (2 * i + 1) * n, u.b + i * n, (size_t)n); }
    HVX_VectorPair r;
    r.v[0] = Q6_V_vzero();
    r.v[1] = Q6_V_vzero();
    memcpy(r.v[0].b, st, 128);
    memcpy(r.v[1].b, st + 128, 128);
    return r;
}
/* vdeal(Vu,Vv,-n): stream = Vv elements then Vu elements; lo = even elements, hi = odd elements */
static inline HVX_VectorPair Q6_W_vdeal_VVR(HVX_Vector u, HVX_Vector v, int32_t rt) {
    const int n = -rt, ne = 128 / n;
    uint8_t st[256];
    memcpy(st, v.b, 128);
    memcpy(st + 128, u.b, 128);
    HVX_VectorPair r;
    r.v[0] = Q6_V_vzero();
    r.v[1] = Q6_V_vzero();
    for (int i = 0; i < ne; i++) { memcpy(r.v[0].b + i * n, st + (2 * i) * n, (size_t)n); memcpy(r.v[1].b + i * n, st + (2 * i + 1) * n, (size_t)n); }
    return r;
}
static inline HVX_Vector Q6_V_valign_VVR(HVX_Vector u, HVX_Vector v, uint32_t rt) { /* (u:v) >> rt bytes */
    HVX_Vector r = Q6_V_vzero();
    uint8_t st[256];
    memcpy(st, v.b, 128);
    memcpy(st + 128, u.b, 128);
    memcpy(r.b, st + (rt & 127), 128);
    return r;
}
static inline HVX_Vector Q6_V_vror_VR(HVX_Vector a, uint32_t rt) {
    HVX_Vector r = Q6_V_vzero();
    for (int i = 0; i < 128; i++) r.b[i] = a.b[(i + rt) & 127];
    return r;
}
static inline HVX_VectorPred Q6_Q_vcmp_gt_VwVw(HVX_Vector a, HVX_Vector b) {
    HVX_VectorPred q;
    for (int i = 0; i < 32; i++) for (int k = 0; k < 4; k++) q.m[4 * i + k] = (int32_t)a.w[i] > (int32_t)b.w[i];
    return q;
}
static inline HVX_Vector Q6_V_vmux_QVV(HVX_VectorPred q, HVX_Vector a, HVX_Vector b) {
    HVX_Vector r = Q6_V_vzero();
    for (int i = 0; i < 128; i++) r.b[i] = q.m[i] ? a.b[i] : b.b[i];
    return r;
}
static inline HVX_Vector lf_vload(const void *p) { HVX_Vector v = Q6_V_vzero(); memcpy(v.h, p, 128); return v; }
static inline void lf_vstore(void *p, HVX_Vector v) { memcpy(p, v.h, 128); }
static inline HVX_Vector lf_vloadu(const void *p) { return lf_vload(p); }
static inline void lf_vstoreu(void *p, HVX_Vector v) { lf_vstore(p, v); }
#define LF_HVX_ALIGN 128
#endif
