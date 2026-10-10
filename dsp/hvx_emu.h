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
    uint16_t h[64]; /* hf / halfword view */
    float q[64];    /* qf16 view (model) */
} HVX_Vector;
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
static inline HVX_Vector lf_vload(const void *p) { HVX_Vector v = Q6_V_vzero(); memcpy(v.h, p, 128); return v; }
static inline void lf_vstore(void *p, HVX_Vector v) { memcpy(p, v.h, 128); }
#define LF_HVX_ALIGN 128
#endif
