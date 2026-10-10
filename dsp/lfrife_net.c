/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * RIFE v4.6 IFNet flow/mask network on HVX.
 *
 * Tensors are NHWC fp16 with a one pixel zero halo; channel strides are padded to
 * multiples of 64 where a tensor is written/read as full HVX vectors (output
 * channels sit on the 64 halfword lanes). Convolutions broadcast the input
 * activation (scalar, "rt" form of the qf16 multiply) against a 64-lane weight
 * vector and accumulate in qf16; see docs/design.md.
 *
 * Only freestanding C is used (no libc, no libm, no divisions) so the same source
 * links into the Hexagon skel and builds on the host with -DLF_HOST for testing.
 */
#include "lfrife_net.h"

#include "../inc/lfrife_weights.h"
#include "hvx.h"
#include "lfpool.h"

#ifdef LF_HOST
#include <stdlib.h>
#include <string.h>
static void *lf_alloc(size_t n) { void *p = aligned_alloc(128, (n + 127) & ~(size_t)127); if (p) memset(p, 0, n); return p; }
static void lf_free(void *p) { free(p); }
static inline uint64_t lf_now(void) { return 0; }
#else
#ifndef LF_UNITY
extern void *memalign(size_t, size_t);
extern void free(void *);
extern void *memset(void *, int, size_t);
extern void *memcpy(void *, const void *, size_t);
#endif
static void *lf_alloc(size_t n) {
    void *p = memalign(128, (n + 127) & ~(size_t)127);
    if (p) memset(p, 0, n);
    return p;
}
static void lf_free(void *p) { free(p); }
static inline uint64_t lf_now(void) { return read_utimer(); }
#endif

/* ---- scalar helpers (no libm) -------------------------------------------- */
static inline float lf_h2f(uint16_t h) {
    union { float f; uint32_t u; } v;
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 31, m = h & 0x3ff;
    if (e == 0) { /* zero / subnormal: m * 2^-24 */
        v.f = (float)m * (1.0f / 16777216.0f);
        v.u |= s;
        return v.f;
    }
    v.u = s | ((e + 112) << 23) | (m << 13);
    return v.f;
}
static inline uint16_t lf_f2h(float f) { /* round to nearest even, saturating */
    union { float f; uint32_t u; } v = {f};
    uint32_t s = (v.u >> 16) & 0x8000u, a = v.u & 0x7fffffffu;
    if (a >= 0x477ff000u) return (uint16_t)(s | 0x7bffu); /* saturate to max finite */
    if (a < 0x38800000u) { /* subnormal half */
        if (a < 0x33000000u) return (uint16_t)s;
        float x = v.f < 0 ? -v.f : v.f;
        uint32_t q = (uint32_t)(x * 16777216.0f + 0.5f);
        return (uint16_t)(s | q);
    }
    uint32_t r = a - 0x38000000u; /* rebias exponent: 127 -> 15 */
    r = (r + 0xfffu + ((r >> 13) & 1u)) >> 13;
    return (uint16_t)(s | r);
}
static inline float lf_recip(float x) { /* x > 0 */
    union { float f; uint32_t u; } v = {x};
    v.u = 0x7ef311c7u - v.u;
    float y = v.f;
    y = y * (2.0f - x * y);
    y = y * (2.0f - x * y);
    y = y * (2.0f - x * y);
    y = y * (2.0f - x * y);
    return y;
}
/* exp(x) for |x| < 80 */
static inline float lf_exp(float x) {
    const float log2e = 1.4426950408889634f;
    float t = x * log2e;
    int i = (int)(t < 0 ? t - 0.5f : t + 0.5f);
    float f = (t - (float)i) * 0.6931471805599453f; /* |f| <= ln2/2 */
    /* degree-7 Taylor: error < 1e-9 relative on |f| <= 0.35 */
    float p = 1.0f + f * (1.0f + f * (0.5f + f * (1.0f / 6.0f + f * (1.0f / 24.0f + f * (1.0f / 120.0f + f * (1.0f / 720.0f + f * (1.0f / 5040.0f)))))));
    union { float f; uint32_t u; } s;
    s.u = (uint32_t)(i + 127) << 23;
    return p * s.f;
}
static inline float lf_sigmoid(float x) {
    if (x > 40.0f) x = 40.0f;
    if (x < -40.0f) x = -40.0f;
    return lf_recip(1.0f + lf_exp(-x));
}

/* ---- data structures ------------------------------------------------------ */
#define NBLK 4
typedef struct {
    const uint16_t *w;   /* HVX weights [ob][tap][ci][64] (deconv: [pu][6][ci][64]) */
    uint16_t *bias;      /* hf [nob*64] (deconv: [64]) built at create time, 128B aligned */
    int cin, cout, nob, stride, type;
} lay;

typedef struct {
    uint16_t *p;
    int h, w, cs;
} tens;

typedef struct {
    int s, hs, ws, cin, c;
    tens X, A0, A1, A2, D[2];
    float *tm; /* [5][hs][ws] */
    int *uy0, *uy1, *ux0, *ux1;
    float *uly, *ulx;
} blk;

struct lfnet {
    int W, H, nt;
    lay L[LFW_NLAYERS];
    blk B[NBLK];
    float *fm; /* [5][H][W] cumulative flow(4) + mask */
    size_t bytes;
    void *allocs[256];
    int nallocs;
    /* per-run state */
    const uint8_t *img0, *img1;
    int stride;
    float t;
    uint16_t t_hf;
    float *flow_out, *mask_out;
    int profile;
    uint64_t gt[LFRIFE_NGROUPS];
    uint64_t last;
    lfpool pool;
};

static void *nalloc(lfnet *n, size_t bytes) {
    void *p = lf_alloc(bytes);
    if (p && n->nallocs < 256) {
        n->allocs[n->nallocs++] = p;
        n->bytes += bytes;
    } else if (p) {
        lf_free(p);
        p = 0;
    }
    return p;
}

static inline int pad64(int c) { return (c + 63) & ~63; }

static int tens_alloc(lfnet *n, tens *t, int h, int w, int cs) {
    t->h = h;
    t->w = w;
    t->cs = cs;
    t->p = nalloc(n, ((size_t)(h + 2) * (w + 2) * cs + 1024) * 2);
    return t->p != 0;
}

void lfnet_destroy(lfnet *n) {
    if (!n) return;
    for (int i = 0; i < n->nallocs; i++) lf_free(n->allocs[i]);
    lf_free(n);
}
size_t lfnet_scratch_bytes(const lfnet *n) { return n->bytes; }

lfnet *lfnet_create(const void *blob, size_t size, int W, int H, int nthreads) {
    const uint8_t *base = (const uint8_t *)blob;
    const lfw_header *hd = (const lfw_header *)blob;
    if (size < sizeof(*hd) || hd->version != LFW_VERSION || hd->kind != LFW_KIND_HVX || hd->nlayers != LFW_NLAYERS ||
        hd->total_bytes != size)
        return 0;
    for (int i = 0; i < 8; i++)
        if (hd->magic[i] != LFW_MAGIC[i]) return 0;
    if (W < 32 || H < 32 || (W & 31) || (H & 31)) return 0;
    lfnet *n = (lfnet *)lf_alloc(sizeof(lfnet));
    if (!n) return 0;
    n->W = W;
    n->H = H;
    n->nt = nthreads < 1 ? 1 : nthreads;
    const lfw_layer *tab = (const lfw_layer *)(base + sizeof(*hd));
    for (int i = 0; i < LFW_NLAYERS; i++) {
        lay *L = &n->L[i];
        L->type = (int)tab[i].type;
        L->cin = (int)tab[i].cin;
        L->cout = (int)tab[i].cout;
        L->stride = (int)tab[i].stride;
        L->nob = L->type == LFW_DECONV ? 1 : (L->cout + 63) / 64;
        L->w = (const uint16_t *)(base + tab[i].w_off);
        const float *b = (const float *)(base + tab[i].b_off); /* [nob][2][32] */
        L->bias = (uint16_t *)nalloc(n, (size_t)L->nob * 128);
        if (!L->bias) goto fail;
        for (int ob = 0; ob < L->nob; ob++)
            for (int k = 0; k < 32; k++) {
                L->bias[ob * 64 + 2 * k] = lf_f2h(b[ob * 64 + k]);
                L->bias[ob * 64 + 2 * k + 1] = lf_f2h(b[ob * 64 + 32 + k]);
            }
    }
    n->fm = (float *)nalloc(n, (size_t)5 * W * H * 4);
    if (!n->fm) goto fail;
    for (int b = 0; b < NBLK; b++) {
        blk *k = &n->B[b];
        k->s = (int)hd->block_scale[b];
        k->cin = (int)hd->block_cin[b];
        k->c = (int)hd->block_c[b];
        k->hs = H / k->s;
        k->ws = W / k->s;
        const int hs = k->hs, ws = k->ws, c = k->c;
        if (!tens_alloc(n, &k->X, hs, ws, (k->cin + 1) & ~1)) goto fail;
        if (!tens_alloc(n, &k->A0, hs / 2, ws / 2, pad64(c / 2))) goto fail;
        if (!tens_alloc(n, &k->A1, hs / 4, ws / 4, pad64(c))) goto fail;
        if (!tens_alloc(n, &k->A2, hs / 4, ws / 4, pad64(c))) goto fail;
        for (int pu = 0; pu < 2; pu++)
            if (!tens_alloc(n, &k->D[pu], hs / 4, ws / 4, 64)) goto fail;
        k->tm = (float *)nalloc(n, (size_t)5 * hs * ws * 4);
        if (!k->tm) goto fail;
        if (k->s > 1) {
            k->uy0 = (int *)nalloc(n, (size_t)H * 4);
            k->uy1 = (int *)nalloc(n, (size_t)H * 4);
            k->ux0 = (int *)nalloc(n, (size_t)W * 4);
            k->ux1 = (int *)nalloc(n, (size_t)W * 4);
            k->uly = (float *)nalloc(n, (size_t)H * 4);
            k->ulx = (float *)nalloc(n, (size_t)W * 4);
            if (!k->uy0 || !k->uy1 || !k->ux0 || !k->ux1 || !k->uly || !k->ulx) goto fail;
            const float inv = 1.0f / (float)k->s;
            for (int pass = 0; pass < 2; pass++) {
                const int len = pass ? W : H, lim = pass ? ws : hs;
                int *i0 = pass ? k->ux0 : k->uy0, *i1 = pass ? k->ux1 : k->uy1;
                float *lf = pass ? k->ulx : k->uly;
                for (int i = 0; i < len; i++) {
                    float sx = ((float)i + 0.5f) * inv - 0.5f;
                    if (sx < 0.0f) sx = 0.0f;
                    int a = (int)sx;
                    if (a > lim - 1) a = lim - 1;
                    int b2 = a + 1 < lim ? a + 1 : lim - 1;
                    i0[i] = a;
                    i1[i] = b2;
                    lf[i] = sx - (float)a;
                }
            }
        }
    }
    return n;
fail:
    lfnet_destroy(n);
    return 0;
}

/* ---- convolution tile kernel ---------------------------------------------- */
enum { EPI_LEAKY = 0, EPI_RES = 1, EPI_NONE = 2 };

#define DUP(x) Q6_R_combine_RlRl((x), (x))
#define SLOPE_RT 0x32663266u /* hf 0.2 in both halves */

static inline __attribute__((always_inline)) void tile_n(const int P, const uint16_t *base, int pstep, const int32_t *toff,
                                                         int ntap, int cin, const uint16_t *wv, HVX_Vector bias,
                                                         uint16_t *outp, int ostep, const uint16_t *resp, int rstep,
                                                         int mode) {
    HVX_Vector acc[16];
    HVX_Vector b0 = Q6_Vqf16_vadd_VhfVhf(bias, Q6_V_vzero());
    for (int p = 0; p < P; p++) acc[p] = b0;
    for (int t = 0; t < ntap; t++) {
        const uint16_t *tb = base + toff[t];
        const uint16_t *wt = wv + (size_t)t * cin * 64;
        for (int ci = 0; ci < cin; ci++) {
            HVX_Vector w = lf_vload(wt + (size_t)ci * 64);
#pragma clang loop unroll(full)
            for (int p = 0; p < P; p++) {
                uint32_t a = tb[(size_t)p * pstep + ci];
                acc[p] = Q6_Vqf16_vadd_Vqf16Vqf16(acc[p], Q6_Vqf16_vmpy_VhfRhf(w, DUP(a)));
            }
        }
    }
#pragma clang loop unroll(full)
    for (int p = 0; p < P; p++) {
        HVX_Vector a = acc[p];
        if (mode == EPI_RES) a = Q6_Vqf16_vadd_Vqf16Vhf(a, lf_vload(resp + (size_t)p * rstep));
        HVX_Vector x = Q6_Vhf_equals_Vqf16(a);
        if (mode != EPI_NONE) {
            HVX_Vector neg = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfRhf(x, SLOPE_RT));
            x = Q6_Vhf_vmax_VhfVhf(x, neg);
        }
        lf_vstore(outp + (size_t)p * ostep, x);
    }
}

typedef struct {
    const lay *L;
    const tens *in;
    tens *out;
    const tens *res;
    int stride;
    int mode;
    int ntap;
    int32_t toff[9];
    int pu;
} cjob;

/* one output row y of one 64-channel output block */
static void conv_row(const cjob *j, int y, int ob) {
    const tens *in = j->in;
    tens *out = j->out;
    const lay *L = j->L;
    const int s = j->stride, cin = L->cin, wo = out->w;
    const uint16_t *wbase = L->w + (size_t)ob * j->ntap * cin * 64;
    if (L->type == LFW_DECONV) wbase = L->w + (size_t)j->pu * j->ntap * cin * 64;
    const uint16_t *wv = wbase;
    HVX_Vector bias = lf_vload(L->bias + ob * 64);
    const int prow = in->w + 2;
    const uint16_t *inrow = in->p + (size_t)(y * s) * prow * in->cs;
    uint16_t *orow = out->p + ((size_t)(y + 1) * (wo + 2) + 1) * out->cs + ob * 64;
    const uint16_t *rrow = j->res ? j->res->p + ((size_t)(y + 1) * (j->res->w + 2) + 1) * j->res->cs + ob * 64 : 0;
    const int pstep = s * in->cs, ostep = out->cs, rstep = j->res ? j->res->cs : 0;
    int x = 0;
    while (x < wo) {
        const int rem = wo - x;
        const uint16_t *b = inrow + (size_t)x * pstep;
        uint16_t *o = orow + (size_t)x * ostep;
        const uint16_t *r = rrow ? rrow + (size_t)x * rstep : 0;
#define TILE(PP)                                                                                      \
    tile_n(PP, b, pstep, j->toff, j->ntap, cin, wv, bias, o, ostep, r, rstep, j->mode);              \
    x += PP;                                                                                          \
    continue;
        if (rem >= 16) { TILE(16) }
        if (rem >= 10 && (rem == 10 || rem == 11 || rem == 12 || rem == 13)) { TILE(10) }
        if (rem >= 8) { TILE(8) }
        if (rem >= 4) { TILE(4) }
        if (rem >= 2) { TILE(2) }
        TILE(1)
#undef TILE
    }
}

static void cjob_init_conv3(cjob *j, const lay *L, const tens *in, tens *out, const tens *res, int mode) {
    j->L = L;
    j->in = in;
    j->out = out;
    j->res = res;
    j->stride = L->stride;
    j->mode = mode;
    j->ntap = 9;
    j->pu = 0;
    for (int dy = 0; dy < 3; dy++)
        for (int dx = 0; dx < 3; dx++) j->toff[dy * 3 + dx] = (dy * (in->w + 2) + dx) * in->cs;
}
static void cjob_init_deconv(cjob *j, const lay *L, const tens *in, tens *out, int pu) {
    j->L = L;
    j->in = in;
    j->out = out;
    j->res = 0;
    j->stride = 1;
    j->mode = EPI_NONE;
    j->ntap = 6;
    j->pu = pu;
    for (int dyi = 0; dyi < 2; dyi++)
        for (int dxi = 0; dxi < 3; dxi++) j->toff[dyi * 3 + dxi] = ((dyi + pu) * (in->w + 2) + dxi) * in->cs;
}

/* ---- stage helpers ---------------------------------------------------------- */
static inline void stage_end(lfnet *n, lfpool *p, int tid, int slot) {
    lfpool_barrier(p);
    if (tid == 0 && n->profile) {
        uint64_t t = lf_now();
        n->gt[slot] += t - n->last;
        n->last = t;
    }
}

static void run_conv(lfnet *n, lfpool *p, int tid, const cjob *j) {
    const int rows = j->out->h, nob = j->L->type == LFW_DECONV ? 1 : j->L->nob;
    const uint32_t items = (uint32_t)rows * (uint32_t)nob;
    (void)n; (void)tid;
    for (;;) {
        uint32_t i = lfpool_next(p);
        if (i >= items) break;
        /* ob-major so that threads working at the same time share weights */
        conv_row(j, (int)(i % (uint32_t)rows), (int)(i / (uint32_t)rows));
    }
}

static void build_input(lfnet *n, int b, int y) {
    blk *k = &n->B[b];
    const int s = k->s, ws = k->ws, W = n->W, H = n->H, cs = k->X.cs;
    const int o0 = s > 1 ? s / 2 - 1 : 0, nn = s > 1 ? 2 : 1;
    const float cnt = (nn == 2) ? 0.25f : 1.0f;
    const float sc = cnt * (1.0f / 255.0f);
    const float invs = 1.0f / (float)s;
    uint16_t *row = k->X.p + ((size_t)(y + 1) * (ws + 2) + 1) * cs;
    for (int x = 0; x < ws; x++) {
        float acc[12];
        for (int c = 0; c < 12; c++) acc[c] = 0.0f;
        float accm = 0.0f;
        for (int dy = 0; dy < nn; dy++)
            for (int dx = 0; dx < nn; dx++) {
                const int py = y * s + o0 + dy, px = x * s + o0 + dx;
                if (b == 0) {
                    const uint8_t *a = n->img0 + (size_t)py * n->stride + (size_t)px * 4;
                    const uint8_t *c1 = n->img1 + (size_t)py * n->stride + (size_t)px * 4;
                    acc[0] += (float)a[0]; acc[1] += (float)a[1]; acc[2] += (float)a[2];
                    acc[3] += (float)c1[0]; acc[4] += (float)c1[1]; acc[5] += (float)c1[2];
                } else {
                    const size_t pi = (size_t)py * W + px, pl = (size_t)W * H;
                    const float f0 = n->fm[pi], f1 = n->fm[pl + pi], f2 = n->fm[2 * pl + pi], f3 = n->fm[3 * pl + pi];
                    for (int im = 0; im < 2; im++) {
                        const uint8_t *img = im ? n->img1 : n->img0;
                        float sx = (float)px + (im ? f2 : f0), sy = (float)py + (im ? f3 : f1);
                        if (sx < 0.0f) sx = 0.0f;
                        if (sx > (float)(W - 1)) sx = (float)(W - 1);
                        if (sy < 0.0f) sy = 0.0f;
                        if (sy > (float)(H - 1)) sy = (float)(H - 1);
                        const int x0 = (int)sx, y0 = (int)sy;
                        const int x1 = x0 + 1 < W ? x0 + 1 : W - 1, y1 = y0 + 1 < H ? y0 + 1 : H - 1;
                        const float ax = sx - (float)x0, ay = sy - (float)y0;
                        const uint8_t *p00 = img + (size_t)y0 * n->stride + (size_t)x0 * 4;
                        const uint8_t *p01 = img + (size_t)y0 * n->stride + (size_t)x1 * 4;
                        const uint8_t *p10 = img + (size_t)y1 * n->stride + (size_t)x0 * 4;
                        const uint8_t *p11 = img + (size_t)y1 * n->stride + (size_t)x1 * 4;
                        for (int c = 0; c < 3; c++) {
                            float top = (float)p00[c] + ((float)p01[c] - (float)p00[c]) * ax;
                            float bot = (float)p10[c] + ((float)p11[c] - (float)p10[c]) * ax;
                            acc[im * 3 + c] += top + (bot - top) * ay;
                        }
                    }
                    accm += n->fm[4 * pl + pi];
                    acc[8] += f0; acc[9] += f1; acc[10] += f2; acc[11] += f3;
                }
            }
        uint16_t *xp = row + (size_t)x * cs;
        for (int c = 0; c < 6; c++) xp[c] = lf_f2h(acc[c] * sc);
        xp[6] = n->t_hf;
        if (b > 0) {
            xp[7] = lf_f2h(accm * cnt);
            const float fs = cnt * invs;
            for (int c = 8; c < 12; c++) xp[c] = lf_f2h(acc[c] * fs);
        }
    }
}

/* ---- scatter + upsample -------------------------------------------------- */
static void scatter_rows(lfnet *n, int b, int pu, int y) {
    blk *k = &n->B[b];
    const tens *d = &k->D[pu];
    const int ws = k->ws, hs = k->hs, wq = d->w;
    const uint16_t *row = d->p + ((size_t)(y + 1) * (wq + 2) + 1) * 64;
    for (int x = 0; x < wq; x++) {
        const uint16_t *v = row + (size_t)x * 64;
        for (int pv = 0; pv < 2; pv++)
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++) {
                    const int Y = 4 * y + 2 * pu + i, X = 4 * x + 2 * pv + j;
                    for (int ch = 0; ch < 5; ch++)
                        k->tm[((size_t)ch * hs + Y) * ws + X] = lf_h2f(v[((pv * 2 + i) * 2 + j) * 5 + ch]);
                }
    }
}

static void upsample_row(lfnet *n, int b, int y) {
    blk *k = &n->B[b];
    const int W = n->W, H = n->H, hs = k->hs, ws = k->ws, s = k->s;
    const size_t pl = (size_t)W * H;
    for (int ch = 0; ch < 5; ch++) {
        const float *tm = k->tm + (size_t)ch * hs * ws;
        float *dst = n->fm + ch * pl + (size_t)y * W;
        const float mul = ch < 4 ? (float)s : 1.0f;
        if (s == 1) {
            for (int x = 0; x < W; x++) dst[x] += tm[(size_t)y * ws + x];
            continue;
        }
        const float *r0 = tm + (size_t)k->uy0[y] * ws, *r1 = tm + (size_t)k->uy1[y] * ws;
        const float ly = k->uly[y];
        for (int x = 0; x < W; x++) {
            const int a = k->ux0[x], c = k->ux1[x];
            const float lx = k->ulx[x];
            const float top = r0[a] + (r0[c] - r0[a]) * lx, bot = r1[a] + (r1[c] - r1[a]) * lx;
            dst[x] += (top + (bot - top) * ly) * mul;
        }
    }
}

/* ---- run ---------------------------------------------------------------- */
static void worker(lfpool *p, int tid, void *arg) {
    lfnet *n = (lfnet *)arg;
    for (int b = 0; b < NBLK; b++) {
        blk *k = &n->B[b];
        const lay *Ls = &n->L[b * LFW_LAYERS_PER_BLOCK];
        /* group 0: build X */
        if (b == 0) { /* fm must be zero before the first accumulate; cleared by lfnet_run */ }
        for (;;) {
            uint32_t y = lfpool_next(p);
            if (y >= (uint32_t)k->hs) break;
            build_input(n, b, (int)y);
        }
        stage_end(n, p, tid, b * 4 + 0);
        cjob j;
        cjob_init_conv3(&j, &Ls[0], &k->X, &k->A0, 0, EPI_LEAKY);
        run_conv(n, p, tid, &j);
        lfpool_barrier(p);
        cjob_init_conv3(&j, &Ls[1], &k->A0, &k->A1, 0, EPI_LEAKY);
        run_conv(n, p, tid, &j);
        stage_end(n, p, tid, b * 4 + 1);
        tens *cur = &k->A1, *oth = &k->A2;
        for (int r = 0; r < 8; r++) {
            cjob_init_conv3(&j, &Ls[2 + r], cur, oth, cur, EPI_RES);
            run_conv(n, p, tid, &j);
            lfpool_barrier(p);
            tens *t = cur; cur = oth; oth = t;
        }
        stage_end(n, p, tid, b * 4 + 2);
        for (int pu = 0; pu < 2; pu++) {
            cjob_init_deconv(&j, &Ls[10], cur, &k->D[pu], pu);
            run_conv(n, p, tid, &j);
            lfpool_barrier(p);
        }
        for (int pu = 0; pu < 2; pu++) {
            for (;;) {
                uint32_t y = lfpool_next(p);
                if (y >= (uint32_t)k->D[pu].h) break;
                scatter_rows(n, b, pu, (int)y);
            }
            lfpool_barrier(p);
        }
        for (;;) {
            uint32_t y = lfpool_next(p);
            if (y >= (uint32_t)n->H) break;
            upsample_row(n, b, (int)y);
        }
        stage_end(n, p, tid, b * 4 + 3);
    }
    /* final copy + sigmoid */
    const size_t pl = (size_t)n->W * n->H;
    for (;;) {
        uint32_t y = lfpool_next(p);
        if (y >= (uint32_t)n->H) break;
        const size_t o = (size_t)y * n->W;
        for (int c = 0; c < 4; c++)
            for (int x = 0; x < n->W; x++) n->flow_out[c * pl + o + x] = n->fm[c * pl + o + x];
        for (int x = 0; x < n->W; x++) n->mask_out[o + x] = lf_sigmoid(n->fm[4 * pl + o + x]);
    }
    lfpool_barrier(p);
}

int lfnet_run(lfnet *n, const uint8_t *rgba0, const uint8_t *rgba1, int stride, float t, float *flow, float *mask,
              lfrife_run_stats *stats) {
    n->img0 = rgba0;
    n->img1 = rgba1;
    n->stride = stride;
    n->t = t;
    n->t_hf = lf_f2h(t);
    n->flow_out = flow;
    n->mask_out = mask;
    n->profile = stats != 0;
    for (int i = 0; i < LFRIFE_NGROUPS; i++) n->gt[i] = 0;
    const size_t fmn = (size_t)5 * n->W * n->H;
    for (size_t i = 0; i < fmn; i++) n->fm[i] = 0.0f;
    n->last = lf_now();
    const uint64_t t0 = n->last;
    int rc = lfpool_run(&n->pool, n->nt, worker, n);
    if (stats) {
        stats->rc = rc;
        stats->nthreads = (uint32_t)n->nt;
        stats->total_ticks = lf_now() - t0;
        for (int i = 0; i < LFRIFE_NGROUPS; i++) stats->group_ticks[i] = n->gt[i];
    }
    return rc;
}
