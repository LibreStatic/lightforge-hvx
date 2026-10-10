/*
 * SPDX-License-Identifier: MIT
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
static inline uint64_t lf_cycles(void) { return 0; }
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
static inline uint64_t lf_cycles(void) { return read_pcycle(); }
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

/* deconv output lane ((pv*2+i)*2+j)*5+ch (blob layout) -> ((ch*2+i)*2+pv)*2+j */
static inline int deconv_lane(int l) {
    const int ch = l % 5, q = l / 5, j = q & 1, i = (q >> 1) & 1, pv = q >> 2;
    return ((ch * 2 + i) * 2 + pv) * 2 + j;
}

/* ---- data structures ------------------------------------------------------ */
#define NBLK 4
typedef struct {
    const uint16_t *w;   /* HVX weights [ob][tap][ci][64] (deconv: [pu][6][ci][64]) */
    uint16_t *uw;        /* ResConv only: Winograd F(2x2,3x3) weights G g G^T, hf [ob][16][ci][64], built at create time */
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
    uint16_t *tm; /* hf [5][hs][tmw]: deconv output scattered to planes (rows padded to 64 halfwords) */
    int tmw;
    int *uy0, *uy1; /* vertical bilinear taps per output row */
    float *uly;
    float *uw; /* horizontal taps, 6 vectors of 32 floats: {wm, w0, wp} x {flow (x s), mask} for src px k-1, k, k+1 */
} blk;

/* per-thread scratch for the vector stages */
typedef struct {
    float *S;        /* [11][sw] window sums of one output row (planes: rgb0, rgb1, mask, flow0-3) */
    uint32_t *offs;  /* [32] pixel byte offsets of one chunk */
    uint32_t *gat;   /* [4][32] gathered RGBA words (p00, p01, p10, p11) */
    uint16_t *hfp;   /* [11][64] hf planes of 64 output pixels */
    float *vb;       /* vertically blended source row with padding */
    uint16_t *wv;    /* Winograd: transformed input tiles hf [16 pos][16 tiles][cs] */
    HVX_Vector *wm;  /* Winograd: products qf16 [16 pos][16 tiles] */
} tscr;

struct lfnet {
    int W, H, nt;
    lay L[LFW_NLAYERS];
    blk B[NBLK];
    float *fm; /* [5][H][W] cumulative flow(4) + mask (block 0 assigns, 1-2 accumulate, 3 is fused into the output) */
    int sw;    /* row length of tscr.S */
    uint32_t *iota; /* words 0..31 */
    tscr ts[LFP_MAX];
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
    t->p = nalloc(n, ((size_t)(h + 3) * (w + 2) * cs + 1024) * 2); /* one spare row: odd-sized Winograd tiles read it */
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
    n->nt = nthreads < 1 ? 1 : nthreads > LFP_MAX ? LFP_MAX : nthreads;
    n->nt = lfpool_probe(&n->pool, n->nt); /* threads that can really hold an HVX context at once */
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
        if (L->type == LFW_RESCONV && L->stride == 1) {
            /* U = G g G^T per (ob, ci, lane), G = [[1,0,0],[.5,.5,.5],[.5,-.5,.5],[0,0,1]] */
            const size_t plane = (size_t)L->cin * 64;
            L->uw = (uint16_t *)nalloc(n, (size_t)L->nob * 16 * plane * 2);
            if (!L->uw) goto fail;
            static const float G[4][3] = {{1, 0, 0}, {.5f, .5f, .5f}, {.5f, -.5f, .5f}, {0, 0, 1}};
            for (int ob = 0; ob < L->nob; ob++)
                for (size_t cl = 0; cl < plane; cl++) {
                    float g[3][3], tmp[4][3];
                    for (int t = 0; t < 9; t++) g[t / 3][t % 3] = lf_h2f(L->w[((size_t)ob * 9 + t) * plane + cl]);
                    for (int r = 0; r < 4; r++)
                        for (int kx = 0; kx < 3; kx++) tmp[r][kx] = G[r][0] * g[0][kx] + G[r][1] * g[1][kx] + G[r][2] * g[2][kx];
                    for (int r = 0; r < 4; r++)
                        for (int c = 0; c < 4; c++)
                            L->uw[((size_t)ob * 16 + r * 4 + c) * plane + cl] =
                                lf_f2h(tmp[r][0] * G[c][0] + tmp[r][1] * G[c][1] + tmp[r][2] * G[c][2]);
                }
        }
        if (L->type == LFW_DECONV) { /* lanes -> ((ch*2+i)*4 + pv*2+j): 4 consecutive X per (ch,i) group of 4 halfwords */
            uint16_t *wn = (uint16_t *)nalloc(n, (size_t)2 * 6 * L->cin * 128);
            if (!wn) goto fail;
            for (size_t r = 0; r < (size_t)2 * 6 * L->cin; r++)
                for (int l = 0; l < 40; l++) wn[r * 64 + deconv_lane(l)] = L->w[r * 64 + l];
            L->w = wn;
            uint16_t bn[64];
            for (int l = 0; l < 64; l++) bn[l] = 0;
            for (int l = 0; l < 40; l++) bn[deconv_lane(l)] = L->bias[l];
            for (int l = 0; l < 64; l++) L->bias[l] = bn[l];
        }
    }
    n->iota = (uint32_t *)nalloc(n, 128);
    if (!n->iota) goto fail;
    for (int i = 0; i < 32; i++) n->iota[i] = (uint32_t)i;
    n->sw = (W + 255) & ~255;
    for (int t = 0; t < LFP_MAX; t++) {
        tscr *ts = &n->ts[t];
        const int vbn = ((W / 2 + 63) & ~63) + 192;
        ts->S = (float *)nalloc(n, (size_t)11 * n->sw * 4);
        ts->offs = (uint32_t *)nalloc(n, 128);
        ts->gat = (uint32_t *)nalloc(n, 4 * 128);
        ts->hfp = (uint16_t *)nalloc(n, 11 * 128);
        ts->vb = (float *)nalloc(n, (size_t)vbn * 4);
        ts->wv = (uint16_t *)nalloc(n, (size_t)16 * 16 * 192 * 2);
        ts->wm = (HVX_Vector *)nalloc(n, (size_t)16 * 16 * sizeof(HVX_Vector));
        if (!ts->wv || !ts->wm || !ts->S || !ts->offs || !ts->gat || !ts->hfp || !ts->vb) goto fail;
    }
    n->fm = (float *)nalloc(n, (size_t)5 * W * H * 4);
    if (!n->fm) goto fail;
    for (int b = 0; b < NBLK; b++) {
        blk *k = &n->B[b];
        k->s = (int)hd->block_scale[b];
        k->cin = (int)hd->block_cin[b];
        k->c = (int)hd->block_c[b];
        { int sh = k->s == 8 ? 3 : k->s == 4 ? 2 : k->s == 2 ? 1 : 0; k->hs = H >> sh; k->ws = W >> sh; }
        const int hs = k->hs, ws = k->ws, c = k->c;
        if (!tens_alloc(n, &k->X, hs, ws, (k->cin + 1) & ~1)) goto fail;
        if (!tens_alloc(n, &k->A0, hs / 2, ws / 2, pad64(c / 2))) goto fail;
        if (!tens_alloc(n, &k->A1, hs / 4, ws / 4, pad64(c))) goto fail;
        if (!tens_alloc(n, &k->A2, hs / 4, ws / 4, pad64(c))) goto fail;
        for (int pu = 0; pu < 2; pu++)
            if (!tens_alloc(n, &k->D[pu], hs / 4, ws / 4, 64)) goto fail;
        k->tmw = (ws + 63) & ~63;
        k->tm = (uint16_t *)nalloc(n, (size_t)5 * hs * k->tmw * 2);
        if (!k->tm) goto fail;
        if (k->s > 1) {
            k->uy0 = (int *)nalloc(n, (size_t)H * 4);
            k->uy1 = (int *)nalloc(n, (size_t)H * 4);
            k->uly = (float *)nalloc(n, (size_t)H * 4);
            k->uw = (float *)nalloc(n, 6 * 128);
            if (!k->uy0 || !k->uy1 || !k->uly || !k->uw) goto fail;
            const float inv = 1.0f / (float)k->s;
            for (int i = 0; i < H; i++) {
                float sx = ((float)i + 0.5f) * inv - 0.5f;
                if (sx < 0.0f) sx = 0.0f;
                int a = (int)sx;
                if (a > hs - 1) a = hs - 1;
                k->uy0[i] = a;
                k->uy1[i] = a + 1 < hs ? a + 1 : hs - 1;
                k->uly[i] = sx - (float)a;
            }
            /* horizontal: output px x = s*k + j sits at k + (j+.5)/s - .5, between k-1 and k (a = -1) or k and k+1 (a = 0);
             * the edges clamp because the source row is padded by replication */
            for (int j = 0; j < 32; j++) {
                const float sx = ((float)(j & (k->s - 1)) + 0.5f) * inv - 0.5f;
                const int a = sx < 0.0f ? -1 : 0;
                const float lx = sx - (float)a;
                float wm = 0.0f, w0 = 1.0f - lx, wp = lx;
                if (a < 0) { wm = 1.0f - lx; w0 = lx; wp = 0.0f; }
                for (int m = 0; m < 2; m++) {
                    const float mul = m ? 1.0f : (float)k->s;
                    k->uw[(m * 3 + 0) * 32 + j] = wm * mul;
                    k->uw[(m * 3 + 1) * 32 + j] = w0 * mul;
                    k->uw[(m * 3 + 2) * 32 + j] = wp * mul;
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

/* weights stream from DRAM once per layer; a software prefetch WPF vectors ahead hides the miss latency */
#ifndef WPF
#define WPF 16
#endif
#ifdef LF_HOST
#define LF_PREFETCH(p) ((void)(p))
#else
#define LF_PREFETCH(p) __builtin_prefetch((p), 0, 3)
#endif
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
            LF_PREFETCH(wt + (size_t)(ci + WPF) * 64);
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


#ifndef LF_WINO
#define LF_WINO 1 /* 0: direct 3x3 ResConv (kept for A/B accuracy and timing checks) */
#endif
/* ---- Winograd F(2x2,3x3) ResConv --------------------------------------------------
 * 16 multiplications per 2x2 output tile instead of 36. Per row of tiles: input tiles d (4x4 pixels,
 * 64 channels per vector) -> V = B^T d B (adds only), 16 positions x up to 16 tiles; then for every
 * position the usual broadcast-MAC loop with 16 tile accumulators (the weight vector is reused 16
 * times, exactly like the pixel tiles of tile_n); finally Y = A^T M A, bias, residual, LeakyReLU. */
#define LF_MEMBAR() __asm__ volatile("" ::: "memory") /* vector stores and scalar loads of the same scratch */

static inline __attribute__((always_inline)) void wmac(const int P, const uint16_t *a, int cs, int cin, const uint16_t *wv,
                                                       HVX_Vector *mo) {
    HVX_Vector acc[16];
#pragma clang loop unroll(full)
    for (int p = 0; p < P; p++) acc[p] = Q6_V_vzero();
    for (int ci = 0; ci < cin; ci++) {
        HVX_Vector w = lf_vload(wv + (size_t)ci * 64);
        LF_PREFETCH(wv + (size_t)(ci + WPF) * 64);
#pragma clang loop unroll(full)
        for (int p = 0; p < P; p++) {
            uint32_t v = a[(size_t)p * cs + ci];
            acc[p] = Q6_Vqf16_vadd_Vqf16Vqf16(acc[p], Q6_Vqf16_vmpy_VhfRhf(w, DUP(v)));
        }
    }
#pragma clang loop unroll(full)
    for (int p = 0; p < P; p++) mo[p] = acc[p];
}

/* tile row ty of output block ob: out = LeakyReLU(conv(in) + bias + res) */
static void wino_item(const lay *L, const tens *in, tens *out, const tens *res, tscr *ts, int ty, int ob) {
    const int cs = in->cs, cin = L->cin, ccn = (cin + 63) >> 6, prow = in->w + 2, wo = out->w, ho = out->h;
    const int ntx = (wo + 1) >> 1;
    const HVX_Vector bias = lf_vload(L->bias + ob * 64);
    const uint16_t *uw = L->uw + (size_t)ob * 16 * cin * 64;
    const int y2 = 2 * ty;
    for (int tx0 = 0; tx0 < ntx; tx0 += 16) {
        const int T = ntx - tx0 < 16 ? ntx - tx0 : 16;
        for (int t = 0; t < T; t++)
            for (int cc = 0; cc < ccn; cc++) {
                const uint16_t *d = in->p + ((size_t)y2 * prow + 2 * (tx0 + t)) * cs + cc * 64;
                HVX_Vector tq[4][4];
                for (int c = 0; c < 4; c++) {
                    HVX_Vector d0 = lf_vload(d + (size_t)c * cs), d1 = lf_vload(d + ((size_t)prow + c) * cs);
                    HVX_Vector d2 = lf_vload(d + ((size_t)2 * prow + c) * cs), d3 = lf_vload(d + ((size_t)3 * prow + c) * cs);
                    tq[0][c] = Q6_Vqf16_vsub_VhfVhf(d0, d2);
                    tq[1][c] = Q6_Vqf16_vadd_VhfVhf(d1, d2);
                    tq[2][c] = Q6_Vqf16_vsub_VhfVhf(d2, d1);
                    tq[3][c] = Q6_Vqf16_vsub_VhfVhf(d1, d3);
                }
                for (int r = 0; r < 4; r++) {
                    uint16_t *v = ts->wv + ((size_t)(r * 4) * 16 + t) * cs + cc * 64;
                    lf_vstore(v, Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_Vqf16Vqf16(tq[r][0], tq[r][2])));
                    lf_vstore(v + (size_t)16 * cs, Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vqf16(tq[r][1], tq[r][2])));
                    lf_vstore(v + (size_t)32 * cs, Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_Vqf16Vqf16(tq[r][2], tq[r][1])));
                    lf_vstore(v + (size_t)48 * cs, Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_Vqf16Vqf16(tq[r][1], tq[r][3])));
                }
            }
        LF_MEMBAR();
        for (int pos = 0; pos < 16; pos++) {
            const uint16_t *a = ts->wv + (size_t)pos * 16 * cs;
            const uint16_t *wp = uw + (size_t)pos * cin * 64;
            HVX_Vector *mo = ts->wm + pos * 16;
            int p0 = 0;
            while (p0 < T) {
                const int rem = T - p0;
                if (rem >= 16) wmac(16, a, cs, cin, wp, mo);
                else if (rem >= 8) { wmac(8, a + (size_t)p0 * cs, cs, cin, wp, mo + p0); p0 += 8; continue; }
                else if (rem >= 4) { wmac(4, a + (size_t)p0 * cs, cs, cin, wp, mo + p0); p0 += 4; continue; }
                else if (rem >= 2) { wmac(2, a + (size_t)p0 * cs, cs, cin, wp, mo + p0); p0 += 2; continue; }
                else { wmac(1, a + (size_t)p0 * cs, cs, cin, wp, mo + p0); p0 += 1; continue; }
                p0 += 16;
            }
        }
        for (int t = 0; t < T; t++) {
            HVX_Vector pr[2][4];
            for (int c = 0; c < 4; c++) {
                HVX_Vector m0 = ts->wm[(0 + c) * 16 + t], m1 = ts->wm[(4 + c) * 16 + t];
                HVX_Vector m2 = ts->wm[(8 + c) * 16 + t], m3 = ts->wm[(12 + c) * 16 + t];
                pr[0][c] = Q6_Vqf16_vadd_Vqf16Vqf16(Q6_Vqf16_vadd_Vqf16Vqf16(m0, m1), m2);
                pr[1][c] = Q6_Vqf16_vsub_Vqf16Vqf16(Q6_Vqf16_vsub_Vqf16Vqf16(m1, m2), m3);
            }
            const int x2 = 2 * (tx0 + t);
#pragma clang loop unroll(full)
            for (int a = 0; a < 2; a++) {
                HVX_Vector y[2];
                y[0] = Q6_Vqf16_vadd_Vqf16Vqf16(Q6_Vqf16_vadd_Vqf16Vqf16(pr[a][0], pr[a][1]), pr[a][2]);
                y[1] = Q6_Vqf16_vsub_Vqf16Vqf16(Q6_Vqf16_vsub_Vqf16Vqf16(pr[a][1], pr[a][2]), pr[a][3]);
                if (y2 + a >= ho) continue;
#pragma clang loop unroll(full)
                for (int b = 0; b < 2; b++) {
                    if (x2 + b >= wo) continue;
                    const size_t off = ((size_t)(y2 + a + 1) * (wo + 2) + (x2 + b + 1)) * cs + ob * 64;
                    HVX_Vector v = Q6_Vqf16_vadd_Vqf16Vhf(y[b], bias);
                    v = Q6_Vqf16_vadd_Vqf16Vhf(v, lf_vload(res->p + off));
                    HVX_Vector x = Q6_Vhf_equals_Vqf16(v);
                    HVX_Vector neg = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfRhf(x, SLOPE_RT));
                    lf_vstore(out->p + off, Q6_Vhf_vmax_VhfVhf(x, neg));
                }
            }
        }
    }
}

static __attribute__((unused)) void run_wino(lfnet *n, lfpool *p, int tid, const lay *L, const tens *in, tens *out) {
    const int rows = (out->h + 1) >> 1;
    const uint32_t items = (uint32_t)rows * (uint32_t)L->nob;
    for (;;) {
        uint32_t i = lfpool_next(p);
        if (i >= items) break;
        int ob = 0;
        while (i >= (uint32_t)rows) { i -= (uint32_t)rows; ob++; __asm__ volatile("" : "+r"(i)); }
        wino_item(L, in, out, in, &n->ts[tid], (int)i, ob);
    }
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
        int ob = 0;
        while (i >= (uint32_t)rows) { i -= (uint32_t)rows; ob++; __asm__ volatile("" : "+r"(i)); }
        conv_row(j, (int)i, ob);
    }
}

/* block 0 (scale 8, no warp): mean of the central 2x2 of each 8x8 cell, scalar (tiny) */
static void build_input0(lfnet *n, int y) {
    blk *k = &n->B[0];
    const int s = k->s, ws = k->ws, cs = k->X.cs, o0 = s / 2 - 1;
    const float sc = 0.25f * (1.0f / 255.0f);
    uint16_t *row = k->X.p + ((size_t)(y + 1) * (ws + 2) + 1) * cs;
    for (int x = 0; x < ws; x++) {
        float acc[6] = {0, 0, 0, 0, 0, 0};
        for (int dy = 0; dy < 2; dy++)
            for (int dx = 0; dx < 2; dx++) {
                const int py = y * s + o0 + dy, px = x * s + o0 + dx;
                const uint8_t *a = n->img0 + (size_t)py * n->stride + (size_t)px * 4;
                const uint8_t *c1 = n->img1 + (size_t)py * n->stride + (size_t)px * 4;
                acc[0] += (float)a[0]; acc[1] += (float)a[1]; acc[2] += (float)a[2];
                acc[3] += (float)c1[0]; acc[4] += (float)c1[1]; acc[5] += (float)c1[2];
            }
        uint16_t *xp = row + (size_t)x * cs;
        for (int c = 0; c < 6; c++) xp[c] = lf_f2h(acc[c] * sc);
        xp[6] = n->t_hf;
    }
}

#define LF_MEMBAR() __asm__ volatile("" ::: "memory") /* vector stores and scalar loads of the same scratch */

/* ---- vector helpers (qf32 arithmetic, sf bits for shuffles/memory) ---------------- */
static inline uint32_t fbits(float f) { union { float f; uint32_t u; } v = {f}; return v.u; }
static inline HVX_Vector vsplatf(float f) { return Q6_V_vsplat_R(fbits(f)); }
static inline HVX_Vector vq(HVX_Vector sf) { return Q6_Vqf32_vadd_VsfVsf(sf, Q6_V_vzero()); } /* sf -> qf32 */
#define VSF(q) Q6_Vsf_equals_Vqf32(q)

/* bilinear taps of one image for 32 pixels: clamped source position -> byte offset of the (x0,y0) texel, fractions.
 * x0 is clamped to W-2 (fraction up to 1) so that x0+1 is always inside, which matches the border clamp. */
static inline void warp_taps(HVX_Vector xf, HVX_Vector yf, HVX_Vector fx, HVX_Vector fy, int W, int H, uint32_t stride,
                             uint32_t *offs, HVX_Vector *ax, HVX_Vector *ay) {
    const HVX_Vector zero = Q6_V_vzero();
    HVX_Vector sx = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(VSF(Q6_Vqf32_vadd_VsfVsf(xf, fx)), zero), vsplatf((float)(W - 1)));
    HVX_Vector sy = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(VSF(Q6_Vqf32_vadd_VsfVsf(yf, fy)), zero), vsplatf((float)(H - 1)));
    HVX_Vector x0 = Q6_Vw_vmin_VwVw(Q6_Vw_equals_Vsf(sx), Q6_V_vsplat_R((uint32_t)(W - 2)));
    HVX_Vector y0 = Q6_Vw_vmin_VwVw(Q6_Vw_equals_Vsf(sy), Q6_V_vsplat_R((uint32_t)(H - 2)));
    HVX_Vector x0f = Q6_Vsf_equals_Vw(x0), y0f = Q6_Vsf_equals_Vw(y0);
    *ax = Q6_Vqf32_vsub_VsfVsf(sx, x0f);
    *ay = Q6_Vqf32_vsub_VsfVsf(sy, y0f);
    /* offsets are exact integers < 2^24 in fp32 */
    HVX_Vector off = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfRsf(y0f, fbits((float)stride)), Q6_Vqf32_vmpy_VsfRsf(x0f, fbits(4.0f)));
    lf_vstore(offs, Q6_Vw_equals_Vsf(VSF(off)));
    LF_MEMBAR();
}

#define CHAN(v, c) Q6_Vsf_equals_Vw((c) ? Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR((v), 8 * (c)), m255) : Q6_V_vand_VV((v), m255))

static inline HVX_Vector lerp4(HVX_Vector v00, HVX_Vector v01, HVX_Vector v10, HVX_Vector v11, HVX_Vector ax, HVX_Vector ay) {
    HVX_Vector top = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_Vqf32_vsub_VsfVsf(v01, v00), ax), v00);
    HVX_Vector bot = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_Vqf32_vsub_VsfVsf(v11, v10), ax), v10);
    return Q6_Vqf32_vadd_Vqf32Vqf32(top, Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(bot, top), ay));
}

/* Blocks 1-3, output row y: for every 32-pixel chunk of the (full-resolution) source rows y*s+o0+{0,nn-1}, warp both images by the
 * accumulated flow (scalar RGBA gather, vector bilinear), add the mask and flow, sum over the rows -> S[11][W]. */
static void build_sums(lfnet *n, tscr *ts, int b, int y) {
    blk *k = &n->B[b];
    const int s = k->s, W = n->W, H = n->H, nn = s > 1 ? 2 : 1, o0 = s > 1 ? s / 2 - 1 : 0, sw = n->sw;
    const size_t pl = (size_t)W * H;
    const uint32_t stride = (uint32_t)n->stride;
    const HVX_Vector m255 = Q6_V_vsplat_R(255), zero = Q6_V_vzero(), iota = lf_vload(n->iota);
    for (int c = 0; c < W / 32; c++) {
        HVX_Vector acc[11];
        const HVX_Vector xf = Q6_Vsf_equals_Vw(Q6_Vw_vadd_VwVw(iota, Q6_V_vsplat_R((uint32_t)(32 * c))));
        for (int dy = 0; dy < nn; dy++) {
            const int py = y * s + o0 + dy;
            const float *fp = n->fm + (size_t)py * W + 32 * c;
            const HVX_Vector f[5] = {lf_vload(fp), lf_vload(fp + pl), lf_vload(fp + 2 * pl), lf_vload(fp + 3 * pl), lf_vload(fp + 4 * pl)};
            const HVX_Vector yf = vsplatf((float)py);
            HVX_Vector res[6];
            for (int im = 0; im < 2; im++) {
                HVX_Vector ax, ay;
                warp_taps(xf, yf, f[im * 2], f[im * 2 + 1], W, H, stride, ts->offs, &ax, &ay);
                const uint8_t *img = im ? n->img1 : n->img0;
                uint32_t *g = ts->gat;
                for (int l = 0; l < 32; l++) {
                    const uint32_t *q = (const uint32_t *)(img + ts->offs[l]);
                    const uint32_t *r = (const uint32_t *)((const uint8_t *)q + stride);
                    g[l] = q[0]; g[32 + l] = q[1]; g[64 + l] = r[0]; g[96 + l] = r[1];
                }
                LF_MEMBAR();
                const HVX_Vector g00 = lf_vload(g), g01 = lf_vload(g + 32), g10 = lf_vload(g + 64), g11 = lf_vload(g + 96);
                res[im * 3 + 0] = lerp4(CHAN(g00, 0), CHAN(g01, 0), CHAN(g10, 0), CHAN(g11, 0), ax, ay);
                res[im * 3 + 1] = lerp4(CHAN(g00, 1), CHAN(g01, 1), CHAN(g10, 1), CHAN(g11, 1), ax, ay);
                res[im * 3 + 2] = lerp4(CHAN(g00, 2), CHAN(g01, 2), CHAN(g10, 2), CHAN(g11, 2), ax, ay);
            }
            if (dy == 0) {
                for (int q = 0; q < 6; q++) acc[q] = res[q];
                acc[6] = Q6_Vqf32_vadd_VsfVsf(f[4], zero);
                for (int q = 0; q < 4; q++) acc[7 + q] = Q6_Vqf32_vadd_VsfVsf(f[q], zero);
            } else {
                for (int q = 0; q < 6; q++) acc[q] = Q6_Vqf32_vadd_Vqf32Vqf32(acc[q], res[q]);
                acc[6] = Q6_Vqf32_vadd_Vqf32Vsf(acc[6], f[4]);
                for (int q = 0; q < 4; q++) acc[7 + q] = Q6_Vqf32_vadd_Vqf32Vsf(acc[7 + q], f[q]);
            }
        }
        for (int q = 0; q < 11; q++) lf_vstore(ts->S + (size_t)q * sw + 32 * c, VSF(acc[q]));
    }
}

/* sums of the central nn x nn window of every s-pixel cell, for 32 output pixels starting at source x0 (s*32 source pixels) */
static inline HVX_Vector window_sum(const float *S, int s) {
    if (s == 1) return Q6_Vqf32_vadd_VsfVsf(lf_vload(S), Q6_V_vzero());
    if (s == 2) { /* lanes 2k, 2k+1 */
        HVX_VectorPair d = Q6_W_vdeal_VVR(lf_vload(S + 32), lf_vload(S), -4);
        return Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(d), Q6_V_hi_W(d));
    }
    /* s == 4: lanes 4k+1, 4k+2 */
    HVX_VectorPair ab = Q6_W_vdeal_VVR(lf_vload(S + 32), lf_vload(S), -4), cd = Q6_W_vdeal_VVR(lf_vload(S + 96), lf_vload(S + 64), -4);
    HVX_Vector p1 = Q6_V_lo_W(Q6_W_vdeal_VVR(Q6_V_hi_W(cd), Q6_V_hi_W(ab), -4));
    HVX_Vector p2 = Q6_V_hi_W(Q6_W_vdeal_VVR(Q6_V_lo_W(cd), Q6_V_lo_W(ab), -4));
    return Q6_Vqf32_vadd_VsfVsf(p1, p2);
}

static void build_input(lfnet *n, int tid, int b, int y) {
    if (b == 0) { build_input0(n, y); return; }
    blk *k = &n->B[b];
    tscr *ts = &n->ts[tid];
    const int s = k->s, ws = k->ws, cs = k->X.cs, sw = n->sw, nn = s > 1 ? 2 : 1;
    const float cnt = nn == 2 ? 0.25f : 1.0f;
    const uint32_t sc_img = fbits(cnt * (1.0f / 255.0f)), sc_m = fbits(cnt), sc_f = fbits(cnt * (1.0f / (float)s));
    build_sums(n, ts, b, y);
    uint16_t *row = k->X.p + ((size_t)(y + 1) * (ws + 2) + 1) * cs;
    for (int g = 0; g < (ws + 63) / 64; g++) {
        for (int q = 0; q < 11; q++) {
            const float *S = ts->S + (size_t)q * sw + (size_t)(2 * g) * 32 * s;
            const uint32_t sc = q < 6 ? sc_img : q == 6 ? sc_m : sc_f;
            HVX_Vector d0 = Q6_Vqf32_vmpy_VsfRsf(VSF(window_sum(S, s)), sc);
            HVX_Vector d1 = Q6_Vqf32_vmpy_VsfRsf(VSF(window_sum(S + 32 * s, s)), sc);
            lf_vstore(ts->hfp + q * 64, Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(d1, d0))); /* h[2i] = px i, h[2i+1] = px 32+i */
        }
        LF_MEMBAR();
        const int cnt_px = ws - 64 * g < 64 ? ws - 64 * g : 64;
        const uint16_t *hp = ts->hfp;
        for (int q = 0; q < cnt_px; q++) {
            const int i = q < 32 ? 2 * q : 2 * (q - 32) + 1;
            uint16_t *xp = row + (size_t)(64 * g + q) * cs;
            xp[0] = hp[i]; xp[1] = hp[64 + i]; xp[2] = hp[128 + i]; xp[3] = hp[192 + i]; xp[4] = hp[256 + i]; xp[5] = hp[320 + i];
            xp[6] = n->t_hf;
            xp[7] = hp[384 + i];
            xp[8] = hp[448 + i]; xp[9] = hp[512 + i]; xp[10] = hp[576 + i]; xp[11] = hp[640 + i];
        }
    }
}

/* ---- scatter + upsample -------------------------------------------------- */
/* D[pu] pixel (y,x) holds 10 groups of 4 halfwords: group ch*2+i = X 4x..4x+3 of row 4y+2pu+i of channel ch */
static void scatter_rows(lfnet *n, int b, int item) {
    blk *k = &n->B[b];
    const int h = k->D[0].h, pu = item >= h, y = item - pu * h, hs = k->hs, wq = k->D[0].w, tmw = k->tmw;
    const uint16_t *row = k->D[pu].p + ((size_t)(y + 1) * (wq + 2) + 1) * 64;
    for (int x = 0; x < wq; x++) {
        const uint64_t *v = (const uint64_t *)(row + (size_t)x * 64);
        for (int g = 0; g < 10; g++) {
            const int ch = g >> 1, i = g & 1;
            *(uint64_t *)(k->tm + ((size_t)ch * hs + 4 * y + 2 * pu + i) * tmw + 4 * x) = v[g];
        }
    }
}

/* out[0..15] = each word of v repeated s times (s = 1 << LS), in order */
static inline __attribute__((always_inline)) void replicate(HVX_Vector *o, HVX_Vector v, const int LS) {
    o[0] = v;
    int cnt = 1;
    for (int st = 0; st < LS; st++) {
        for (int i = cnt - 1; i >= 0; i--) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(o[i], o[i], -4);
            o[2 * i] = Q6_V_lo_W(p);
            o[2 * i + 1] = Q6_V_hi_W(p);
        }
        cnt *= 2;
    }
}

/* tm row of channel ch (hf, tmw wide) vertically blended into vb[0..ws) as fp32, edges replicated */
static inline void vblend(const uint16_t *r0, const uint16_t *r1, float ly, int ws, float *vb) {
    const HVX_Vector one = Q6_Vh_vsplat_R(0x3c00);
    const uint32_t lyb = fbits(ly);
    for (int c = 0; c < (ws + 63) / 64; c++) {
        HVX_VectorPair pa = Q6_Wqf32_vmpy_VhfVhf(lf_vload(r0 + c * 64), one), pb = Q6_Wqf32_vmpy_VhfVhf(lf_vload(r1 + c * 64), one);
        HVX_Vector la = Q6_V_lo_W(pa), ha = Q6_V_hi_W(pa);
        HVX_Vector lo = VSF(Q6_Vqf32_vadd_Vqf32Vqf32(la, Q6_Vqf32_vmpy_VsfRsf(VSF(Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_lo_W(pb), la)), lyb)));
        HVX_Vector hi = VSF(Q6_Vqf32_vadd_Vqf32Vqf32(ha, Q6_Vqf32_vmpy_VsfRsf(VSF(Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_hi_W(pb), ha)), lyb)));
        HVX_VectorPair sq = Q6_W_vshuff_VVR(hi, lo, -4); /* lo = even source px, hi = odd */
        lf_vstore(vb + c * 64, Q6_V_lo_W(sq));
        lf_vstore(vb + c * 64 + 32, Q6_V_hi_W(sq));
    }
    LF_MEMBAR();
    vb[-1] = vb[0];
    vb[ws] = vb[ws - 1];
    LF_MEMBAR();
}

static inline __attribute__((always_inline)) void hexpand(const float *vb, const HVX_Vector *wt, float *dst, int nov, int acc, const int LS) {
    const int s = 1 << LS;
    const HVX_Vector wm = wt[0], w0 = wt[1], wp = wt[2];
    for (int m = 0; m * s < nov; m++) {
        HVX_Vector rm[8], r0[8], rp[8];
        replicate(rm, lf_vloadu(vb + 32 * m - 1), LS);
        replicate(r0, lf_vload(vb + 32 * m), LS);
        replicate(rp, lf_vloadu(vb + 32 * m + 1), LS);
        for (int j = 0; j < s && m * s + j < nov; j++) {
            HVX_Vector o = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(rm[j], wm), Q6_Vqf32_vmpy_VsfVsf(r0[j], w0)),
                                                   Q6_Vqf32_vmpy_VsfVsf(rp[j], wp));
            float *d = dst + 32 * (m * s + j);
            if (acc) o = Q6_Vqf32_vadd_Vqf32Vsf(o, lf_vload(d));
            lf_vstore(d, VSF(o));
        }
    }
}

/* fm[0..4] (+)= bilinear upsample of the block's 5 planes by s (flow planes times s) for output row y */
static void upsample_row(lfnet *n, int tid, int b, int y) {
    blk *k = &n->B[b];
    const int W = n->W, hs = k->hs, ws = k->ws, s = k->s, tmw = k->tmw, nov = W / 32, acc = b > 0;
    const size_t pl = (size_t)W * n->H;
    float *vb = n->ts[tid].vb + 32;
    const float ly = k->uly[y];
    for (int ch = 0; ch < 5; ch++) {
        vblend(k->tm + ((size_t)ch * hs + k->uy0[y]) * tmw, k->tm + ((size_t)ch * hs + k->uy1[y]) * tmw, ly, ws, vb);
        const HVX_Vector wt[3] = {lf_vload(k->uw + (ch < 4 ? 0 : 3) * 32), lf_vload(k->uw + (ch < 4 ? 1 : 4) * 32), lf_vload(k->uw + (ch < 4 ? 2 : 5) * 32)};
        float *dst = n->fm + ch * pl + (size_t)y * W;
        if (s == 8) hexpand(vb, wt, dst, nov, acc, 3);
        else if (s == 4) hexpand(vb, wt, dst, nov, acc, 2);
        else hexpand(vb, wt, dst, nov, acc, 1);
    }
}

/* sigmoid of 32 sf lanes: exp via 2^i * poly(f), reciprocal by Newton from the integer seed (no libm, no division) */
static inline HVX_Vector vsigmoid(HVX_Vector x) {
    x = Q6_Vsf_vmax_VsfVsf(Q6_Vsf_vmin_VsfVsf(x, vsplatf(40.0f)), vsplatf(-40.0f));
    HVX_Vector t = Q6_Vqf32_vmpy_VsfRsf(x, fbits(-1.4426950408889634f)); /* -x log2 e */
    HVX_Vector ti = Q6_Vw_equals_Vsf(VSF(Q6_Vqf32_vadd_Vqf32Vsf(t, vsplatf(64.5f)))); /* round(t) + 64 */
    HVX_Vector r = Q6_Vw_vsub_VwVw(ti, Q6_V_vsplat_R(64));
    HVX_Vector f = Q6_Vqf32_vmpy_VsfRsf(VSF(Q6_Vqf32_vsub_Vqf32Vsf(t, Q6_Vsf_equals_Vw(r))), fbits(0.6931471805599453f));
    HVX_Vector p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfRsf(VSF(f), fbits(1.0f / 720.0f)), vsplatf(1.0f / 120.0f));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, f), vsplatf(1.0f / 24.0f));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, f), vsplatf(1.0f / 6.0f));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, f), vsplatf(0.5f));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, f), vsplatf(1.0f));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, f), vsplatf(1.0f)); /* exp(f) */
    HVX_Vector e2 = Q6_Vw_vasl_VwR(Q6_Vw_vadd_VwVw(r, Q6_V_vsplat_R(127)), 23);  /* 2^r */
    HVX_Vector d = VSF(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(VSF(p), e2), vsplatf(1.0f))); /* 1 + exp(-x) */
    HVX_Vector y = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7ef311c7u), d);
    const HVX_Vector two = vsplatf(2.0f);
    for (int it = 0; it < 4; it++) y = VSF(Q6_Vqf32_vmpy_VsfVsf(y, VSF(Q6_Vqf32_vsub_VsfVqf32(two, Q6_Vqf32_vmpy_VsfVsf(d, y)))));
    return y;
}

/* last block (scale 1) fused with the output: flow = fm + tm, mask = sigmoid(fm + tm) */
static void final_row(lfnet *n, int y) {
    blk *k = &n->B[NBLK - 1];
    const int W = n->W, hs = k->hs, tmw = k->tmw;
    const size_t pl = (size_t)W * n->H;
    const HVX_Vector one = Q6_Vh_vsplat_R(0x3c00);
    for (int ch = 0; ch < 5; ch++) {
        const uint16_t *tr = k->tm + ((size_t)ch * hs + y) * tmw;
        for (int c = 0; c * 64 < W; c++) {
            HVX_VectorPair pr = Q6_Wqf32_vmpy_VhfVhf(lf_vload(tr + c * 64), one);
            HVX_VectorPair sq = Q6_W_vshuff_VVR(VSF(Q6_V_hi_W(pr)), VSF(Q6_V_lo_W(pr)), -4);
            for (int h = 0; h < 2 && c * 64 + h * 32 < W; h++) {
                const size_t o = (size_t)y * W + c * 64 + h * 32;
                HVX_Vector v = VSF(Q6_Vqf32_vadd_VsfVsf(h ? Q6_V_hi_W(sq) : Q6_V_lo_W(sq), lf_vload(n->fm + ch * pl + o)));
                if (ch < 4) lf_vstoreu(n->flow_out + ch * pl + o, v);
                else lf_vstoreu(n->mask_out + o, vsigmoid(v));
            }
        }
    }
}

/* ---- run ---------------------------------------------------------------- */
enum { G_BUILD = 0, G_CONV01, G_RES, G_DECONV, G_SCATTER, G_UP, G_PER_BLOCK };

static void worker(lfpool *p, int tid, void *arg) {
    lfnet *n = (lfnet *)arg;
    for (int b = 0; b < NBLK; b++) {
        blk *k = &n->B[b];
        const lay *Ls = &n->L[b * LFW_LAYERS_PER_BLOCK];
        const int g0 = b * G_PER_BLOCK;
        for (;;) {
            uint32_t y = lfpool_next(p);
            if (y >= (uint32_t)k->hs) break;
            build_input(n, tid, b, (int)y);
        }
        stage_end(n, p, tid, g0 + G_BUILD);
        cjob j;
        cjob_init_conv3(&j, &Ls[0], &k->X, &k->A0, 0, EPI_LEAKY);
        run_conv(n, p, tid, &j);
        lfpool_barrier(p);
        cjob_init_conv3(&j, &Ls[1], &k->A0, &k->A1, 0, EPI_LEAKY);
        run_conv(n, p, tid, &j);
        stage_end(n, p, tid, g0 + G_CONV01);
        tens *cur = &k->A1, *oth = &k->A2;
        for (int r = 0; r < 8; r++) {
#if LF_WINO
            run_wino(n, p, tid, &Ls[2 + r], cur, oth);
#else
            cjob_init_conv3(&j, &Ls[2 + r], cur, oth, cur, EPI_RES);
            run_conv(n, p, tid, &j);
#endif
            lfpool_barrier(p);
            tens *t = cur; cur = oth; oth = t;
        }
        stage_end(n, p, tid, g0 + G_RES);
        cjob_init_deconv(&j, &Ls[10], cur, &k->D[0], 0);
        run_conv(n, p, tid, &j);
        lfpool_barrier(p);
        cjob_init_deconv(&j, &Ls[10], cur, &k->D[1], 1);
        run_conv(n, p, tid, &j);
        stage_end(n, p, tid, g0 + G_DECONV);
        for (;;) {
            uint32_t i = lfpool_next(p);
            if (i >= (uint32_t)(2 * k->D[0].h)) break;
            scatter_rows(n, b, (int)i);
        }
        stage_end(n, p, tid, g0 + G_SCATTER);
        for (;;) {
            uint32_t y = lfpool_next(p);
            if (y >= (uint32_t)n->H) break;
            if (b == NBLK - 1) final_row(n, (int)y);
            else upsample_row(n, tid, b, (int)y);
        }
        stage_end(n, p, tid, g0 + G_UP);
    }
}

int lfnet_run(lfnet *n, const uint8_t *rgba0, const uint8_t *rgba1, int stride, float t, float *flow, float *mask,
              lfrife_run_stats *stats) {
    /* texel offsets are computed in fp32 (exact below 2^24); images need 4-byte alignment */
    if (stride < n->W * 4 || (stride & 3) || ((uintptr_t)rgba0 & 3) || ((uintptr_t)rgba1 & 3) || (size_t)stride * n->H >= (1u << 24)) return -5;
    n->img0 = rgba0;
    n->img1 = rgba1;
    n->stride = stride;
    n->t = t;
    n->t_hf = lf_f2h(t);
    n->flow_out = flow;
    n->mask_out = mask;
    n->profile = stats != 0;
    for (int i = 0; i < LFRIFE_NGROUPS; i++) n->gt[i] = 0;
    n->last = lf_now();
    const uint64_t t0 = n->last, c0 = lf_cycles();
    int rc = lfpool_run(&n->pool, n->nt, worker, n);
    if (stats) {
        stats->rc = rc;
        stats->nthreads = (uint32_t)lfpool_nthreads(&n->pool);
        stats->total_ticks = lf_now() - t0;
        stats->pcycles = lf_cycles() - c0;
        for (int i = 0; i < LFRIFE_NGROUPS; i++) stats->group_ticks[i] = n->gt[i];
    }
    return rc;
}
