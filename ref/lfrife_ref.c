/* SPDX-License-Identifier: BSD-3-Clause */
#include "lfrife_ref.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../inc/lfrife_weights.h"

typedef struct {
    int type, cin, cout, stride;
    float *w; /* conv: [ky][kx][ci][co]; deconv: [ky][kx][ci][20] */
    float *b; /* [cout] or [20] */
} layer;

struct lfref {
    int w, h;
    layer layers[LFW_NLAYERS];
    int cin[LFW_NBLOCKS], c[LFW_NBLOCKS], scale[LFW_NBLOCKS];
    float *fm; /* h*w*5: flow(4) + mask */
    float *img0, *img1; /* h*w*3 */
    float *dbg[LFW_NBLOCKS];
    int keep;
};

static float *xmalloc(size_t n) {
    float *p = malloc(n * sizeof(float));
    return p;
}

lfref *lfref_create(const void *blob, size_t size, int w, int h) {
    const uint8_t *base = blob;
    const lfw_header *hd = blob;
    if (size < sizeof(*hd) || memcmp(hd->magic, LFW_MAGIC, 8) || hd->kind != LFW_KIND_F32 ||
        hd->nlayers != LFW_NLAYERS || hd->total_bytes != size)
        return NULL;
    if (w % 32 || h % 32 || w <= 0 || h <= 0) return NULL;
    lfref *r = calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->w = w;
    r->h = h;
    for (int b = 0; b < LFW_NBLOCKS; b++) {
        r->cin[b] = (int)hd->block_cin[b];
        r->c[b] = (int)hd->block_c[b];
        r->scale[b] = (int)hd->block_scale[b];
    }
    const lfw_layer *tab = (const lfw_layer *)(base + sizeof(*hd));
    for (int i = 0; i < LFW_NLAYERS; i++) {
        layer *L = &r->layers[i];
        L->type = (int)tab[i].type;
        L->cin = (int)tab[i].cin;
        L->cout = (int)tab[i].cout;
        L->stride = (int)tab[i].stride;
        const float *w0 = (const float *)(base + tab[i].w_off);
        const float *b0 = (const float *)(base + tab[i].b_off);
        if (L->type == LFW_DECONV) {
            /* torch [ci][co=24][ky][kx] -> [ky][kx][ci][20]; co = ch*4+i*2+j, ch<5 */
            L->w = xmalloc((size_t)16 * L->cin * 20);
            L->b = xmalloc(20);
            for (int ci = 0; ci < L->cin; ci++)
                for (int co = 0; co < 20; co++)
                    for (int ky = 0; ky < 4; ky++)
                        for (int kx = 0; kx < 4; kx++)
                            L->w[((size_t)(ky * 4 + kx) * L->cin + ci) * 20 + co] =
                                w0[(((size_t)ci * 24 + co) * 4 + ky) * 4 + kx];
            for (int co = 0; co < 20; co++) L->b[co] = b0[co];
        } else {
            L->w = xmalloc((size_t)9 * L->cin * L->cout);
            L->b = xmalloc((size_t)L->cout);
            for (int co = 0; co < L->cout; co++)
                for (int ci = 0; ci < L->cin; ci++)
                    for (int ky = 0; ky < 3; ky++)
                        for (int kx = 0; kx < 3; kx++)
                            L->w[((size_t)(ky * 3 + kx) * L->cin + ci) * L->cout + co] =
                                w0[(((size_t)co * L->cin + ci) * 3 + ky) * 3 + kx];
            for (int co = 0; co < L->cout; co++) L->b[co] = b0[co];
        }
    }
    r->fm = xmalloc((size_t)w * h * 5);
    r->img0 = xmalloc((size_t)w * h * 3);
    r->img1 = xmalloc((size_t)w * h * 3);
    return r;
}

void lfref_destroy(lfref *r) {
    if (!r) return;
    for (int i = 0; i < LFW_NLAYERS; i++) {
        free(r->layers[i].w);
        free(r->layers[i].b);
    }
    for (int b = 0; b < LFW_NBLOCKS; b++) free(r->dbg[b]);
    free(r->fm);
    free(r->img0);
    free(r->img1);
    free(r);
}

void lfref_keep_debug(lfref *r, int enable) { r->keep = enable; }
const float *lfref_debug_block(const lfref *r, int block) { return r->dbg[block]; }

/* NHWC 3x3 conv, pad 1, stride 1 or 2, + bias, LeakyReLU(0.2) (optionally with residual added before it). */
static void conv3(const layer *L, const float *in, int ih, int iw, float *out, const float *res) {
    const int s = L->stride, oh = (ih + s - 1) / s, ow = (iw + s - 1) / s;
    const int cin = L->cin, cout = L->cout;
    float acc[256];
    for (int y = 0; y < oh; y++)
        for (int x = 0; x < ow; x++) {
            for (int co = 0; co < cout; co++) acc[co] = L->b[co];
            for (int ky = 0; ky < 3; ky++) {
                int yy = y * s + ky - 1;
                if (yy < 0 || yy >= ih) continue;
                for (int kx = 0; kx < 3; kx++) {
                    int xx = x * s + kx - 1;
                    if (xx < 0 || xx >= iw) continue;
                    const float *ip = in + ((size_t)yy * iw + xx) * cin;
                    const float *wp = L->w + (size_t)(ky * 3 + kx) * cin * cout;
                    for (int ci = 0; ci < cin; ci++) {
                        const float v = ip[ci];
                        const float *wr = wp + (size_t)ci * cout;
                        for (int co = 0; co < cout; co++) acc[co] += v * wr[co];
                    }
                }
            }
            float *op = out + ((size_t)y * ow + x) * cout;
            if (res) {
                const float *rp = res + ((size_t)y * ow + x) * cout;
                for (int co = 0; co < cout; co++) acc[co] += rp[co];
            }
            for (int co = 0; co < cout; co++) op[co] = acc[co] > 0 ? acc[co] : 0.2f * acc[co];
        }
}

/* ConvTranspose 4x4 s2 p1 (20 live outputs) + PixelShuffle(2): in [ih][iw][cin] -> tmp [2*ih*2][2*iw*2][5]. */
static void deconv_shuffle(const layer *L, const float *in, int ih, int iw, float *tmp) {
    const int cin = L->cin, oh = ih * 2, ow = iw * 2, th = oh * 2, tw = ow * 2;
    float *d = xmalloc((size_t)oh * ow * 20);
    for (int p = 0; p < oh * ow; p++)
        for (int co = 0; co < 20; co++) d[(size_t)p * 20 + co] = L->b[co];
    for (int y = 0; y < ih; y++)
        for (int x = 0; x < iw; x++) {
            const float *ip = in + ((size_t)y * iw + x) * cin;
            for (int ky = 0; ky < 4; ky++) {
                int u = 2 * y - 1 + ky;
                if (u < 0 || u >= oh) continue;
                for (int kx = 0; kx < 4; kx++) {
                    int v = 2 * x - 1 + kx;
                    if (v < 0 || v >= ow) continue;
                    float *dp = d + ((size_t)u * ow + v) * 20;
                    const float *wp = L->w + (size_t)(ky * 4 + kx) * cin * 20;
                    for (int ci = 0; ci < cin; ci++) {
                        const float a = ip[ci];
                        const float *wr = wp + (size_t)ci * 20;
                        for (int co = 0; co < 20; co++) dp[co] += a * wr[co];
                    }
                }
            }
        }
    for (int u = 0; u < oh; u++)
        for (int v = 0; v < ow; v++)
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    for (int ch = 0; ch < 5; ch++)
                        tmp[((size_t)(2 * u + i) * tw + (2 * v + j)) * 5 + ch] =
                            d[((size_t)u * ow + v) * 20 + ch * 4 + i * 2 + j];
    (void)th;
    free(d);
}

static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* grid_sample(border, align_corners=True) with a px-space flow, one pixel, 3 channels */
static inline void warp_px(const float *img, int w, int h, int x, int y, float fx, float fy, float *o) {
    float sx = clampf((float)x + fx, 0.f, (float)(w - 1)), sy = clampf((float)y + fy, 0.f, (float)(h - 1));
    int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
    int x1 = x0 + 1 < w ? x0 + 1 : w - 1, y1 = y0 + 1 < h ? y0 + 1 : h - 1;
    float ax = sx - (float)x0, ay = sy - (float)y0;
    const float *p00 = img + ((size_t)y0 * w + x0) * 3, *p01 = img + ((size_t)y0 * w + x1) * 3;
    const float *p10 = img + ((size_t)y1 * w + x0) * 3, *p11 = img + ((size_t)y1 * w + x1) * 3;
    for (int c = 0; c < 3; c++)
        o[c] = (p00[c] * (1 - ax) + p01[c] * ax) * (1 - ay) + (p10[c] * (1 - ax) + p11[c] * ax) * ay;
}

int lfref_run(lfref *r, const uint8_t *rgba0, const uint8_t *rgba1, int stride, float t, float *flow,
              float *mask) {
    const int W = r->w, H = r->h;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int c = 0; c < 3; c++) {
                r->img0[((size_t)y * W + x) * 3 + c] = (float)rgba0[(size_t)y * stride + x * 4 + c] / 255.0f;
                r->img1[((size_t)y * W + x) * 3 + c] = (float)rgba1[(size_t)y * stride + x * 4 + c] / 255.0f;
            }
    for (int b = 0; b < LFW_NBLOCKS; b++) {
        const int s = r->scale[b], hs = H / s, ws = W / s, cin = r->cin[b], c = r->c[b];
        float *xin = xmalloc((size_t)hs * ws * cin);
        /* bilinear 1/s (half-pixel) == mean of the central 2x2 of each sxs block (s>1) */
        const int o0 = s > 1 ? s / 2 - 1 : 0, n = s > 1 ? 2 : 1;
        const float inv = 1.0f / (float)(n * n);
        for (int y = 0; y < hs; y++)
            for (int x = 0; x < ws; x++) {
                float *xp = xin + ((size_t)y * ws + x) * cin;
                float acc[12] = {0};
                for (int dy = 0; dy < n; dy++)
                    for (int dx = 0; dx < n; dx++) {
                        const int py = y * s + o0 + dy, px = x * s + o0 + dx;
                        float v[12];
                        if (b == 0) {
                            for (int k = 0; k < 3; k++) {
                                v[k] = r->img0[((size_t)py * W + px) * 3 + k];
                                v[3 + k] = r->img1[((size_t)py * W + px) * 3 + k];
                            }
                            v[6] = t;
                        } else {
                            const float *f = r->fm + ((size_t)py * W + px) * 5;
                            warp_px(r->img0, W, H, px, py, f[0], f[1], v);
                            warp_px(r->img1, W, H, px, py, f[2], f[3], v + 3);
                            v[6] = t;
                            v[7] = f[4];
                            v[8] = f[0] / (float)s;
                            v[9] = f[1] / (float)s;
                            v[10] = f[2] / (float)s;
                            v[11] = f[3] / (float)s;
                        }
                        for (int k = 0; k < cin; k++) acc[k] += v[k];
                    }
                for (int k = 0; k < cin; k++) xp[k] = acc[k] * inv;
            }
        float *a0 = xmalloc((size_t)(hs / 2) * (ws / 2) * (c / 2));
        float *a1 = xmalloc((size_t)(hs / 4) * (ws / 4) * c);
        float *a2 = xmalloc((size_t)(hs / 4) * (ws / 4) * c);
        const layer *L = &r->layers[b * LFW_LAYERS_PER_BLOCK];
        conv3(&L[0], xin, hs, ws, a0, NULL);
        conv3(&L[1], a0, hs / 2, ws / 2, a1, NULL);
        for (int k = 0; k < 8; k++) {
            conv3(&L[2 + k], a1, hs / 4, ws / 4, a2, a1);
            float *tmp = a1;
            a1 = a2;
            a2 = tmp;
        }
        float *tm = xmalloc((size_t)hs * ws * 5);
        deconv_shuffle(&L[10], a1, hs / 4, ws / 4, tm);
        /* bilinear upsample xs (align_corners=False), flow *= s, accumulate */
        for (int y = 0; y < H; y++) {
            float sy = clampf(((float)y + 0.5f) / (float)s - 0.5f, 0.f, (float)hs);
            int y0 = (int)sy, y1 = y0 + 1 < hs ? y0 + 1 : hs - 1;
            if (y0 > hs - 1) y0 = hs - 1;
            float ly = sy - (float)y0;
            for (int x = 0; x < W; x++) {
                float sx = clampf(((float)x + 0.5f) / (float)s - 0.5f, 0.f, (float)ws);
                int x0 = (int)sx, x1 = x0 + 1 < ws ? x0 + 1 : ws - 1;
                if (x0 > ws - 1) x0 = ws - 1;
                float lx = sx - (float)x0;
                float *fp = r->fm + ((size_t)y * W + x) * 5;
                for (int k = 0; k < 5; k++) {
                    float v = (tm[((size_t)y0 * ws + x0) * 5 + k] * (1 - lx) + tm[((size_t)y0 * ws + x1) * 5 + k] * lx) * (1 - ly) +
                              (tm[((size_t)y1 * ws + x0) * 5 + k] * (1 - lx) + tm[((size_t)y1 * ws + x1) * 5 + k] * lx) * ly;
                    if (k < 4) v *= (float)s;
                    fp[k] = (b == 0 ? 0.f : fp[k]) + v;
                }
            }
        }
        if (r->keep) {
            if (!r->dbg[b]) r->dbg[b] = xmalloc((size_t)W * H * 5);
            memcpy(r->dbg[b], r->fm, sizeof(float) * (size_t)W * H * 5);
        }
        free(xin); free(a0); free(a1); free(a2); free(tm);
    }
    for (int p = 0; p < W * H; p++) {
        const float *f = r->fm + (size_t)p * 5;
        for (int k = 0; k < 4; k++) flow[(size_t)k * W * H + p] = f[k];
        mask[p] = 1.0f / (1.0f + expf(-f[4]));
    }
    return 0;
}
