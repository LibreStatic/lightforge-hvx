/* SPDX-License-Identifier: MIT
 * Tiny helpers shared by the host test and the on-device bench: file loading,
 * golden comparison. Header-only, C11. */
#ifndef LF_GOLDEN_UTIL_H
#define LF_GOLDEN_UTIL_H
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline void *lf_read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *p = malloc((size_t)n + 128);
    if (fread(p, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(p); return NULL; }
    fclose(f);
    if (size) *size = (size_t)n;
    return p;
}

/* t is stored in <dir>/manifest.txt lines "name t" (written by export_weights.py) */
static inline float lf_pair_t(const char *dir, const char *name) {
    char p[512], nm[128];
    snprintf(p, sizeof p, "%s/manifest.txt", dir);
    FILE *f = fopen(p, "r");
    float t = 0.5f, v;
    if (!f) return t;
    while (fscanf(f, "%127s %f", nm, &v) == 2)
        if (!strcmp(nm, name)) t = v;
    fclose(f);
    return t;
}

/* deterministic textured pair: img1 is img0 moved by (dx, dy) pixels (checker + hash texture, 3 colour channels) */
static inline void lf_synth_pair(uint8_t *i0, uint8_t *i1, int w, int h, int dx, int dy) {
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < 3; c++) {
                int xs = x - dx, ys = y - dy;
                i0[((size_t)y * w + x) * 4 + c] = (uint8_t)(128 + 90 * ((((x >> 3) + (y >> 4) + c) & 1) ? 1 : -1) / 2 + ((x * 7 + y * 3 + c * 50) & 31));
                i1[((size_t)y * w + x) * 4 + c] = (uint8_t)(128 + 90 * ((((xs >> 3) + (ys >> 4) + c) & 1) ? 1 : -1) / 2 + ((xs * 7 + ys * 3 + c * 50) & 31));
            }
            i0[((size_t)y * w + x) * 4 + 3] = i1[((size_t)y * w + x) * 4 + 3] = 255;
        }
}

/* dst = src moved by (dx, dy) pixels, edges clamped */
static inline void lf_shift_rgba(uint8_t *dst, const uint8_t *src, int w, int h, int dx, int dy) {
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            int xs = x - dx, ys = y - dy;
            xs = xs < 0 ? 0 : xs >= w ? w - 1 : xs;
            ys = ys < 0 ? 0 : ys >= h ? h - 1 : ys;
            memcpy(dst + ((size_t)y * w + x) * 4, src + ((size_t)ys * w + xs) * 4, 4);
        }
}

typedef struct { double fmax, fmean, fbad, mmax, mmean; } lf_cmp;

/* flow: planar 4*n, mask: n; fbad = fraction of pixels where any flow channel differs by > 0.5 px */
static inline lf_cmp lf_compare(const float *flow, const float *gflow, const float *mask, const float *gmask, int n) {
    lf_cmp c = {0, 0, 0, 0, 0};
    long bad = 0;
    for (int i = 0; i < n; i++) {
        int isbad = 0;
        for (int k = 0; k < 4; k++) {
            double e = fabs((double)flow[(size_t)k * n + i] - (double)gflow[(size_t)k * n + i]);
            if (e > c.fmax) c.fmax = e;
            c.fmean += e;
            if (e > 0.5) isbad = 1;
        }
        bad += isbad;
        double m = fabs((double)mask[i] - (double)gmask[i]);
        if (m > c.mmax) c.mmax = m;
        c.mmean += m;
    }
    c.fmean /= 4.0 * n;
    c.mmean /= n;
    c.fbad = (double)bad / n;
    return c;
}
#endif
