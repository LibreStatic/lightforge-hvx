/* SPDX-License-Identifier: BSD-3-Clause
 * Tiny helpers shared by the host test and the on-device bench: file loading,
 * golden comparison. Header-only, C11. */
#ifndef LF_GOLDEN_UTIL_H
#define LF_GOLDEN_UTIL_H
#include <math.h>
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
