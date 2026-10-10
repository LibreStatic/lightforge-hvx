/* SPDX-License-Identifier: MIT
 * Host/arm64 test + benchmark for the plain-C reference.
 * usage: test_ref <rife46_f32.bin> <golden dir> <pair prefix>... [--max-err px] [--bench N]
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "lfrife_ref.h"
#include "../tests/golden_util.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s blob golden_dir pair... [--max-err px] [--bench N] [--size WxH]\n", argv[0]);
        return 2;
    }
    size_t bsz;
    void *blob = lf_read_file(argv[1], &bsz);
    if (!blob) return 2;
    double max_err = 2e-3;
    int bench = 0, w = 320, h = 192;
    const char *pairs[16];
    int np = 0;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--max-err") && i + 1 < argc) max_err = atof(argv[++i]);
        else if (!strcmp(argv[i], "--bench") && i + 1 < argc) bench = atoi(argv[++i]);
        else pairs[np++] = argv[i];
    }
    lfref *r = lfref_create(blob, bsz, w, h);
    if (!r) { fprintf(stderr, "lfref_create failed\n"); return 2; }
    int fail = 0;
    float *flow = malloc(sizeof(float) * 4 * w * h), *mask = malloc(sizeof(float) * w * h);
    for (int k = 0; k < np; k++) {
        char p[512];
        size_t n;
        snprintf(p, sizeof p, "%s/%s_img0.rgba", argv[2], pairs[k]);
        uint8_t *i0 = lf_read_file(p, &n);
        snprintf(p, sizeof p, "%s/%s_img1.rgba", argv[2], pairs[k]);
        uint8_t *i1 = lf_read_file(p, &n);
        snprintf(p, sizeof p, "%s/%s_flow.f32", argv[2], pairs[k]);
        float *gf = lf_read_file(p, &n);
        snprintf(p, sizeof p, "%s/%s_mask.f32", argv[2], pairs[k]);
        float *gm = lf_read_file(p, &n);
        float t = lf_pair_t(argv[2], pairs[k]);
        if (!i0 || !i1 || !gf || !gm) return 2;
        double t0 = now_ms();
        lfref_run(r, i0, i1, w * 4, t, flow, mask);
        double dt = now_ms() - t0;
        lf_cmp c = lf_compare(flow, gf, mask, gm, w * h);
        printf("%s t=%.3f: flow max %.3e mean %.3e  frac>0.5px %.4f  mask max %.3e mean %.3e  (%.0f ms)\n",
               pairs[k], t, c.fmax, c.fmean, c.fbad, c.mmax, c.mmean, dt);
        if (c.fmax > max_err) fail = 1;
        for (int j = 1; j < bench; j++) {
            t0 = now_ms();
            lfref_run(r, i0, i1, w * 4, t, flow, mask);
            printf("  run %d: %.0f ms\n", j, now_ms() - t0);
        }
        free(i0); free(i1); free(gf); free(gm);
    }
    printf(fail ? "FAIL (flow max err > %.1e px)\n" : "PASS (flow max err <= %.1e px)\n", max_err);
    lfref_destroy(r);
    return fail;
}
