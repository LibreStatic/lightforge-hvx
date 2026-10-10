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
    if (argc >= 8 && !strcmp(argv[2], "--synth")) { /* test_ref <blob> --synth <w> <h> <dx> <dy> <dsp flow/mask f32 file>: DSP output vs this reference */
        int sw = atoi(argv[3]), sh = atoi(argv[4]);
        lfref *sr = lfref_create(blob, bsz, sw, sh);
        uint8_t *a = malloc((size_t)sw * sh * 4), *b = malloc((size_t)sw * sh * 4);
        lf_synth_pair(a, b, sw, sh, atoi(argv[5]), atoi(argv[6]));
        float *rf = malloc(sizeof(float) * 5 * (size_t)sw * sh);
        lfref_run(sr, a, b, sw * 4, 0.5f, rf, rf + (size_t)4 * sw * sh);
        size_t dn;
        float *df = lf_read_file(argv[7], &dn);
        if (!df || dn != sizeof(float) * 5 * (size_t)sw * sh) { fprintf(stderr, "bad dsp file\n"); return 2; }
        lf_cmp c = lf_compare(df, rf, df + (size_t)4 * sw * sh, rf + (size_t)4 * sw * sh, sw * sh);
        double mf = 0;
        for (int i = 0; i < sw * sh; i++) mf += rf[i];
        printf("synth %dx%d shift (%s,%s): ref mean flow0 %.2f; DSP vs ref: flow max %.3e mean %.3e frac>0.5px %.4f mask max %.3e mean %.3e\n", sw, sh,
               argv[5], argv[6], mf / (sw * sh), c.fmax, c.fmean, c.fbad, c.mmax, c.mmean);
        return 0;
    }
    if (argc >= 8 && !strcmp(argv[2], "--shift")) { /* test_ref <blob> --shift <golden dir> <pair> <dx> <dy> <dsp f32 file> */
        const int sw = 320, sh = 192;
        lfref *sr = lfref_create(blob, bsz, sw, sh);
        char pp[512];
        size_t dn;
        snprintf(pp, sizeof pp, "%s/%s_img0.rgba", argv[3], argv[4]);
        uint8_t *a = lf_read_file(pp, &dn), *b = malloc((size_t)sw * sh * 4);
        lf_shift_rgba(b, a, sw, sh, atoi(argv[5]), atoi(argv[6]));
        float *rf = malloc(sizeof(float) * 5 * (size_t)sw * sh);
        lfref_run(sr, a, b, sw * 4, 0.5f, rf, rf + (size_t)4 * sw * sh);
        float *df = lf_read_file(argv[7], &dn);
        if (!df || dn != sizeof(float) * 5 * (size_t)sw * sh) { fprintf(stderr, "bad dsp file\n"); return 2; }
        lf_cmp c = lf_compare(df, rf, df + (size_t)4 * sw * sh, rf + (size_t)4 * sw * sh, sw * sh);
        double mf = 0;
        for (int i = 0; i < sw * sh; i++) mf += rf[i];
        printf("%s shifted (%s,%s): ref mean flow0 %.2f; DSP vs ref: flow max %.3e mean %.3e frac>0.5px %.4f mask max %.3e mean %.3e\n", argv[4], argv[5],
               argv[6], mf / (sw * sh), c.fmax, c.fmean, c.fbad, c.mmax, c.mmean);
        return 0;
    }
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
