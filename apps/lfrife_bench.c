/*
 * SPDX-License-Identifier: BSD-3-Clause
 * On-device driver for the RIFE DSP library: probes, accuracy vs golden, timing.
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../inc/lfrife.h"
#include "../tests/golden_util.h"

static uint16_t f2h(float f) {
    union { float f; uint32_t u; } v = {f};
    uint32_t s = (v.u >> 16) & 0x8000, e = ((v.u >> 23) & 0xff), m = v.u & 0x7fffff;
    if (e == 0) return (uint16_t)s;
    int ne = (int)e - 127 + 15;
    if (ne <= 0) return (uint16_t)s;
    if (ne >= 31) return (uint16_t)(s | 0x7c00);
    return (uint16_t)(s | (ne << 10) | (m >> 13));
}
static float h2f(uint16_t h) {
    uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 31, m = h & 0x3ff;
    union { float f; uint32_t u; } v;
    if (e == 0) { v.u = s; if (m) { v.f = (float)m / 16777216.0f; if (s) v.f = -v.f; } return v.f; }
    v.u = s | ((e + 112) << 23) | (m << 13);
    return v.f;
}

static void probe1(lfrife *h) {
    uint16_t in[128];
    for (int i = 0; i < 64; i++) { in[i] = f2h((float)(i + 1)); in[64 + i] = f2h(0.5f * (float)(i % 7 + 1)); }
    uint8_t out[1280];
    memset(out, 0, sizeof out);
    uint32_t rt = ((uint32_t)f2h(3.0f) << 16) | f2h(2.0f); /* hi=3, lo=2 */
    int rc = lfrife_probe(h, 1, rt, 0, 0, in, sizeof in, out, sizeof out);
    printf("probe1 rc=%d (last 0x%x)\n", rc, lfrife_last_rc(h));
    if (rc) return;
    const float *f = (const float *)out;
    const uint16_t *hh = (const uint16_t *)out;
    for (int b = 0; b < 10; b++) { printf("blk%d:", b); for (int i = 0; i < 8; i++) printf(" %08x", ((uint32_t*)out)[b*32+i]); printf("\n"); }
    printf("A[i]=i+1, B[i]=0.5*(i%%7+1)\n");
    printf("pair.lo sf[0..5]: "); for (int i = 0; i < 6; i++) printf("%g ", f[i]); printf("\n");
    printf("pair.hi sf[0..5]: "); for (int i = 0; i < 6; i++) printf("%g ", f[32 + i]); printf("\n");
    printf("cvt(hi,lo) hf[0..7]: "); for (int i = 0; i < 8; i++) printf("%g ", h2f(hh[128 + i])); printf("\n");
    printf("cvt(lo,hi) hf[0..7]: "); for (int i = 0; i < 8; i++) printf("%g ", h2f(hh[192 + i])); printf("\n");
    printf("cvt_sf(A).lo[0..5]: "); for (int i = 0; i < 6; i++) printf("%g ", f[256 + i]); printf("\n");
    printf("cvt_sf(A).hi[0..5]: "); for (int i = 0; i < 6; i++) printf("%g ", f[288 + i]); printf("\n");
    printf("vmpy_rt_hf(A, Rt(lo=2,hi=3)) hf[0..7]: "); for (int i = 0; i < 8; i++) printf("%g ", h2f(hh[384 + i])); printf("\n");
    printf("vsplat(Rt) h[0..3] raw: %04x %04x %04x %04x\n", hh[448], hh[449], hh[450], hh[451]);
    printf("vmpy_hf_hf(A,B) hf[0..7]: "); for (int i = 0; i < 8; i++) printf("%g ", h2f(hh[512 + i])); printf("\n");
    printf("vmax_hf(A,B) hf[0..7]: "); for (int i = 0; i < 8; i++) printf("%g ", h2f(hh[576 + i])); printf("\n");
}

static void probe3(lfrife *h) {
    uint16_t in[128];
    for (int i = 0; i < 64; i++) { in[i] = f2h((float)(i + 1)); in[64 + i] = f2h(0.5f * (float)(i % 7 + 1)); }
    uint8_t out[1280];
    memset(out, 0, sizeof out);
    int rc = lfrife_probe(h, 3, 0, 0, 0, in, sizeof in, out, sizeof out);
    printf("probe3 rc=%d\n", rc);
    const char *nm[] = {"qf16 vmpy->hf", "IEEE vadd hf", "qf32 vmpy lo", "qf32 vmpy hi", "qf16 vadd->hf", "sf=vadd(hf,hf) lo", "sf=vadd(hf,hf) hi", "sf+=vmpy lo", "sf+=vmpy hi", "hf=vmpy(hf,hf)"};
    for (int b = 0; b < 10; b++) {
        printf("%-18s:", nm[b]);
        for (int i = 0; i < 6; i++) {
            if (b == 2 || b == 3 || b == 5 || b == 6 || b == 7 || b == 8) printf(" %g", ((float *)out)[b * 32 + i]);
            else printf(" %g", h2f(((uint16_t *)out)[b * 64 + i]));
        }
        printf("\n");
    }
}

static void probe10(lfrife *h) {
    const char *nm[] = {"", "qf16 rt (8 px)", "qf32 widen splat (4 px)", "qf16 vec splat (8 px)"};
    for (int k = 1; k <= 3; k++) {
        uint64_t o[2];
        uint32_t iters = 100000;
        lfrife_probe(h, 10, (uint32_t)k, iters, 0, NULL, 0, o, sizeof o);
        lfrife_probe(h, 10, (uint32_t)k, iters, 0, NULL, 0, o, sizeof o);
        int macs = k == 2 ? 4 : 8;
        printf("%-26s: %.2f cycles/vmpy (%.1f MAC/cycle), %.3f ms; clock %.0f MHz\n", nm[k],
               (double)o[1] / ((double)iters * macs), 64.0 * macs * iters / (double)o[1],
               o[0] / 19200.0, (double)o[1] / (o[0] / 19.2));
    }
}

static void probe12(lfrife *h) {
    const char *nm[] = {"8x vmpy_rt only", "8x vadd_qf16 only", "8x(vmpy_rt+vadd)", "8x int16 vmpyacc rt (pair)"};
    for (int k = 0; k < 4; k++) {
        uint64_t o[2];
        uint32_t iters = 100000;
        lfrife_probe(h, 10, 10u + k, iters, 0, NULL, 0, o, sizeof o);
        lfrife_probe(h, 10, 10u + k, iters, 0, NULL, 0, o, sizeof o);
        printf("%-30s: %.2f cycles/op\n", nm[k], (double)o[1] / ((double)iters * 8));
    }
}

static void probe11(lfrife *h) {
    for (int k = 1; k <= 3; k++)
        for (int nt = 1; nt <= 6; nt++) {
            uint64_t o[3];
            uint32_t iters = 100000;
            lfrife_probe(h, 11, (uint32_t)k, iters, (uint32_t)nt, NULL, 0, o, sizeof o);
            lfrife_probe(h, 11, (uint32_t)k, iters, (uint32_t)nt, NULL, 0, o, sizeof o);
            int macs = k == 2 ? 4 : 8;
            printf("kernel %d nt=%d: wall %.3f ms, per-thread %.2f cyc/vmpy, aggregate %.1f MAC/cycle (rc %d)\n", k, nt,
                   o[0] / 19200.0, (double)o[1] / ((double)iters * macs),
                   64.0 * macs * iters * nt / (o[0] / 19.2 * 1.0) / 1248.0 * 1.0 , (int)o[2]);
        }
}

static void probe2(lfrife *h, uint32_t sz) {
    uint32_t o[16] = {0};
    int rc = lfrife_probe(h, 2, sz, 0, 0, NULL, 0, o, sizeof o);
    printf("probe2 rc=%d malloc=%u memalign=%u calloc=%u free=%u HAP_malloc=%u posix_memalign=%u mmap=%u hvx_units=0x%x ptr(%u)=0x%x\n",
           rc, o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], sz, o[8]);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}
static int cmp_d(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

/* run <threads> <iters> <pair>... : accuracy vs golden + cold/warm timing + per-group breakdown */
static int run_mode(const char *dir, int nthreads, int iters, int npairs, char **pairs) {
    const int w = 320, h = 192;
    char p[512];
    snprintf(p, sizeof p, "%s/rife46_hvx.bin", dir);
    lfrife_config cfg = {.skel_dir = dir, .weights_path = p, .width = w, .height = h, .nthreads = nthreads, .qos = -1};
    lfrife *r = NULL;
    double t0 = now_ms();
    int rc = lfrife_open(&cfg, &r);
    printf("open+setup: %d (%s) %.0f ms raw 0x%x\n", rc, lfrife_strerror(rc), now_ms() - t0, r ? lfrife_last_rc(r) : 0);
    if (rc) return 1;
    lfrife_set_profiling(r, 1);
    float *flow = malloc(sizeof(float) * 4 * w * h), *mask = malloc(sizeof(float) * w * h);
    double *ts = malloc(sizeof(double) * (size_t)iters), *dsp = malloc(sizeof(double) * (size_t)iters);
    double grp[16] = {0};
    for (int k = 0; k < npairs; k++) {
        char q[512];
        size_t n;
        snprintf(q, sizeof q, "%s/golden/%s_img0.rgba", dir, pairs[k]);
        uint8_t *i0 = lf_read_file(q, &n);
        snprintf(q, sizeof q, "%s/golden/%s_img1.rgba", dir, pairs[k]);
        uint8_t *i1 = lf_read_file(q, &n);
        snprintf(q, sizeof q, "%s/golden/%s_flow.f32", dir, pairs[k]);
        float *gf = lf_read_file(q, &n);
        snprintf(q, sizeof q, "%s/golden/%s_mask.f32", dir, pairs[k]);
        float *gm = lf_read_file(q, &n);
        snprintf(q, sizeof q, "%s/golden", dir);
        float t = lf_pair_t(q, pairs[k]);
        if (!i0 || !i1 || !gf || !gm) { printf("missing golden for %s\n", pairs[k]); return 1; }
        for (int i = 0; i < iters; i++) {
            double a = now_ms();
            rc = lfrife_flow(r, i0, i1, w * 4, w, h, t, flow, mask);
            ts[i] = now_ms() - a;
            lfrife_stats st;
            lfrife_get_stats(r, &st);
            dsp[i] = st.total_ms;
            if (i > 0) for (int g = 0; g < 16; g++) grp[g] += st.group_ms[g];
            if (rc) { printf("lfrife_flow failed: %d (%s) raw 0x%x\n", rc, lfrife_strerror(rc), lfrife_last_rc(r)); return 1; }
        }
        lf_cmp c = lf_compare(flow, gf, mask, gm, w * h);
        printf("%s t=%.3f: flow max %.3e mean %.3e frac>0.5px %.4f mask max %.3e mean %.3e\n", pairs[k], t, c.fmax,
               c.fmean, c.fbad, c.mmax, c.mmean);
        double cold = ts[0];
        qsort(ts, (size_t)iters, sizeof(double), cmp_d);
        qsort(dsp, (size_t)iters, sizeof(double), cmp_d);
        printf("  cold %.1f ms; wall p50 %.1f p90 %.1f ms; dsp p50 %.1f p90 %.1f ms (%d iters, %d threads)\n", cold,
               ts[iters / 2], ts[(iters * 9) / 10], dsp[iters / 2], dsp[(iters * 9) / 10], iters, nthreads);
        free(i0); free(i1); free(gf); free(gm);
    }
    if (iters > 1) {
        double den = (double)(iters - 1) * npairs;
        printf("per-group mean ms (build X / conv0+1 / res / deconv+up) per block:\n");
        for (int b = 0; b < 4; b++)
            printf("  block%d: %.2f %.2f %.2f %.2f\n", b, grp[b * 4] / den, grp[b * 4 + 1] / den, grp[b * 4 + 2] / den,
                   grp[b * 4 + 3] / den);
    }
    lfrife_close(r);
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "run") && argc > 4)
        return run_mode("/data/local/tmp/lightforge-hvx", atoi(argv[2]), atoi(argv[3]), argc - 4, argv + 4);
    const char *dir = "/data/local/tmp/lightforge-hvx";
    const char *cmd = argc > 1 ? argv[1] : "probe";
    lfrife_config cfg = {.skel_dir = dir, .qos = -1};
    lfrife *h = NULL;
    int rc = lfrife_open(&cfg, &h);
    printf("lfrife_open: %d (%s) raw 0x%x\n", rc, lfrife_strerror(rc), h ? lfrife_last_rc(h) : 0);
    if (rc) return 1;
    if (!strcmp(cmd, "probe1")) probe1(h);
    else if (!strcmp(cmd, "probe3")) probe3(h);
    else if (!strcmp(cmd, "probe10")) probe10(h);
    else if (!strcmp(cmd, "probe12")) probe12(h);
    else if (!strcmp(cmd, "probe11")) probe11(h);
    else if (!strcmp(cmd, "probe2")) probe2(h, argc > 2 ? (uint32_t)atoi(argv[2]) : 0);
    lfrife_close(h);
    return 0;
}
