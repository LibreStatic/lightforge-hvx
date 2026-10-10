/* SPDX-License-Identifier: BSD-3-Clause
 * Runs the DSP network source (dsp/lfrife_net.c) on the host with the HVX emulation
 * and compares against the goldens. usage: test_hostemu <rife46_hvx.bin> <golden dir> <pair>... */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../dsp/lfrife_net.h"
#include "golden_util.h"

int main(int argc, char **argv) {
    if (argc < 4) return 2;
    size_t bsz;
    void *blob = lf_read_file(argv[1], &bsz);
    if (!blob) return 2;
    const int w = 320, h = 192;
    lfnet *n = lfnet_create(blob, bsz, w, h, 1);
    if (!n) { fprintf(stderr, "lfnet_create failed\n"); return 2; }
    float *flow = malloc(sizeof(float) * 4 * w * h), *mask = malloc(sizeof(float) * w * h);
    for (int k = 3; k < argc; k++) {
        char p[512];
        size_t sz;
        snprintf(p, sizeof p, "%s/%s_img0.rgba", argv[2], argv[k]);
        uint8_t *i0 = lf_read_file(p, &sz);
        snprintf(p, sizeof p, "%s/%s_img1.rgba", argv[2], argv[k]);
        uint8_t *i1 = lf_read_file(p, &sz);
        snprintf(p, sizeof p, "%s/%s_flow.f32", argv[2], argv[k]);
        float *gf = lf_read_file(p, &sz);
        snprintf(p, sizeof p, "%s/%s_mask.f32", argv[2], argv[k]);
        float *gm = lf_read_file(p, &sz);
        float t = lf_pair_t(argv[2], argv[k]);
        lfnet_run(n, i0, i1, w * 4, t, flow, mask, NULL);
        lf_cmp c = lf_compare(flow, gf, mask, gm, w * h);
        printf("%s t=%.3f: flow max %.3e mean %.3e frac>0.5px %.4f mask max %.3e mean %.3e\n", argv[k], t, c.fmax,
               c.fmean, c.fbad, c.mmax, c.mmean);
    }
    return 0;
}
