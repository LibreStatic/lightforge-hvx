/*
 * SPDX-License-Identifier: MIT
 *
 * Portable plain-C (C11, no dependencies) fp32 implementation of the RIFE v4.6
 * IFNet flow/mask network. Oracle for the HVX kernels and CPU fallback.
 */
#ifndef LFRIFE_REF_H
#define LFRIFE_REF_H

#include <stddef.h>
#include <stdint.h>

typedef struct lfref lfref;

/* blob: contents of rife46_f32.bin (must stay alive). w,h: multiples of 32. */
lfref *lfref_create(const void *blob, size_t size, int w, int h);
void lfref_destroy(lfref *r);

/*
 * rgba0/rgba1: 8-bit RGBA, row stride in bytes. t: timestep 0..1.
 * flow: 4*h*w floats (planar; [0:2] towards img0, [2:4] towards img1, px at input resolution).
 * mask: h*w floats (after sigmoid).
 */
int lfref_run(lfref *r, const uint8_t *rgba0, const uint8_t *rgba1, int stride, float t, float *flow,
              float *mask);

/* Debug: when enabled, the cumulative flow/mask after each block (NHWC, 5 floats/px) is kept. */
void lfref_keep_debug(lfref *r, int enable);
const float *lfref_debug_block(const lfref *r, int block); /* h*w*5 floats */

#endif
