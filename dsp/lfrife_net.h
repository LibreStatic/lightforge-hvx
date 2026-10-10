/* SPDX-License-Identifier: MIT
 * RIFE v4.6 IFNet flow network, HVX implementation (also builds on the host with LF_HOST). */
#ifndef LFRIFE_NET_H
#define LFRIFE_NET_H
#include <stddef.h>
#include <stdint.h>

#include "../inc/lfrife_proto.h"

typedef struct lfnet lfnet;

/* blob: rife46_hvx.bin contents, must stay alive and be readable (128-byte aligned start). */
lfnet *lfnet_create(const void *blob, size_t size, int w, int h, int nthreads);
void lfnet_destroy(lfnet *n);
size_t lfnet_scratch_bytes(const lfnet *n);
/* flow: 4*h*w floats (planar), mask: h*w floats. Fills stats->group_ticks when profiling. */
int lfnet_run(lfnet *n, const uint8_t *rgba0, const uint8_t *rgba1, int stride, float t, float *flow, float *mask,
              lfrife_run_stats *stats);
#endif
