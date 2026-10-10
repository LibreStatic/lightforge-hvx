/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * APPS-side (arm64 Android) driver for the lfhvx skel.
 *
 * libcdsprpc.so is loaded with dlopen and every symbol we use is declared here
 * from the BSD-3 quic/fastrpc sources (inc/remote.h, inc/rpcmem.h). No vendor
 * headers are used.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../inc/lfhvx_proto.h"

/* ---- types from quic/fastrpc inc/remote.h ------------------------------- */
typedef struct {
    void *pv;
    size_t nLen;
} remote_buf;

typedef union {
    remote_buf buf;
    uint32_t h;
    uint64_t h64;
    struct {
        int32_t fd;
        uint32_t offset;
    } dma;
} remote_arg;

struct remote_rpc_control_unsigned_module {
    int domain;
    int enable;
};

#define DSPRPC_CONTROL_UNSIGNED_MODULE 2 /* enum session_control_req_id */
#define CDSP_DOMAIN_ID 3
#define CDSP1_DOMAIN_ID 4
#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS 1

typedef int (*pfn_session_control)(uint32_t req, void *data, uint32_t len);
typedef int (*pfn_open)(const char *uri, uint64_t *h);
typedef int (*pfn_invoke)(uint64_t h, uint32_t sc, remote_arg *pra);
typedef int (*pfn_close)(uint64_t h);
typedef int (*pfn_handle_control)(uint64_t h, uint32_t req, void *data, uint32_t len);
typedef void *(*pfn_rpcmem_alloc)(int heapid, uint32_t flags, int size);
typedef void (*pfn_rpcmem_free)(void *p);

static pfn_session_control p_session_control;
static pfn_open p_open;
static pfn_invoke p_invoke;
static pfn_close p_close;
static pfn_handle_control p_handle_control;
static pfn_rpcmem_alloc p_rpcmem_alloc;
static pfn_rpcmem_free p_rpcmem_free;

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static void *load_sym(void *lib, const char *name, int required) {
    void *p = dlsym(lib, name);
    if (!p && required) {
        fprintf(stderr, "missing symbol %s: %s\n", name, dlerror());
        exit(2);
    }
    return p;
}

static int g_use_rpcmem = 1;

static void *buf_alloc(size_t n) {
    if (g_use_rpcmem && p_rpcmem_alloc) {
        void *p = p_rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int)n);
        if (p) return p;
        fprintf(stderr, "rpcmem_alloc(%zu) failed, falling back to malloc\n", n);
    }
    void *p = NULL;
    if (posix_memalign(&p, 4096, n)) return NULL;
    return p;
}

static void buf_free(void *p) {
    if (g_use_rpcmem && p_rpcmem_free)
        p_rpcmem_free(p);
    else
        free(p);
}

static int run_kernel(uint64_t h, int method, void *src, void *dst, uint32_t len, uint32_t reps,
                      lfhvx_kernel_result *res, double *wall_us) {
    lfhvx_kernel_args args = {.len = len, .reps = reps};
    remote_arg pra[4];
    memset(res, 0, sizeof(*res));
    pra[0].buf.pv = &args;
    pra[0].buf.nLen = sizeof(args);
    pra[1].buf.pv = src;
    pra[1].buf.nLen = len;
    uint32_t sc;
    if (method == LFHVX_M_MAC_HVX) {
        pra[2].buf.pv = res;
        pra[2].buf.nLen = sizeof(*res);
        sc = LF_SCALARS_MAKE(method, 2, 1);
    } else {
        pra[2].buf.pv = dst;
        pra[2].buf.nLen = len;
        pra[3].buf.pv = res;
        pra[3].buf.nLen = sizeof(*res);
        sc = LF_SCALARS_MAKE(method, 2, 2);
    }
    double t0 = now_us();
    int rc = p_invoke(h, sc, pra);
    *wall_us = now_us() - t0;
    return rc;
}

static const char *method_name(int m) {
    switch (m) {
    case LFHVX_M_U8ADD_HVX: return "u8 sat-add HVX";
    case LFHVX_M_U8ADD_SCALAR: return "u8 sat-add scalar";
    case LFHVX_M_MAC_HVX: return "i16 vdmpy MAC HVX";
    default: return "?";
    }
}

int main(int argc, char **argv) {
    const char *dom = "cdsp";
    int domain_id = CDSP_DOMAIN_ID;
    int unsigned_pd = 1;
    uint32_t size = 1 << 20;
    int add_iters = 2000;
    int qos = -1; /* remote_rpc_latency_flags: 0 off, 1 PM_QOS, 2 ADAPTIVE, 3 POLL */
    const char *libdir = "/data/local/tmp/lightforge-hvx";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dom") && i + 1 < argc) {
            dom = argv[++i];
            domain_id = !strcmp(dom, "cdsp1") ? CDSP1_DOMAIN_ID : CDSP_DOMAIN_ID;
        } else if (!strcmp(argv[i], "--signed")) {
            unsigned_pd = 0;
        } else if (!strcmp(argv[i], "--malloc")) {
            g_use_rpcmem = 0;
        } else if (!strcmp(argv[i], "--size") && i + 1 < argc) {
            size = (uint32_t)strtoul(argv[++i], NULL, 0);
        } else if (!strcmp(argv[i], "--qos") && i + 1 < argc) {
            qos = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--libdir") && i + 1 < argc) {
            libdir = argv[++i];
        } else {
            fprintf(stderr,
                    "usage: %s [--dom cdsp|cdsp1] [--signed] [--malloc] [--size N] [--qos 0|1|2|3] [--libdir D]\n",
                    argv[0]);
            return 1;
        }
    }
    size &= ~127u;

    /* The DSP loads the skel through reverse-RPC file reads on these paths. */
    char path[512];
    snprintf(path, sizeof(path), "%s;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/vendor/dsp", libdir);
    setenv("ADSP_LIBRARY_PATH", path, 1);
    setenv("DSP_LIBRARY_PATH", path, 1);

    void *lib = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        fprintf(stderr, "dlopen(libcdsprpc.so): %s\n", dlerror());
        lib = dlopen("/vendor/lib64/libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (!lib) {
        fprintf(stderr, "dlopen(/vendor/lib64/libcdsprpc.so): %s\n", dlerror());
        return 2;
    }
    p_session_control = (pfn_session_control)load_sym(lib, "remote_session_control", 1);
    p_open = (pfn_open)load_sym(lib, "remote_handle64_open", 1);
    p_invoke = (pfn_invoke)load_sym(lib, "remote_handle64_invoke", 1);
    p_close = (pfn_close)load_sym(lib, "remote_handle64_close", 1);
    p_handle_control = (pfn_handle_control)load_sym(lib, "remote_handle64_control", 0);
    p_rpcmem_alloc = (pfn_rpcmem_alloc)load_sym(lib, "rpcmem_alloc", 0);
    p_rpcmem_free = (pfn_rpcmem_free)load_sym(lib, "rpcmem_free", 0);

    if (unsigned_pd) {
        struct remote_rpc_control_unsigned_module ctl = {.domain = domain_id, .enable = 1};
        int rc = p_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &ctl, sizeof(ctl));
        printf("remote_session_control(UNSIGNED_MODULE, domain=%d) = 0x%x\n", domain_id, rc);
        if (rc) return 3;
    }

    char uri[256];
    snprintf(uri, sizeof(uri),
             "file:///liblfhvx_skel.so?lfhvx_skel_handle_invoke&_modver=1.0&_dom=%s", dom);
    uint64_t h = 0;
    if (getenv("LFHVX_LOG_WARMUP")) {
        /* Spawn the user PD with a throw-away open and give libcdsprpc's
         * .farf file watcher time to enable DSP logging before the real
         * load, otherwise loader errors are emitted before logging is on. */
        char dummy[128];
        snprintf(dummy, sizeof(dummy), "file:///liblfhvx_nonexistent_skel.so?x&_modver=1.0&_dom=%s", dom);
        uint64_t hd = 0;
        printf("warm-up open (expected to fail) = 0x%x\n", p_open(dummy, &hd));
        usleep(800 * 1000);
    }
    double t0 = now_us();
    int rc = p_open(uri, &h);
    double t_open = now_us() - t0;
    printf("remote_handle64_open(%s) = 0x%x, h=0x%" PRIx64 " (%.1f ms)\n", uri, rc, h,
           t_open / 1000.0);
    if (rc) {
        sleep(2); /* let the DSP log (adspmsgd) thread flush to logcat */
        return 4;
    }

    if (qos >= 0 && p_handle_control) {
        /* struct remote_rpc_control_latency { uint32 enable; uint32 latency_us; } */
        uint32_t lat[2] = {(uint32_t)qos, 100};
        rc = p_handle_control(h, 1 /* DSPRPC_CONTROL_LATENCY */, lat, sizeof(lat));
        printf("remote_handle64_control(DSPRPC_CONTROL_LATENCY, enable=%d, 100us) = 0x%x\n", qos, rc);
    }

    /* ---- info ---- */
    lfhvx_info info;
    remote_arg pra[4];
    pra[0].buf.pv = &info;
    pra[0].buf.nLen = sizeof(info);
    rc = p_invoke(h, LF_SCALARS_MAKE(LFHVX_M_INFO, 0, 1), pra);
    printf("info rc=0x%x pcycle=%" PRIu64 " utimer=%" PRIu64
           " hvx_mode_before=%d hvx_lock_rc=%d hvx_mode_locked=%d skel_v=%u\n",
           rc, info.pcycle, info.utimer, (int)info.hvx_mode_before, info.hvx_lock_rc,
           (int)info.hvx_mode_locked, info.skel_version);

    /* ---- add() round trip ---- */
    int32_t ab[2] = {40, 2}, sum = 0;
    pra[0].buf.pv = ab;
    pra[0].buf.nLen = sizeof(ab);
    pra[1].buf.pv = &sum;
    pra[1].buf.nLen = sizeof(sum);
    rc = p_invoke(h, LF_SCALARS_MAKE(LFHVX_M_ADD, 1, 1), pra);
    printf("add(40, 2) rc=0x%x -> %d\n", rc, sum);
    if (rc || sum != 42) return 5;
    double *lat = malloc(sizeof(double) * add_iters);
    for (int i = 0; i < 100; i++) p_invoke(h, LF_SCALARS_MAKE(LFHVX_M_ADD, 1, 1), pra); /* warm */
    int bad = 0;
    for (int i = 0; i < add_iters; i++) {
        ab[0] = i;
        ab[1] = 3 * i;
        double s = now_us();
        rc = p_invoke(h, LF_SCALARS_MAKE(LFHVX_M_ADD, 1, 1), pra);
        lat[i] = now_us() - s;
        if (rc || sum != 4 * i) bad++;
    }
    qsort(lat, add_iters, sizeof(double), cmp_double);
    double mean = 0;
    for (int i = 0; i < add_iters; i++) mean += lat[i];
    mean /= add_iters;
    printf("add() warm round trip over %d calls: min %.1f us, p50 %.1f us, p90 %.1f us, p99 %.1f "
           "us, mean %.1f us, errors %d\n",
           add_iters, lat[0], lat[add_iters / 2], lat[add_iters * 9 / 10],
           lat[add_iters * 99 / 100], mean, bad);

    /* ---- kernels ---- */
    printf("buffers: %s, size %u bytes\n", g_use_rpcmem ? "rpcmem_alloc (dmabuf)" : "malloc", size);
    uint8_t *src = buf_alloc(size), *dst = buf_alloc(size);
    if (!src || !dst) return 6;
    for (uint32_t i = 0; i < size; i++) src[i] = (uint8_t)(i * 7 + (i >> 8));

    const int methods[] = {LFHVX_M_U8ADD_HVX, LFHVX_M_U8ADD_SCALAR, LFHVX_M_MAC_HVX};
    for (unsigned mi = 0; mi < sizeof(methods) / sizeof(methods[0]); mi++) {
        int m = methods[mi];
        lfhvx_kernel_result res;
        double wall;
        /* correctness + wall clock including transfer, 1 rep */
        memset(dst, 0, size);
        rc = run_kernel(h, m, src, dst, size, 1, &res, &wall);
        if (rc) {
            printf("%s: rc=0x%x hvx_lock=%d hvx_mode=%d\n", method_name(m), rc, res.hvx_lock,
                   res.hvx_mode);
            continue;
        }
        int ok = 1;
        if (m != LFHVX_M_MAC_HVX) {
            for (uint32_t i = 0; i < size; i++) {
                unsigned v = src[i] * 2u;
                if (dst[i] != (v > 255 ? 255 : v)) {
                    ok = 0;
                    printf("  mismatch at %u: %u vs %u\n", i, dst[i], v > 255 ? 255 : v);
                    break;
                }
            }
        } else {
            /* reference: per-lane saturating 32-bit dot product of int16 pairs */
            const int16_t *s16 = (const int16_t *)src;
            int64_t lanes[4][32] = {{0}};
            uint32_t nvec = size / 128;
            for (uint32_t v = 0; v < nvec; v++) {
                int acc = v < (nvec & ~3u) ? (int)(v & 3) : 0;
                for (int k = 0; k < 32; k++) {
                    int64_t a0 = s16[v * 64 + 2 * k], a1 = s16[v * 64 + 2 * k + 1];
                    int64_t x = lanes[acc][k] + a0 * a0 + a1 * a1;
                    if (x > INT32_MAX) x = INT32_MAX;
                    if (x < INT32_MIN) x = INT32_MIN;
                    lanes[acc][k] = x;
                }
            }
            int64_t ref = 0;
            for (int k = 0; k < 32; k++) {
                int32_t s01 = (int32_t)(lanes[0][k] + lanes[1][k]);
                int32_t s23 = (int32_t)(lanes[2][k] + lanes[3][k]);
                ref += (int32_t)(s01 + s23);
            }
            ok = ref == res.checksum;
            if (!ok) printf("  checksum %" PRId64 " vs ref %" PRId64 "\n", res.checksum, ref);
        }
        double mhz = res.ticks ? res.pcycles * 19.2 / res.ticks : 0;
        printf("%-18s 1 rep : wall %.3f ms (incl. RPC+transfer), DSP %" PRIu64
               " pcycles, %.3f ms DSP, %.2f B/cycle, clk~%.0f MHz, %s\n",
               method_name(m), wall / 1000.0, res.pcycles, res.ticks / 19200.0,
               res.pcycles ? (double)size / res.pcycles : 0, mhz, ok ? "OK" : "WRONG");
        /* DSP-side throughput with the buffer hot in L2 */
        uint32_t reps = m == LFHVX_M_U8ADD_SCALAR ? 4 : 32;
        double best_wall = 1e30;
        lfhvx_kernel_result best = {0};
        for (int t = 0; t < 5; t++) {
            rc = run_kernel(h, m, src, dst, size, reps, &res, &wall);
            if (rc) break;
            if (!best.pcycles || res.pcycles < best.pcycles) best = res;
            if (wall < best_wall) best_wall = wall;
        }
        mhz = best.ticks ? best.pcycles * 19.2 / best.ticks : 0;
        double bytes = (double)size * reps;
        printf("%-18s %u reps: best DSP %" PRIu64 " pcycles, %.2f B/cycle, %.2f GB/s at ~%.0f MHz, "
               "best wall %.3f ms\n",
               method_name(m), reps, best.pcycles, best.pcycles ? bytes / best.pcycles : 0,
               best.ticks ? bytes / (best.ticks / 19.2e6) / 1e9 : 0, mhz, best_wall / 1000.0);
    }

    /* wall-clock of repeated 1-rep HVX calls (steady state, incl. transfer) */
    {
        double w[20];
        lfhvx_kernel_result res;
        for (int t = 0; t < 20; t++) run_kernel(h, LFHVX_M_U8ADD_HVX, src, dst, size, 1, &res, &w[t]);
        qsort(w, 20, sizeof(double), cmp_double);
        printf("u8 sat-add HVX 1 rep x20: wall min %.3f ms, p50 %.3f ms (%u bytes in + out)\n",
               w[0] / 1000.0, w[10] / 1000.0, size);
    }

    buf_free(src);
    buf_free(dst);
    rc = p_close(h);
    printf("remote_handle64_close = 0x%x\n", rc);
    return 0;
}
