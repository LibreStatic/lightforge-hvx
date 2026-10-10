/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * JNI port of apps/lfhvx_bench.c for the installed-app test (untrusted_app).
 * libcdsprpc.so is dlopen'ed; every symbol, struct and constant is declared
 * here from the BSD-3 quic/fastrpc sources (inc/remote.h, inc/rpcmem.h).
 * Output goes to logcat (tag LFHVX) and is returned to the activity.
 */
#define _GNU_SOURCE
#include <android/log.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <jni.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "../../inc/lfhvx_proto.h"

/* ---- output: logcat + accumulated text for the UI ----------------------- */
static char g_out[32768];
static size_t g_out_len;

static void out(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void out(const char *fmt, ...) {
    char line[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    __android_log_write(ANDROID_LOG_INFO, "LFHVX", line);
    size_t n = strlen(line);
    if (g_out_len + n + 2 < sizeof(g_out)) {
        memcpy(g_out + g_out_len, line, n);
        g_out_len += n;
        g_out[g_out_len++] = '\n';
        g_out[g_out_len] = 0;
    }
}

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

struct remote_dsp_capability {
    uint32_t domain;
    uint32_t attribute_ID;
    uint32_t capability;
};

#define DSPRPC_CONTROL_UNSIGNED_MODULE 2 /* enum session_control_req_id */
#define DSPRPC_CONTROL_LATENCY 1         /* enum handle_control_req_id */
#define DSPRPC_GET_DSP_INFO 2
#define CDSP_DOMAIN_ID 3
#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS 1

typedef int (*pfn_session_control)(uint32_t req, void *data, uint32_t len);
typedef int (*pfn_open)(const char *uri, uint64_t *h);
typedef int (*pfn_invoke)(uint64_t h, uint32_t sc, remote_arg *pra);
typedef int (*pfn_close)(uint64_t h);
typedef int (*pfn_handle_control)(uint64_t h, uint32_t req, void *data, uint32_t len);
typedef int (*pfn_remote_handle_control)(uint32_t req, void *data, uint32_t len);
typedef void *(*pfn_rpcmem_alloc)(int heapid, uint32_t flags, int size);
typedef void (*pfn_rpcmem_free)(void *p);

static pfn_session_control p_session_control;
static pfn_open p_open;
static pfn_invoke p_invoke;
static pfn_close p_close;
static pfn_handle_control p_handle_control;
static pfn_remote_handle_control p_remote_handle_control;
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

static void *buf_alloc(size_t n) {
    if (p_rpcmem_alloc) {
        void *p = p_rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int)n);
        if (p) return p;
        out("rpcmem_alloc(%zu) failed, falling back to malloc", n);
    }
    void *p = NULL;
    if (posix_memalign(&p, 4096, n)) return NULL;
    return p;
}

static void buf_free(void *p) {
    if (p_rpcmem_free)
        p_rpcmem_free(p);
    else
        free(p);
}

static int run_kernel(uint64_t h, int method, void *src, void *dst, uint32_t len, uint32_t reps,
                      uint32_t flags, lfhvx_kernel_result *res, double *wall_us) {
    lfhvx_kernel_args args = {.len = len, .reps = reps, .flags = flags};
    remote_arg pra[4];
    memset(res, 0, sizeof(*res));
    pra[0].buf.pv = &args;
    pra[0].buf.nLen = sizeof(args);
    pra[1].buf.pv = src;
    pra[1].buf.nLen = len;
    uint32_t sc;
    if (method == LFHVX_M_MAC_HVX || method == LFHVX_M_MAC_MT) {
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
    case LFHVX_M_MAC_HVX: return "i16 MAC HVX";
    default: return "?";
    }
}

/* Saturating per-lane reference for the MAC, matching the 4-accumulator
 * split of the skel over the vectors [v0, v1) of the buffer. */
static int64_t mac_reference(const int16_t *s16, uint32_t nvec) {
    static int64_t lanes[4][32];
    memset(lanes, 0, sizeof(lanes));
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
    return ref;
}

static int copy_file(const char *from, const char *to) {
    int in = open(from, O_RDONLY | O_CLOEXEC);
    if (in < 0) return -errno;
    int o = open(to, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (o < 0) {
        int e = errno;
        close(in);
        return -e;
    }
    char b[8192];
    ssize_t n;
    int rc = 0;
    while ((n = read(in, b, sizeof(b))) > 0)
        if (write(o, b, (size_t)n) != n) rc = -EIO;
    close(in);
    close(o);
    return rc;
}

static void probe_nodes(void) {
    const char *nodes[] = {"/dev/fastrpc-cdsp", "/dev/fastrpc-cdsp-secure", "/dev/adsprpc-smd"};
    for (unsigned i = 0; i < sizeof(nodes) / sizeof(nodes[0]); i++) {
        int fd = open(nodes[i], O_RDONLY | O_NONBLOCK | O_CLOEXEC);
        out("probe open(%s) -> %s", nodes[i], fd >= 0 ? "OK" : strerror(errno));
        if (fd >= 0) close(fd);
    }
}

static void dsp_info(void) {
    if (!p_remote_handle_control) return;
    static const char *names[] = {"DOMAIN_SUPPORT", "UNSIGNED_PD_SUPPORT", "HVX_SUPPORT_64B",
                                  "HVX_SUPPORT_128B", "VTCM_PAGE", "VTCM_COUNT", "ARCH_VER",
                                  "HMX_SUPPORT_DEPTH", "HMX_SUPPORT_SPATIAL"};
    for (uint32_t a = 0; a < sizeof(names) / sizeof(names[0]); a++) {
        struct remote_dsp_capability cap = {CDSP_DOMAIN_ID, a, 0};
        int rc = p_remote_handle_control(DSPRPC_GET_DSP_INFO, &cap, sizeof(cap));
        out("DSP_INFO %-19s rc=0x%x value=0x%x (%u)", names[a], rc, cap.capability, cap.capability);
    }
}

static void bench_mt(uint64_t h, void *src, uint32_t size) {
    /* multithreaded int16 MAC: skel splits the buffer over N qurt threads,
     * each running the same vdmpy kernel under its own HVX context. */
    const int16_t *s16 = src;
    uint32_t reps = 32;
    lfhvx_kernel_result res;
    double wall;
    uint64_t base_cycles = 0;
    for (uint32_t nt = 1; nt <= 4; nt++) {
        int rc = run_kernel(h, LFHVX_M_MAC_MT, src, NULL, size, 1, nt, &res, &wall);
        if (rc) {
            out("MAC MT threads=%u: rc=0x%x (hvx_lock=%d)", nt, rc, res.hvx_lock);
            continue;
        }
        /* reference: sum of per-chunk references */
        uint32_t nvec = size / 128, per = nvec / nt;
        int64_t ref = 0;
        for (uint32_t t = 0; t < nt; t++) {
            uint32_t v0 = t * per, cnt = t == nt - 1 ? nvec - v0 : per;
            ref += mac_reference(s16 + (size_t)v0 * 64, cnt);
        }
        int ok = ref == res.checksum;
        uint64_t best = 0, best_ticks = 0;
        int units = res.hvx_mode;
        for (int t = 0; t < 5; t++) {
            rc = run_kernel(h, LFHVX_M_MAC_MT, src, NULL, size, reps, nt, &res, &wall);
            if (rc) break;
            if (!best || res.ticks < best_ticks) {
                best = res.pcycles;
                best_ticks = res.ticks;
            }
        }
        if (nt == 1) base_cycles = best_ticks;
        double bytes = (double)size * reps;
        out("MAC MT threads=%u: %s, hvx_units=%d, %u reps wall-on-DSP %.3f ms, %.2f GB/s, "
            "speedup x%.2f",
            nt, ok ? "OK" : "WRONG", units, reps, best_ticks / 19200.0,
            best_ticks ? bytes / (best_ticks / 19.2e6) / 1e9 : 0,
            best_ticks ? (double)base_cycles / best_ticks : 0);
    }
}

static void vtcm_probe(uint64_t h) {
    lfhvx_vtcm_info vi;
    memset(&vi, 0, sizeof(vi));
    remote_arg pra[1];
    pra[0].buf.pv = &vi;
    pra[0].buf.nLen = sizeof(vi);
    int rc = p_invoke(h, LF_SCALARS_MAKE(LFHVX_M_VTCM, 0, 1), pra);
    out("VTCM probe rc=0x%x: syms compute_res_attr_init=%d acquire=%d HAP_request_VTCM=%d "
        "qurt_hvx_get_units=%d",
        rc, vi.has_attr_init, vi.has_acquire, vi.has_request_vtcm, vi.has_hvx_units);
    out("VTCM probe: hvx_units=0x%x compute_res: init=%d set_vtcm=%d ctx=0x%x ptr=0x%x "
        "size=%u, rw_ok=%d, release=%d; HAP_request_VTCM(%u)=0x%x",
        vi.hvx_units, vi.attr_init_rc, vi.set_vtcm_rc, vi.ctx_id, vi.vtcm_ptr, vi.vtcm_size,
        vi.vtcm_rw_ok, vi.release_rc, vi.req_size, vi.req_ptr);
    out("VTCM probe: u8 sat-add 128 KiB in VTCM, 64 reps: %.2f B/cycle (DDR-backed same size %.2f "
        "B/cycle)",
        vi.vtcm_pcycles ? (double)vi.bench_bytes / vi.vtcm_pcycles : 0,
        vi.ddr_pcycles ? (double)vi.bench_bytes / vi.ddr_pcycles : 0);
}

/* mode: 0 = skel from nativeLibraryDir, 1 = skel copied to filesDir,
 *       2 = nativeLibraryDir without the explicit unsigned-module request */
static int run_all(const char *native_dir, const char *files_dir, int mode) {
    g_out_len = 0;
    g_out[0] = 0;
    out("=== lfhvx app test, mode %d (uid %d, pid %d) ===", mode, getuid(), getpid());
    out("nativeLibraryDir=%s", native_dir);

    char skel[512];
    snprintf(skel, sizeof(skel), "%s/liblfhvx_skel.so", native_dir);
    struct stat st;
    if (stat(skel, &st) == 0)
        out("skel present in nativeLibraryDir: %lld bytes", (long long)st.st_size);
    else
        out("skel NOT in nativeLibraryDir: %s", strerror(errno));

    const char *libdir = native_dir;
    if (mode == 1) {
        char dst[512];
        snprintf(dst, sizeof(dst), "%s/liblfhvx_skel.so", files_dir);
        int rc = copy_file(skel, dst);
        out("copy skel to %s -> %d", dst, rc);
        libdir = files_dir;
    }

    probe_nodes();

    /* Must happen before libcdsprpc initializes; the DSP fetches the skel
     * through reverse-RPC file reads on these paths. */
    char path[1024];
    snprintf(path, sizeof(path), "%s;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/vendor/dsp", libdir);
    setenv("ADSP_LIBRARY_PATH", path, 1);
    setenv("DSP_LIBRARY_PATH", path, 1);
    out("DSP_LIBRARY_PATH=%s", path);

    void *lib = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        out("dlopen(libcdsprpc.so) FAILED: %s", dlerror());
        return 2;
    }
    out("dlopen(libcdsprpc.so) OK");
    p_session_control = (pfn_session_control)dlsym(lib, "remote_session_control");
    p_open = (pfn_open)dlsym(lib, "remote_handle64_open");
    p_invoke = (pfn_invoke)dlsym(lib, "remote_handle64_invoke");
    p_close = (pfn_close)dlsym(lib, "remote_handle64_close");
    p_handle_control = (pfn_handle_control)dlsym(lib, "remote_handle64_control");
    p_remote_handle_control = (pfn_remote_handle_control)dlsym(lib, "remote_handle_control");
    p_rpcmem_alloc = (pfn_rpcmem_alloc)dlsym(lib, "rpcmem_alloc");
    p_rpcmem_free = (pfn_rpcmem_free)dlsym(lib, "rpcmem_free");
    if (!p_session_control || !p_open || !p_invoke || !p_close) {
        out("missing libcdsprpc symbols");
        return 2;
    }

    dsp_info();

    if (mode != 2) {
        struct remote_rpc_control_unsigned_module ctl = {.domain = CDSP_DOMAIN_ID, .enable = 1};
        int rc = p_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &ctl, sizeof(ctl));
        out("remote_session_control(UNSIGNED_MODULE, cdsp) = 0x%x", rc);
        if (rc) return 3;
    }

    const char *uri = "file:///liblfhvx_skel.so?lfhvx_skel_handle_invoke&_modver=1.0&_dom=cdsp";
    uint64_t h = 0;
    double t0 = now_us();
    int rc = p_open(uri, &h);
    out("remote_handle64_open = 0x%x, h=0x%" PRIx64 " (%.1f ms)", rc, h, (now_us() - t0) / 1000.0);
    if (rc) return 4;

    lfhvx_info info;
    remote_arg pra[4];
    pra[0].buf.pv = &info;
    pra[0].buf.nLen = sizeof(info);
    rc = p_invoke(h, LF_SCALARS_MAKE(LFHVX_M_INFO, 0, 1), pra);
    out("info rc=0x%x hvx_lock_rc=%d hvx_mode_locked=%d skel_v=%u", rc, info.hvx_lock_rc,
        (int)info.hvx_mode_locked, info.skel_version);

    /* add() */
    int32_t ab[2] = {40, 2}, sum = 0;
    pra[0].buf.pv = ab;
    pra[0].buf.nLen = sizeof(ab);
    pra[1].buf.pv = &sum;
    pra[1].buf.nLen = sizeof(sum);
    rc = p_invoke(h, LF_SCALARS_MAKE(LFHVX_M_ADD, 1, 1), pra);
    out("add(40, 2) rc=0x%x -> %d %s", rc, sum, (!rc && sum == 42) ? "OK" : "WRONG");
    if (rc || sum != 42) return 5;
    enum { ITERS = 1000 };
    static double lat[ITERS];
    for (int i = 0; i < 100; i++) p_invoke(h, LF_SCALARS_MAKE(LFHVX_M_ADD, 1, 1), pra);
    int bad = 0;
    for (int i = 0; i < ITERS; i++) {
        ab[0] = i;
        ab[1] = 3 * i;
        double s = now_us();
        rc = p_invoke(h, LF_SCALARS_MAKE(LFHVX_M_ADD, 1, 1), pra);
        lat[i] = now_us() - s;
        if (rc || sum != 4 * i) bad++;
    }
    qsort(lat, ITERS, sizeof(double), cmp_double);
    out("add() x%d: min %.1f us, p50 %.1f us, p90 %.1f us, p99 %.1f us, errors %d", ITERS, lat[0],
        lat[ITERS / 2], lat[ITERS * 9 / 10], lat[ITERS * 99 / 100], bad);

    /* kernels, 1 MiB, rpcmem buffers */
    const uint32_t size = 1u << 20;
    uint8_t *src = buf_alloc(size), *dst = buf_alloc(size);
    if (!src || !dst) return 6;
    for (uint32_t i = 0; i < size; i++) src[i] = (uint8_t)(i * 7 + (i >> 8));
    int all_ok = 1;
    const int methods[] = {LFHVX_M_U8ADD_HVX, LFHVX_M_U8ADD_SCALAR, LFHVX_M_MAC_HVX};
    for (unsigned mi = 0; mi < 3; mi++) {
        int m = methods[mi];
        lfhvx_kernel_result res;
        double wall;
        memset(dst, 0, size);
        rc = run_kernel(h, m, src, dst, size, 1, 0, &res, &wall);
        if (rc) {
            out("%s: rc=0x%x hvx_lock=%d", method_name(m), rc, res.hvx_lock);
            all_ok = 0;
            continue;
        }
        int ok = 1;
        if (m != LFHVX_M_MAC_HVX) {
            for (uint32_t i = 0; i < size; i++) {
                unsigned v = src[i] * 2u;
                if (dst[i] != (v > 255 ? 255 : v)) {
                    ok = 0;
                    out("  mismatch at %u", i);
                    break;
                }
            }
        } else {
            int64_t ref = mac_reference((const int16_t *)src, size / 128);
            ok = ref == res.checksum;
            if (!ok) out("  checksum %" PRId64 " vs ref %" PRId64, res.checksum, ref);
        }
        all_ok &= ok;
        out("%-17s 1 rep: wall %.3f ms, DSP %.3f ms, %.2f B/cycle, clk~%.0f MHz, %s",
            method_name(m), wall / 1000.0, res.ticks / 19200.0,
            res.pcycles ? (double)size / res.pcycles : 0,
            res.ticks ? res.pcycles * 19.2 / res.ticks : 0, ok ? "BIT-EXACT" : "WRONG");
        uint32_t reps = m == LFHVX_M_U8ADD_SCALAR ? 4 : 32;
        lfhvx_kernel_result best = {0};
        for (int t = 0; t < 5; t++) {
            rc = run_kernel(h, m, src, dst, size, reps, 0, &res, &wall);
            if (rc) break;
            if (!best.pcycles || res.pcycles < best.pcycles) best = res;
        }
        double bytes = (double)size * reps;
        out("%-17s %u reps: %.2f B/cycle, %.2f GB/s", method_name(m), reps,
            best.pcycles ? bytes / best.pcycles : 0,
            best.ticks ? bytes / (best.ticks / 19.2e6) / 1e9 : 0);
    }

    if (getenv("LFHVX_NO_BONUS") == NULL) {
        vtcm_probe(h);
        bench_mt(h, src, size);
    }

    buf_free(src);
    buf_free(dst);
    rc = p_close(h);
    out("remote_handle64_close = 0x%x", rc);
    out("RESULT: %s", all_ok ? "PASS (all kernels bit-exact)" : "FAIL");
    return all_ok ? 0 : 7;
}

JNIEXPORT jstring JNICALL Java_com_librestatic_lfhvxtest_MainActivity_run(JNIEnv *env, jclass cls,
                                                                         jstring jnative,
                                                                         jstring jfiles, jint mode) {
    (void)cls;
    const char *nd = (*env)->GetStringUTFChars(env, jnative, NULL);
    const char *fd = (*env)->GetStringUTFChars(env, jfiles, NULL);
    int rc = run_all(nd, fd, mode);
    out("exit code %d", rc);
    (*env)->ReleaseStringUTFChars(env, jnative, nd);
    (*env)->ReleaseStringUTFChars(env, jfiles, fd);
    return (*env)->NewStringUTF(env, g_out);
}
