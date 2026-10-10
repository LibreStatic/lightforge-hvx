/*
 * SPDX-License-Identifier: MIT
 *
 * Minimal FastRPC skel for the Hexagon cDSP, built with upstream clang/lld only.
 * No Hexagon SDK, no QAIC: the marshalling is hand-written against the
 * protocol documented in inc/lfhvx_proto.h.
 *
 * Every OS symbol used here (qurt_*, HAP_*) is declared by us and resolved at
 * load time by the DSP's dynamic loader against the FastRPC user PD shell.
 */
#include <stddef.h>
#include <stdint.h>

#include "../inc/lfhvx_proto.h"

#define EXPORT __attribute__((visibility("default")))

/* ---- DSP-side remote_arg layout (32-bit pointers) ---------------------- */
typedef struct {
    void *pv;
    uint32_t nLen;
} dsp_remote_buf;

typedef union {
    dsp_remote_buf buf;
    uint32_t h;
    uint64_t h64;
    struct {
        int32_t fd;
        uint32_t offset;
    } dma;
} dsp_remote_arg;

_Static_assert(sizeof(dsp_remote_arg) == 8, "remote_arg must be 8 bytes on hexagon");

/* ---- OS symbols provided by the user PD (declared weak so a missing one
 *      shows up as a NULL instead of a load failure) ----------------------- */
extern int qurt_hvx_lock(int mode) __attribute__((weak));
extern int qurt_hvx_unlock(void) __attribute__((weak));
extern int qurt_hvx_get_mode(void) __attribute__((weak));
extern void HAP_debug_v2(int level, const char *file, int line, const char *fmt, ...)
    __attribute__((weak));

/* QuRT threads. Prototypes and the attribute layout follow Halide's MIT
 * src/runtime/mini_qurt.h; the struct is padded because newer QuRT releases
 * may have appended fields (the leading fields keep their offsets). */
typedef struct {
    char name[16];
    unsigned char tcb_partition;
    unsigned char affinity;       /* "stid" on newer QuRT */
    unsigned short priority;
    unsigned char asid;           /* "autostack" on newer QuRT */
    unsigned char bus_priority;   /* "group_id" on newer QuRT */
    unsigned short timetest_id;
    unsigned int stack_size;
    void *stack_addr;
    unsigned char pad[96];
} lf_qurt_thread_attr;
extern int qurt_thread_create(unsigned int *tid, lf_qurt_thread_attr *attr, void (*entry)(void *),
                              void *arg) __attribute__((weak));
extern int qurt_thread_join(unsigned int tid, int *status) __attribute__((weak));
extern unsigned int qurt_thread_get_id(void) __attribute__((weak));
extern int qurt_thread_get_priority(unsigned int tid) __attribute__((weak));
extern int qurt_hvx_get_units(void) __attribute__((weak));

/* VTCM. Legacy HAP_request_VTCM/HAP_release_VTCM prototypes come from
 * Halide's MIT mini_qurt_vtcm.h. The compute-resource-manager calls are
 * declared from their use in llama.cpp's MIT ggml-hexagon/htp/main.c; the
 * attribute object is treated as opaque (a zeroed, generously sized blob). */
extern void *HAP_request_VTCM(unsigned int size, unsigned int single_page_flag) __attribute__((weak));
extern int HAP_release_VTCM(void *va) __attribute__((weak));
typedef struct {
    uint64_t opaque[32];
} lf_compute_res_attr;
extern int HAP_compute_res_query_VTCM(unsigned int client, unsigned int *total_size,
                                      void *total_layout, unsigned int *avail_size,
                                      void *avail_layout) __attribute__((weak));
extern int HAP_compute_res_attr_init(lf_compute_res_attr *attr) __attribute__((weak));
extern int HAP_compute_res_attr_set_serialize(lf_compute_res_attr *attr, unsigned char b)
    __attribute__((weak));
extern int HAP_compute_res_attr_set_vtcm_param(lf_compute_res_attr *attr, unsigned int size,
                                               unsigned char single_page) __attribute__((weak));
extern int HAP_compute_res_attr_set_vtcm_param_v2(lf_compute_res_attr *attr, unsigned int size,
                                                  unsigned int min_page, unsigned int min_size)
    __attribute__((weak));
extern unsigned int HAP_compute_res_acquire(lf_compute_res_attr *attr, unsigned int timeout_us)
    __attribute__((weak));
extern void *HAP_compute_res_attr_get_vtcm_ptr(lf_compute_res_attr *attr) __attribute__((weak));
extern int HAP_compute_res_attr_get_vtcm_ptr_v2(lf_compute_res_attr *attr, void **ptr,
                                                unsigned int *size) __attribute__((weak));
extern int HAP_compute_res_release(unsigned int ctx) __attribute__((weak));
/* Candidate names for the PD-exported backend of the HAP_compute_res_* API.
 * Not documented in any open source: probed as weak imports (NULL if the
 * shell does not export them), with the same argument lists as the HAP_ forms. */
extern int compute_resource_attr_init(lf_compute_res_attr *attr) __attribute__((weak));
extern int compute_resource_attr_set_serialize(lf_compute_res_attr *attr, unsigned char b)
    __attribute__((weak));
extern int compute_resource_attr_set_vtcm_param(lf_compute_res_attr *attr, unsigned int size,
                                                unsigned char single_page) __attribute__((weak));
extern int compute_resource_attr_set_vtcm_param_v2(lf_compute_res_attr *attr, unsigned int size,
                                                   unsigned int min_page, unsigned int min_size)
    __attribute__((weak));
extern unsigned int compute_resource_acquire(lf_compute_res_attr *attr, unsigned int timeout_us)
    __attribute__((weak));
extern void *compute_resource_attr_get_vtcm_ptr(lf_compute_res_attr *attr) __attribute__((weak));
extern int compute_resource_attr_get_vtcm_ptr_v2(lf_compute_res_attr *attr, void **ptr,
                                                 unsigned int *size) __attribute__((weak));
extern int compute_resource_release(unsigned int ctx) __attribute__((weak));
extern int compute_resource_query_VTCM(unsigned int client, unsigned int *total_size,
                                       void *total_layout, unsigned int *avail_size,
                                       void *avail_layout) __attribute__((weak));

#define LOG(...)                                                         \
    do {                                                                 \
        if (HAP_debug_v2) HAP_debug_v2(2 /*HIGH*/, "lfhvx_skel.c", __LINE__, __VA_ARGS__); \
    } while (0)

#define QURT_HVX_MODE_128B 1

/* ---- freestanding helpers --------------------------------------------- */
/* The compiler may emit calls to these; provide hidden local copies. */
__attribute__((visibility("hidden"))) void *memcpy(void *d, const void *s, size_t n) {
    uint8_t *dd = d;
    const uint8_t *ss = s;
    while (n--) *dd++ = *ss++;
    return d;
}
__attribute__((visibility("hidden"))) void *memset(void *d, int c, size_t n) {
    uint8_t *dd = d;
    while (n--) *dd++ = (uint8_t)c;
    return d;
}

static inline uint64_t read_pcycle(void) {
    uint64_t v;
    __asm__ volatile("%0 = upcycle" : "=r"(v));
    return v;
}
static inline uint64_t read_utimer(void) {
    uint64_t v;
    __asm__ volatile("%0 = utimer" : "=r"(v));
    return v;
}

/* ---- HVX kernels -------------------------------------------------------- */
typedef int32_t HVX_Vector __attribute__((vector_size(128), aligned(128)));
typedef int32_t HVX_UVector __attribute__((vector_size(128), aligned(4)));

#define U8ADD_BODY(VT)                                                    \
    for (uint32_t r = 0; r < reps; r++) {                                 \
        const VT *s = (const VT *)src;                                    \
        VT *d = (VT *)dst;                                                \
        for (uint32_t i = 0; i < n; i++) {                                \
            HVX_Vector v = s[i];                                          \
            d[i] = __builtin_HEXAGON_V6_vaddubsat_128B(v, v);             \
        }                                                                 \
    }

static int64_t u8add_hvx(const uint8_t *src, uint8_t *dst, uint32_t len, uint32_t reps) {
    const uint32_t n = len / 128;
    if ((((uintptr_t)src | (uintptr_t)dst) & 127) == 0) {
        U8ADD_BODY(HVX_Vector) /* aligned vmem */
    } else {
        U8ADD_BODY(HVX_UVector) /* unaligned vmemu */
    }
    return dst[0] + dst[len - 1];
}

static int64_t u8add_scalar(const uint8_t *src, uint8_t *dst, uint32_t len, uint32_t reps) {
    for (uint32_t r = 0; r < reps; r++) {
#pragma clang loop vectorize(disable)
        for (uint32_t i = 0; i < len; i++) {
            uint32_t v = (uint32_t)src[i] * 2u;
            dst[i] = v > 255u ? 255u : (uint8_t)v;
        }
    }
    return dst[0] + dst[len - 1];
}

#define MAC_BODY(VT)                                                      \
    for (uint32_t r = 0; r < reps; r++) {                                 \
        const VT *s = (const VT *)src;                                    \
        uint32_t i = 0;                                                   \
        for (; i + 4 <= n; i += 4) {                                      \
            HVX_Vector a = s[i], b = s[i + 1], c = s[i + 2], d = s[i + 3]; \
            acc0 = __builtin_HEXAGON_V6_vdmpyhvsat_acc_128B(acc0, a, a);  \
            acc1 = __builtin_HEXAGON_V6_vdmpyhvsat_acc_128B(acc1, b, b);  \
            acc2 = __builtin_HEXAGON_V6_vdmpyhvsat_acc_128B(acc2, c, c);  \
            acc3 = __builtin_HEXAGON_V6_vdmpyhvsat_acc_128B(acc3, d, d);  \
        }                                                                 \
        for (; i < n; i++) {                                              \
            HVX_Vector a = s[i];                                          \
            acc0 = __builtin_HEXAGON_V6_vdmpyhvsat_acc_128B(acc0, a, a);  \
        }                                                                 \
    }

static int64_t mac_hvx(const int16_t *src, uint32_t len, uint32_t reps) {
    const uint32_t n = len / 128;
    HVX_Vector acc0 = __builtin_HEXAGON_V6_vd0_128B();
    HVX_Vector acc1 = acc0, acc2 = acc0, acc3 = acc0;
    if (((uintptr_t)src & 127) == 0) {
        MAC_BODY(HVX_Vector)
    } else {
        MAC_BODY(HVX_UVector)
    }
    acc0 = __builtin_HEXAGON_V6_vaddw_128B(acc0, acc1);
    acc2 = __builtin_HEXAGON_V6_vaddw_128B(acc2, acc3);
    acc0 = __builtin_HEXAGON_V6_vaddw_128B(acc0, acc2);
    int64_t sum = 0;
    for (int k = 0; k < 32; k++) sum += acc0[k];
    return sum;
}

/* ---- method implementations -------------------------------------------- */
static int do_kernel(uint32_t method, uint32_t sc, dsp_remote_arg *pra) {
    const uint32_t nin = LF_SCALARS_INBUFS(sc), nout = LF_SCALARS_OUTBUFS(sc);
    const int is_mac = method == LFHVX_M_MAC_HVX;
    if (nin != 2 || nout != (is_mac ? 1u : 2u)) return 0x10; /* AEE_EBADPARM-ish */
    if (pra[0].buf.nLen < sizeof(lfhvx_kernel_args)) return 0x11;
    lfhvx_kernel_args args;
    memcpy(&args, pra[0].buf.pv, sizeof(args));
    const uint8_t *src = pra[1].buf.pv;
    if (args.len == 0 || (args.len & 127) || pra[1].buf.nLen < args.len) return 0x12;
    uint8_t *dst = 0;
    dsp_remote_arg *res_arg;
    if (is_mac) {
        res_arg = &pra[2];
    } else {
        dst = pra[2].buf.pv;
        if (pra[2].buf.nLen < args.len) return 0x13;
        res_arg = &pra[3];
    }
    if (res_arg->buf.nLen < sizeof(lfhvx_kernel_result)) return 0x14;

    lfhvx_kernel_result res = {0};
    int use_hvx = method != LFHVX_M_U8ADD_SCALAR;
    if (use_hvx) {
        res.hvx_lock = qurt_hvx_lock ? qurt_hvx_lock(QURT_HVX_MODE_128B) : -1000;
        res.hvx_mode = qurt_hvx_get_mode ? qurt_hvx_get_mode() : -1000;
        if (res.hvx_lock != 0) {
            LOG("lfhvx: qurt_hvx_lock failed %d", res.hvx_lock);
            memcpy(res_arg->buf.pv, &res, sizeof(res));
            return 0x15;
        }
    }
    uint64_t t0 = read_utimer(), c0 = read_pcycle();
    if (method == LFHVX_M_U8ADD_HVX)
        res.checksum = u8add_hvx(src, dst, args.len, args.reps);
    else if (method == LFHVX_M_U8ADD_SCALAR)
        res.checksum = u8add_scalar(src, dst, args.len, args.reps);
    else
        res.checksum = mac_hvx((const int16_t *)src, args.len, args.reps);
    uint64_t c1 = read_pcycle(), t1 = read_utimer();
    if (use_hvx && qurt_hvx_unlock) qurt_hvx_unlock();
    res.pcycles = c1 - c0;
    res.ticks = t1 - t0;
    memcpy(res_arg->buf.pv, &res, sizeof(res));
    return 0;
}


/* ---- multithreaded MAC -------------------------------------------------- */
#define MT_MAX 4
#define MT_STACK (16 * 1024)
static uint8_t g_stacks[MT_MAX][MT_STACK] __attribute__((aligned(128)));

typedef struct {
    const int16_t *src;
    uint32_t len, reps;
    int32_t lock_rc;
    int64_t checksum;
    uint64_t pcycles;
    volatile int done;
} mt_job;

static void mt_worker(void *arg) {
    mt_job *j = arg;
    j->lock_rc = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (j->lock_rc == 0) {
        uint64_t c0 = read_pcycle();
        j->checksum = mac_hvx(j->src, j->len, j->reps);
        j->pcycles = read_pcycle() - c0;
        qurt_hvx_unlock();
    }
    j->done = 1;
}

/* small unsigned divide, so no __hexagon_udivsi3 import is needed */
static uint32_t udiv_small(uint32_t n, uint32_t d) {
    uint32_t q = 0;
    while (n >= d) {
        n -= d;
        q++;
    }
    return q;
}

static int g_attr_variant = -1; /* which attribute defaults the QuRT build accepted */

static int spawn(unsigned int *tid, int idx, unsigned short prio, mt_job *job) {
    for (int variant = g_attr_variant < 0 ? 0 : g_attr_variant; variant < 2; variant++) {
        lf_qurt_thread_attr a;
        memset(&a, 0, sizeof(a));
        a.name[0] = 'l'; a.name[1] = 'f'; a.name[2] = '0' + idx;
        a.priority = prio;
        a.timetest_id = (unsigned short)-2;
        if (variant == 0) { /* Halide defaults */
            a.affinity = 0xff;
            a.bus_priority = 0xff;
        }
        a.stack_size = MT_STACK;
        a.stack_addr = g_stacks[idx];
        int rc = qurt_thread_create(tid, &a, mt_worker, job);
        if (rc == 0) {
            g_attr_variant = variant;
            return 0;
        }
        LOG("lfhvx: qurt_thread_create variant %d rc=%d", variant, rc);
        if (g_attr_variant >= 0) return rc;
    }
    return -1;
}

static int do_mac_mt(uint32_t sc, dsp_remote_arg *pra) {
    if (LF_SCALARS_INBUFS(sc) != 2 || LF_SCALARS_OUTBUFS(sc) != 1) return 0x30;
    if (pra[0].buf.nLen < sizeof(lfhvx_kernel_args) || pra[2].buf.nLen < sizeof(lfhvx_kernel_result))
        return 0x31;
    if (!qurt_thread_create || !qurt_thread_join || !qurt_hvx_lock) return 0x32;
    lfhvx_kernel_args args;
    memcpy(&args, pra[0].buf.pv, sizeof(args));
    uint32_t nt = args.flags;
    if (nt < 1 || nt > MT_MAX || !args.len || (args.len & 127) || pra[1].buf.nLen < args.len)
        return 0x33;
    unsigned short prio = 100;
    if (qurt_thread_get_priority && qurt_thread_get_id) {
        int p = qurt_thread_get_priority(qurt_thread_get_id());
        if (p > 0 && p < 255) prio = (unsigned short)p;
    }
    const int16_t *src = (const int16_t *)pra[1].buf.pv;
    uint32_t nvec = args.len / 128, per = udiv_small(nvec, nt);
    mt_job jobs[MT_MAX];
    unsigned int tids[MT_MAX];
    lfhvx_kernel_result res = {0};
    uint64_t t0 = read_utimer();
    uint32_t started = 0;
    int rc = 0;
    for (uint32_t t = 0; t < nt; t++) {
        uint32_t v0 = t * per, cnt = t == nt - 1 ? nvec - v0 : per;
        memset(&jobs[t], 0, sizeof(jobs[t]));
        jobs[t].src = src + (size_t)v0 * 64;
        jobs[t].len = cnt * 128;
        jobs[t].reps = args.reps;
        rc = spawn(&tids[t], (int)t, prio, &jobs[t]);
        if (rc) break;
        started++;
    }
    for (uint32_t t = 0; t < started; t++) {
        int status = 0;
        qurt_thread_join(tids[t], &status); /* may report "already exited" */
        while (!jobs[t].done) {
        }
    }
    uint64_t t1 = read_utimer();
    if (rc) {
        res.hvx_lock = -2000 + rc;
        memcpy(pra[2].buf.pv, &res, sizeof(res));
        return 0x34;
    }
    for (uint32_t t = 0; t < nt; t++) {
        if (jobs[t].lock_rc) res.hvx_lock = jobs[t].lock_rc;
        res.checksum += jobs[t].checksum;
        if (jobs[t].pcycles > res.pcycles) res.pcycles = jobs[t].pcycles;
    }
    res.ticks = t1 - t0;
    res.hvx_mode = qurt_hvx_get_units ? qurt_hvx_get_units() : 0;
    memcpy(pra[2].buf.pv, &res, sizeof(res));
    LOG("lfhvx: mac_mt nt=%u variant=%d prio=%u ticks=%u", nt, g_attr_variant, prio,
        (unsigned)res.ticks);
    return res.hvx_lock ? 0x35 : 0;
}

/* ---- VTCM probe + bench ------------------------------------------------- */
static int do_vtcm(uint32_t sc, dsp_remote_arg *pra) {
    if (LF_SCALARS_INBUFS(sc) != 2 || LF_SCALARS_OUTBUFS(sc) != 1) return 0x40;
    if (pra[0].buf.nLen < sizeof(lfhvx_kernel_args) || pra[2].buf.nLen < sizeof(lfhvx_vtcm_info))
        return 0x41;
    lfhvx_kernel_args args;
    memcpy(&args, pra[0].buf.pv, sizeof(args));
    if (!args.len || (args.len & 127) || pra[1].buf.nLen < args.len) return 0x42;
    lfhvx_vtcm_info vi;
    memset(&vi, 0, sizeof(vi));
    /* bit 0: HAP_compute_res_* family, bit 1: compute_resource_* family */
    vi.has_attr_init = (HAP_compute_res_attr_init != 0) | (compute_resource_attr_init != 0) << 1;
    vi.has_acquire = (HAP_compute_res_acquire != 0) | (compute_resource_acquire != 0) << 1;
    vi.has_request_vtcm = HAP_request_VTCM != 0;
    vi.has_hvx_units = qurt_hvx_get_units != 0;
    vi.hvx_units = qurt_hvx_get_units ? qurt_hvx_get_units() : -1;
    int (*query)(unsigned int, unsigned int *, void *, unsigned int *, void *) =
        HAP_compute_res_query_VTCM ? HAP_compute_res_query_VTCM : compute_resource_query_VTCM;
    vi.query_rc = query ? query(0, &vi.vtcm_total, 0, 0, 0) : -1000;

    const uint32_t want = 2 * args.len;
    uint8_t *vtcm = 0;
    lf_compute_res_attr attr;
    int fam_hap = HAP_compute_res_attr_init && HAP_compute_res_acquire;
    int (*a_init)(lf_compute_res_attr *) = fam_hap ? HAP_compute_res_attr_init : compute_resource_attr_init;
    int (*a_ser)(lf_compute_res_attr *, unsigned char) =
        fam_hap ? HAP_compute_res_attr_set_serialize : compute_resource_attr_set_serialize;
    int (*a_v1)(lf_compute_res_attr *, unsigned int, unsigned char) =
        fam_hap ? HAP_compute_res_attr_set_vtcm_param : compute_resource_attr_set_vtcm_param;
    int (*a_v2)(lf_compute_res_attr *, unsigned int, unsigned int, unsigned int) =
        fam_hap ? HAP_compute_res_attr_set_vtcm_param_v2 : compute_resource_attr_set_vtcm_param_v2;
    unsigned int (*a_acq)(lf_compute_res_attr *, unsigned int) =
        fam_hap ? HAP_compute_res_acquire : compute_resource_acquire;
    void *(*a_ptr)(lf_compute_res_attr *) =
        fam_hap ? HAP_compute_res_attr_get_vtcm_ptr : compute_resource_attr_get_vtcm_ptr;
    int (*a_ptr2)(lf_compute_res_attr *, void **, unsigned int *) =
        fam_hap ? HAP_compute_res_attr_get_vtcm_ptr_v2 : compute_resource_attr_get_vtcm_ptr_v2;
    int (*a_rel)(unsigned int) = fam_hap ? HAP_compute_res_release : compute_resource_release;
    if (a_init && a_acq && (a_v1 || a_v2) && (a_ptr || a_ptr2)) {
        memset(&attr, 0, sizeof(attr));
        a_init(&attr);
        if (a_ser) a_ser(&attr, 0);
        vi.set_vtcm_rc = a_v2 ? a_v2(&attr, want, want, want) : a_v1(&attr, want, 1);
        vi.ctx_id = a_acq(&attr, 100000);
        if (vi.ctx_id) {
            void *p = 0;
            unsigned int sz = 0;
            if (a_ptr2) {
                a_ptr2(&attr, &p, &sz);
            } else {
                p = a_ptr(&attr);
                sz = p ? want : 0;
            }
            vtcm = p;
            vi.vtcm_ptr = (uint32_t)(uintptr_t)p;
            vi.vtcm_size = sz;
        }
    }
    int legacy = 0;
    if (!vtcm && HAP_request_VTCM) {
        vtcm = HAP_request_VTCM(want, 1);
        if (!vtcm) vtcm = HAP_request_VTCM(want, 0);
        vi.req_ptr = (uint32_t)(uintptr_t)vtcm;
        if (vtcm) {
            vi.vtcm_ptr = vi.req_ptr;
            vi.vtcm_size = want;
            legacy = 1;
        }
    }
    if (vtcm && vi.vtcm_size >= want && qurt_hvx_lock && qurt_hvx_lock(QURT_HVX_MODE_128B) == 0) {
        uint8_t *vsrc = vtcm, *vdst = vtcm + args.len;
        const HVX_UVector *s = (const HVX_UVector *)pra[1].buf.pv;
        HVX_Vector *d = (HVX_Vector *)vsrc;
        for (uint32_t i = 0; i < args.len / 128; i++) d[i] = s[i];
        vi.bench_len = args.len;
        vi.bench_reps = args.reps;
        uint64_t t0 = read_utimer(), c0 = read_pcycle();
        u8add_hvx(vsrc, vdst, args.len, args.reps);
        uint64_t c1 = read_pcycle(), t1 = read_utimer();
        vi.u8add_pcycles = c1 - c0;
        vi.ticks_u8add = t1 - t0;
        uint32_t sum = 0;
        for (uint32_t i = 0; i < args.len; i++) sum += vdst[i];
        vi.dst_sum = sum;
        t0 = read_utimer();
        c0 = read_pcycle();
        vi.mac_checksum = mac_hvx((const int16_t *)vsrc, args.len, args.reps);
        c1 = read_pcycle();
        t1 = read_utimer();
        vi.mac_pcycles = c1 - c0;
        vi.ticks_mac = t1 - t0;
        qurt_hvx_unlock();
    }
    if (vi.ctx_id && a_rel) vi.release_rc = a_rel(vi.ctx_id);
    if (legacy && HAP_release_VTCM) vi.release_rc = HAP_release_VTCM(vtcm);
    memcpy(pra[2].buf.pv, &vi, sizeof(vi));
    LOG("lfhvx: vtcm ctx=%u ptr=0x%x size=%u", vi.ctx_id, vi.vtcm_ptr, vi.vtcm_size);
    return 0;
}

static int g_open_count;

EXPORT int lfhvx_skel_handle_invoke(uint64_t h, uint32_t sc, dsp_remote_arg *pra) {
    const uint32_t method = LF_SCALARS_METHOD(sc);
    (void)h;
    switch (method) {
    case LFHVX_M_OPEN:
        /* in[0] = int32 len, in[1] = uri; out handle = pra[2].h64 */
        if (LF_SCALARS_INBUFS(sc) != 2 || LF_SCALARS_OUTHANDLES(sc) != 1) return 0x20;
        g_open_count++;
        pra[2].h64 = 0x4c460000u + (uint32_t)g_open_count; /* opaque, non-zero */
        LOG("lfhvx: open #%d", g_open_count);
        return 0;
    case LFHVX_M_CLOSE:
        LOG("lfhvx: close");
        return 0;
    case LFHVX_M_ADD: {
        if (LF_SCALARS_INBUFS(sc) != 1 || LF_SCALARS_OUTBUFS(sc) != 1) return 0x21;
        if (pra[0].buf.nLen < 8 || pra[1].buf.nLen < 4) return 0x22;
        int32_t ab[2];
        memcpy(ab, pra[0].buf.pv, 8);
        int32_t sum = ab[0] + ab[1];
        memcpy(pra[1].buf.pv, &sum, 4);
        return 0;
    }
    case LFHVX_M_U8ADD_HVX:
    case LFHVX_M_U8ADD_SCALAR:
    case LFHVX_M_MAC_HVX:
        return do_kernel(method, sc, pra);
    case LFHVX_M_MAC_MT:
        return do_mac_mt(sc, pra);
    case LFHVX_M_VTCM:
        return do_vtcm(sc, pra);
    case LFHVX_M_INFO: {
        if (LF_SCALARS_OUTBUFS(sc) != 1 || pra[0].buf.nLen < sizeof(lfhvx_info)) return 0x23;
        lfhvx_info info = {0};
        info.pcycle = read_pcycle();
        info.utimer = read_utimer();
        info.hvx_mode_before = qurt_hvx_get_mode ? (uint32_t)qurt_hvx_get_mode() : 0xffffffffu;
        info.hvx_lock_rc = qurt_hvx_lock ? qurt_hvx_lock(QURT_HVX_MODE_128B) : -1000;
        info.hvx_mode_locked = qurt_hvx_get_mode ? (uint32_t)qurt_hvx_get_mode() : 0xffffffffu;
        if (info.hvx_lock_rc == 0 && qurt_hvx_unlock) qurt_hvx_unlock();
        info.skel_version = 1;
        memcpy(pra[0].buf.pv, &info, sizeof(info));
        return 0;
    }
    default:
        return 0x2f;
    }
}
