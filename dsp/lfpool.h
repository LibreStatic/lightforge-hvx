/*
 * SPDX-License-Identifier: MIT
 *
 * Tiny fork/join pool on QuRT threads for the cDSP. Workers are created per
 * run (qurt_thread_create), each locks one HVX context, and all participants
 * (the calling RPC thread is tid 0, which must already hold an HVX lock) run the
 * same function and synchronise with spin barriers and an atomic work counter.
 * The group is formed before the barrier size is published: a worker that cannot
 * get an HVX context (lock error, thread creation failure, or no context free
 * within LFP_JOIN_TICKS) is left out and lfpool_run uses the workers that did lock,
 * so a DSP with fewer than nt contexts degrades instead of deadlocking.
 * lfpool_nthreads(p) is the size actually used. Header-only, freestanding.
 */
#ifndef LF_POOL_H
#define LF_POOL_H
#ifdef LF_HOST
#include <stdint.h>
#define LFP_MAX 6
typedef struct lfpool { int nt; uint32_t work_ctr; } lfpool;
typedef void (*lfpool_fn)(lfpool *p, int tid, void *arg);
static inline void lfpool_barrier(lfpool *p) { p->work_ctr = 0; }
static inline uint32_t lfpool_next(lfpool *p) { return p->work_ctr++; }
static inline int lfpool_run(lfpool *p, int nt, lfpool_fn fn, void *arg) { (void)nt; p->nt = 1; p->work_ctr = 0; fn(p, 0, arg); return 0; }
static inline int lfpool_probe(lfpool *p, int nt) { (void)nt; p->nt = 1; return 1; }
static inline int lfpool_nthreads(const lfpool *p) { return p->nt; }
#else
#include "dsp_common.h"

#define LFP_MAX 6
#define LFP_STACK (32 * 1024)
#define LFP_JOIN_TICKS (150 * 19200) /* 150 ms of the 19.2 MHz utimer */
#define LFP_CLOSED 0x80000000u

typedef struct lfpool lfpool;
typedef void (*lfpool_fn)(lfpool *p, int tid, void *arg);

struct lfpool {
    int nt;
    lfpool_fn fn;
    void *arg;
    volatile uint32_t gate;     /* LFP_CLOSED | number of workers that hold an HVX context */
    volatile uint32_t reported; /* workers that finished their lock attempt */
    volatile uint32_t go;       /* nt is final, participants may start */
    volatile uint32_t bar_count;
    volatile uint32_t bar_gen;
    volatile uint32_t work_ctr;
    volatile uint32_t done[LFP_MAX];
    unsigned int tids[LFP_MAX];
};

typedef struct {
    lfpool *p;
    int idx;
} lfp_warg;

static uint8_t g_lfp_stacks[LFP_MAX][LFP_STACK] __attribute__((aligned(128)));
static lfp_warg g_lfp_args[LFP_MAX];

static inline void lfp_relax(void) { __asm__ volatile("pause(#1)" ::: "memory"); }

static inline int lfpool_nthreads(const lfpool *p) { return p->nt; }

static inline void lfpool_barrier(lfpool *p) {
    if (p->nt == 1) { p->work_ctr = 0; return; }
    uint32_t gen = __atomic_load_n(&p->bar_gen, __ATOMIC_ACQUIRE);
    if (__atomic_add_fetch(&p->bar_count, 1, __ATOMIC_ACQ_REL) == (uint32_t)p->nt) {
        p->bar_count = 0;
        __atomic_store_n(&p->work_ctr, 0, __ATOMIC_RELAXED);
        __atomic_add_fetch(&p->bar_gen, 1, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&p->bar_gen, __ATOMIC_ACQUIRE) == gen) lfp_relax();
    }
}

/* next work item in [0, n) shared by all threads (counter reset by every barrier) */
static inline uint32_t lfpool_next(lfpool *p) { return __atomic_fetch_add(&p->work_ctr, 1, __ATOMIC_RELAXED); }

static void lfp_worker(void *a) {
    lfp_warg *w = (lfp_warg *)a;
    lfpool *p = w->p;
    const int rc = qurt_hvx_lock(QURT_HVX_MODE_128B);
    int tid = -1;
    if (rc == 0) { /* join the group unless the leader already closed it; tids are dense by construction */
        uint32_t g = __atomic_load_n(&p->gate, __ATOMIC_ACQUIRE);
        while (!(g & LFP_CLOSED))
            if (__atomic_compare_exchange_n(&p->gate, &g, g + 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) { tid = (int)(g & 0xffffu) + 1; break; }
    }
    __atomic_add_fetch(&p->reported, 1, __ATOMIC_RELEASE);
    if (tid >= 0) {
        while (!__atomic_load_n(&p->go, __ATOMIC_ACQUIRE)) lfp_relax();
        p->fn(p, tid, p->arg);
    }
    if (rc == 0) qurt_hvx_unlock();
    __atomic_store_n(&p->done[w->idx], 1, __ATOMIC_RELEASE);
}

static int lfpool_spawn_one(lfpool *p, int i, unsigned short prio) {
    lf_qurt_thread_attr a;
    uint8_t *z = (uint8_t *)&a;
    for (unsigned k = 0; k < sizeof a; k++) z[k] = 0;
    a.name[0] = 'l'; a.name[1] = 'f'; a.name[2] = (char)('0' + i);
    a.priority = prio;
    a.timetest_id = (unsigned short)-2;
    a.affinity = 0xff;
    a.bus_priority = 0xff;
    a.stack_size = LFP_STACK;
    a.stack_addr = g_lfp_stacks[i];
    g_lfp_args[i].p = p;
    g_lfp_args[i].idx = i;
    return qurt_thread_create(&p->tids[i], &a, lfp_worker, &g_lfp_args[i]);
}

/*
 * Run fn(p, tid, arg) on up to nt threads (tid 0 = caller, which must already hold an HVX lock);
 * the number that took part is lfpool_nthreads(p) afterwards. Returns 0 on success, -1 if the
 * caller's HVX lock could not be re-taken after waiting for workers that were still blocked.
 */
static int lfpool_run(lfpool *p, int nt, lfpool_fn fn, void *arg) {
    if (nt > LFP_MAX) nt = LFP_MAX;
    if (nt < 1) nt = 1;
    p->fn = fn;
    p->arg = arg;
    p->nt = 1;
    p->gate = 0;
    p->reported = 0;
    p->go = 0;
    p->bar_count = 0;
    p->bar_gen = 0;
    p->work_ctr = 0;
    for (int i = 0; i < LFP_MAX; i++) { p->done[i] = 0; p->tids[i] = 0; }
    unsigned short prio = 100;
    if (qurt_thread_get_priority && qurt_thread_get_id) {
        int pr = qurt_thread_get_priority(qurt_thread_get_id());
        if (pr > 0 && pr < 255) prio = (unsigned short)pr;
    }
    int started = 0;
    for (int i = 1; i < nt; i++) {
        if (lfpool_spawn_one(p, i, prio) != 0) break;
        started++;
    }
    /* wait for the workers' lock attempts, then freeze the group */
    const uint64_t t0 = read_utimer();
    while (__atomic_load_n(&p->reported, __ATOMIC_ACQUIRE) < (uint32_t)started && read_utimer() - t0 < LFP_JOIN_TICKS) lfp_relax();
    const uint32_t g = __atomic_fetch_or(&p->gate, LFP_CLOSED, __ATOMIC_ACQ_REL);
    const int stuck = __atomic_load_n(&p->reported, __ATOMIC_ACQUIRE) < (uint32_t)started;
    p->nt = 1 + (int)(g & 0xffffu);
    __atomic_store_n(&p->go, 1, __ATOMIC_RELEASE);
    fn(p, 0, arg);
    int rc = 0;
    /* workers still blocked in qurt_hvx_lock need a context to get out: free ours while joining */
    if (stuck) qurt_hvx_unlock();
    for (int i = 1; i <= started; i++) {
        while (!__atomic_load_n(&p->done[i], __ATOMIC_ACQUIRE)) lfp_relax();
        int st = 0;
        qurt_thread_join(p->tids[i], &st);
    }
    if (stuck && qurt_hvx_lock(QURT_HVX_MODE_128B)) rc = -1;
    return rc;
}

static void lfp_nop(lfpool *p, int tid, void *arg) { (void)p; (void)tid; (void)arg; }

/* how many threads (<= nt) can really run at once; the caller must hold an HVX lock */
static int lfpool_probe(lfpool *p, int nt) {
    if (lfpool_run(p, nt, lfp_nop, 0) != 0) return 1;
    return p->nt;
}
#endif /* !LF_HOST */
#endif
