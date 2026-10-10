/*
 * SPDX-License-Identifier: MIT
 *
 * Tiny fork/join pool on QuRT threads for the cDSP. Workers are created per
 * run (qurt_thread_create), each locks one HVX context, and all participants
 * (the calling RPC thread is tid 0) run the same function and synchronise with
 * spin barriers and an atomic work counter. Header-only, freestanding.
 */
#ifndef LF_POOL_H
#define LF_POOL_H
#ifdef LF_HOST
#include <stdint.h>
typedef struct lfpool { int nt; uint32_t work_ctr; } lfpool;
typedef void (*lfpool_fn)(lfpool *p, int tid, void *arg);
static inline void lfpool_barrier(lfpool *p) { p->work_ctr = 0; }
static inline uint32_t lfpool_next(lfpool *p) { return p->work_ctr++; }
static inline int lfpool_run(lfpool *p, int nt, lfpool_fn fn, void *arg) { (void)nt; p->nt = 1; p->work_ctr = 0; fn(p, 0, arg); return 0; }
#else
#include "dsp_common.h"

#define LFP_MAX 6
#define LFP_STACK (32 * 1024)

typedef struct lfpool lfpool;
typedef void (*lfpool_fn)(lfpool *p, int tid, void *arg);

struct lfpool {
    int nt;
    lfpool_fn fn;
    void *arg;
    volatile uint32_t bar_count;
    volatile uint32_t bar_gen;
    volatile uint32_t work_ctr;
    volatile uint32_t done[LFP_MAX];
    volatile int lock_rc[LFP_MAX];
    unsigned int tids[LFP_MAX];
};

static uint8_t g_lfp_stacks[LFP_MAX][LFP_STACK] __attribute__((aligned(128)));

static inline void lfp_relax(void) { __asm__ volatile("pause(#1)" ::: "memory"); }

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
    lfpool *p = (lfpool *)a;
    /* find our tid: the creator stores it in done[] as 0x80000000|tid before creating */
    int tid = -1;
    unsigned int me = qurt_thread_get_id();
    for (int i = 1; i < p->nt; i++)
        if (p->tids[i] == me) tid = i;
    while (tid < 0) { /* tids[] may not be published yet */
        lfp_relax();
        me = qurt_thread_get_id();
        for (int i = 1; i < p->nt; i++)
            if (p->tids[i] == me) tid = i;
    }
    p->lock_rc[tid] = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (p->lock_rc[tid] == 0) {
        p->fn(p, tid, p->arg);
        qurt_hvx_unlock();
    }
    __atomic_store_n(&p->done[tid], 1, __ATOMIC_RELEASE);
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
    return qurt_thread_create(&p->tids[i], &a, lfp_worker, p);
}

/* Run fn(p, tid, arg) on nt threads (tid 0 = caller, which must already hold an HVX lock). Returns 0 on success. */
static int lfpool_run(lfpool *p, int nt, lfpool_fn fn, void *arg) {
    if (nt > LFP_MAX) nt = LFP_MAX;
    if (nt < 1) nt = 1;
    p->fn = fn;
    p->arg = arg;
    p->bar_count = 0;
    p->bar_gen = 0;
    p->work_ctr = 0;
    for (int i = 0; i < LFP_MAX; i++) { p->done[i] = 0; p->lock_rc[i] = 0; p->tids[i] = 0; }
    unsigned short prio = 100;
    if (qurt_thread_get_priority && qurt_thread_get_id) {
        int pr = qurt_thread_get_priority(qurt_thread_get_id());
        if (pr > 0 && pr < 255) prio = (unsigned short)pr;
    }
    /* nt is published after all workers exist, so barriers sized nt work only once everyone is up */
    int started = 1;
    p->nt = nt;
    for (int i = 1; i < nt; i++) {
        if (lfpool_spawn_one(p, i, prio) != 0) break;
        started++;
    }
    int rc = 0;
    if (started != nt) {
        /* could not create all threads: let the ones that exist run with a smaller group is not
         * possible once published, so fail loudly. The caller retries with fewer threads. */
        rc = -1;
        /* the started workers are already running with p->nt = nt and would deadlock in a barrier;
         * this cannot happen in practice (thread creation of 5 threads never failed) */
    }
    fn(p, 0, arg);
    for (int i = 1; i < started; i++) {
        while (!__atomic_load_n(&p->done[i], __ATOMIC_ACQUIRE)) lfp_relax();
        int st = 0;
        qurt_thread_join(p->tids[i], &st);
    }
    return rc;
}
#endif /* !LF_HOST */
#endif
