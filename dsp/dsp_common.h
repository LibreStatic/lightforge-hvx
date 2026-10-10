/*
 * SPDX-License-Identifier: MIT
 *
 * Shared freestanding definitions for the Hexagon skels: DSP-side remote_arg,
 * QuRT/HAP symbols (declared by us, resolved by the user-PD loader), timers.
 * See dsp/lfhvx_skel.c for how each symbol was found.
 */
#ifndef LF_DSP_COMMON_H
#define LF_DSP_COMMON_H
#include <stddef.h>
#include <stdint.h>

#define EXPORT __attribute__((visibility("default")))

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

extern int qurt_hvx_lock(int mode) __attribute__((weak));
extern int qurt_hvx_unlock(void) __attribute__((weak));
extern int qurt_hvx_get_mode(void) __attribute__((weak));
extern int qurt_hvx_get_units(void) __attribute__((weak));
extern void HAP_debug_v2(int level, const char *file, int line, const char *fmt, ...) __attribute__((weak));

typedef struct {
    char name[16];
    unsigned char tcb_partition;
    unsigned char affinity;
    unsigned short priority;
    unsigned char asid;
    unsigned char bus_priority;
    unsigned short timetest_id;
    unsigned int stack_size;
    void *stack_addr;
    unsigned char pad[96];
} lf_qurt_thread_attr;
extern int qurt_thread_create(unsigned int *tid, lf_qurt_thread_attr *attr, void (*entry)(void *), void *arg)
    __attribute__((weak));
extern int qurt_thread_join(unsigned int tid, int *status) __attribute__((weak));
extern unsigned int qurt_thread_get_id(void) __attribute__((weak));
extern int qurt_thread_get_priority(unsigned int tid) __attribute__((weak));

#define LOG(...)                                                                     \
    do {                                                                             \
        if (HAP_debug_v2) HAP_debug_v2(2 /*HIGH*/, __FILE__, __LINE__, __VA_ARGS__); \
    } while (0)

#define QURT_HVX_MODE_128B 1

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
#endif
