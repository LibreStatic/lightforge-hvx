# SPDX-License-Identifier: BSD-3-Clause
#
# Open-source-only build: upstream clang/lld for the Hexagon skel,
# Android NDK clang for the arm64 driver. No Hexagon SDK.

NDK        ?= $(HOME)/Android/Sdk/ndk/28.2.13676358
API        ?= 31
HEX_CPU    ?= hexagonv73
HEX_CC     ?= clang
HEX_LD     ?= ld.lld
ANDROID_CC ?= $(NDK)/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android$(API)-clang

ADB        ?= adb -s 192.168.0.250:40989
DEVDIR     ?= /data/local/tmp/lightforge-hvx

B := build

HEX_CFLAGS := --target=hexagon -mcpu=$(HEX_CPU) -mhvx -mhvx-length=128b -O2 \
              -fPIC -G0 -ffreestanding -fno-builtin -fno-stack-protector -nostdlib \
              -fvisibility=hidden -Wall -Wextra -Wno-unused-parameter

all: $(B)/liblfhvx_skel.so $(B)/lfhvx_bench

$(B):
	mkdir -p $(B)

$(B)/lfhvx_skel.o: dsp/lfhvx_skel.c inc/lfhvx_proto.h | $(B)
	$(HEX_CC) $(HEX_CFLAGS) -c $< -o $@

# Undefined symbols (qurt_*, HAP_*) are resolved by the DSP loader at dlopen time.
# The QuRT loader mprotect()s every PT_LOAD segment and fails ("Cannot protect
# segment N") unless each segment starts on a 4 KiB page boundary in memory and
# in the file, so: 4 KiB pages, R+RX merged, no RELRO split, padded segments.
HEX_LDFLAGS := -shared -soname liblfhvx_skel.so --hash-style=sysv \
               -z max-page-size=4096 -z norelro --no-rosegment -z separate-loadable-segments

$(B)/liblfhvx_skel.so: $(B)/lfhvx_skel.o
	$(HEX_LD) $(HEX_LDFLAGS) -o $@ $<

$(B)/lfhvx_bench: apps/lfhvx_bench.c inc/lfhvx_proto.h | $(B)
	$(ANDROID_CC) -O2 -Wall -Wextra -o $@ $< -ldl

# ---- plain-C reference (host x86 and arm64) -----------------------------------
HOST_CC ?= cc
PYTHON  ?= $(HOME)/.cache/claude-tmp/rife-npu/.venv/bin/python
GOLDEN_PAIRS := pair000_320x192 pair002_320x192 pair003_320x192 pair011_320x192

$(B)/weights/rife46_f32.bin $(B)/weights/rife46_hvx.bin tests/golden/manifest.txt: tools/export_weights.py
	$(PYTHON) tools/export_weights.py

weights: $(B)/weights/rife46_f32.bin

$(B)/test_ref: ref/test_ref.c ref/lfrife_ref.c ref/lfrife_ref.h tests/golden_util.h inc/lfrife_weights.h | $(B)
	$(HOST_CC) -O3 -std=c11 -Wall -Wextra -o $@ ref/test_ref.c ref/lfrife_ref.c -lm

$(B)/test_ref_arm64: ref/test_ref.c ref/lfrife_ref.c ref/lfrife_ref.h tests/golden_util.h inc/lfrife_weights.h | $(B)
	$(ANDROID_CC) -O3 -std=c11 -Wall -Wextra -o $@ ref/test_ref.c ref/lfrife_ref.c -lm

test-host: $(B)/test_ref $(B)/weights/rife46_f32.bin
	$(B)/test_ref $(B)/weights/rife46_f32.bin tests/golden $(GOLDEN_PAIRS) --max-err 2e-3

push: all
	$(ADB) shell mkdir -p $(DEVDIR)
	$(ADB) push $(B)/liblfhvx_skel.so $(B)/lfhvx_bench $(DEVDIR)/
	$(ADB) shell chmod 755 $(DEVDIR)/lfhvx_bench

run: push
	$(ADB) shell $(DEVDIR)/lfhvx_bench

clean-device:
	$(ADB) shell rm -rf $(DEVDIR)

clean:
	rm -rf $(B)

.PHONY: weights test-host all push run clean clean-device
