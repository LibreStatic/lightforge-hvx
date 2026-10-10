#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Install the test APK, launch it in each mode and print the LFHVX log.
# usage: android-test/run.sh [modes...]   (default: 0)
set -uo pipefail
ADB=(adb ${ADB_SERIAL:+-s "$ADB_SERIAL"})
PKG=com.librestatic.lfhvxtest
"${ADB[@]}" install -r --no-incremental "$(dirname "$0")/out/lfhvx-test.apk" >/dev/null || exit 1
for m in "${@:-0}"; do
    "${ADB[@]}" shell am force-stop $PKG
    "${ADB[@]}" logcat -c
    "${ADB[@]}" shell am start -n $PKG/.MainActivity --ei mode "$m" >/dev/null
    sleep 2
    echo "== mode $m, process: $("${ADB[@]}" shell ps -AZ | grep $PKG)"
    for _ in $(seq 30); do
        "${ADB[@]}" logcat -d -s LFHVX | grep -q "exit code" && break
        sleep 1
    done
    "${ADB[@]}" logcat -d -s LFHVX | sed -n 's/.*LFHVX *: //p'
    if ! "${ADB[@]}" logcat -d -s LFHVX | grep -q "exit code 0"; then
        echo "== failure context"
        "${ADB[@]}" logcat -d | grep -iE "fastrpc|adsprpc|cdsp|avc: denied" | tail -40
    fi
done
