// SPDX-License-Identifier: MIT
package com.librestatic.lfhvxtest;

import android.app.Activity;
import android.os.Bundle;
import android.widget.ScrollView;
import android.widget.TextView;

/** Runs the lfhvx cDSP tests on launch; results go to logcat (tag LFHVX) and the screen. */
public class MainActivity extends Activity {
    static {
        System.loadLibrary("lfhvx_jni");
    }

    /** mode: 0 = skel from nativeLibraryDir, 1 = copied to filesDir, 2 = no unsigned request. */
    private static native String run(String nativeLibraryDir, String filesDir, int mode);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        final TextView text = new TextView(this);
        text.setTextIsSelectable(true);
        text.setTypeface(android.graphics.Typeface.MONOSPACE);
        text.setTextSize(10);
        text.setPadding(16, 96, 16, 96);
        text.setText("running...");
        ScrollView scroll = new ScrollView(this);
        scroll.addView(text);
        setContentView(scroll);
        final int mode = getIntent().getIntExtra("mode", 0);
        final String nativeDir = getApplicationInfo().nativeLibraryDir;
        final String filesDir = getFilesDir().getAbsolutePath();
        new Thread(() -> {
            String result;
            try {
                result = run(nativeDir, filesDir, mode);
            } catch (Throwable t) {
                result = "exception: " + t;
                android.util.Log.e("LFHVX", "run failed", t);
            }
            final String r = result;
            runOnUiThread(() -> text.setText(r));
        }, "lfhvx").start();
    }
}
