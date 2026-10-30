package net.pxanetwork.control;

import android.Manifest;
import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.Menu;
import android.view.MenuItem;
import android.webkit.CookieManager;
import android.webkit.PermissionRequest;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.ArrayList;
import java.util.HashSet;

/**
 * PXA Control for Android: one WebView. It starts on the bundled host picker (assets/hosts.html, the same page
 * PXA Control serves at /hosts.html) and opens the PXA Control page of a rig on the network inside the app.
 * There is no engine on the phone; everything runs on the rig.
 *
 * When the rig does not answer (rig off, Control stopped, phone off the LAN) the page load fails: the app then shows the
 * bundled picker in its offline mode (?offline=1&from=<rig>), which names the rig and offers Try again. The app keeps
 * probing the rig (a TCP connect to its port, 3 s, then every 3 s backing off to 30 s, paused while the app is in the
 * background, at once when it comes back) and tells the picker to reopen the rig the moment it answers (a page on
 * file:// cannot fetch an http rig itself).
 *
 * Rigs: an http(s) origin becomes a rig only when the bundled picker opens it (the person chose it). A rig's own pages
 * stay in the app; a link or redirect to any other site opens in the phone's browser, so no other site ever runs inside
 * the app (security review P10 M1: before, every http(s) page stayed in the app and could ask for the microphone).
 *
 * Microphone: a page of the rig may ask for the microphone (voice input). Only audio capture is ever granted, only to a
 * rig origin (never the bundled picker, never another site or a frame of one), and only after Android's own
 * RECORD_AUDIO prompt said yes; camera and every other resource are refused. Browsers expose the microphone to secure pages only (https, or localhost),
 * so on a plain-http LAN address the page cannot ask at all.
 */
public class MainActivity extends Activity {
    private static final String HOSTS = "file:///android_asset/hosts.html";
    private static final int PICK_FILE = 1;
    private static final int ASK_MIC = 2;
    private WebView web;
    private ValueCallback<Uri[]> pending;
    private PermissionRequest micAsk;
    private final Handler ui = new Handler(Looper.getMainLooper());
    private String probeRig;                                     // the rig the offline picker waits for, or null
    private final HashSet<String> rigs = new HashSet<>();        // origins the picker opened: the only sites the app shows
    private String shown;                                        // the page last started in the WebView
    private int probeWait;
    private final Runnable probeTick = new Runnable() { public void run() { probeOnce(); } };

    /** true when a TCP connection to the origin's host and port opens within 3 s */
    static boolean reachable(String origin) {
        Uri u = Uri.parse(origin);
        int port = u.getPort() > 0 ? u.getPort() : ("https".equals(u.getScheme()) ? 443 : 80);
        Socket s = new Socket();
        try {
            s.connect(new InetSocketAddress(u.getHost(), port), 3000);
            return true;
        } catch (IOException | IllegalArgumentException e) {
            return false;
        } finally {
            try { s.close(); } catch (IOException e) { }
        }
    }

    private void startProbe(String o, int firstMs) {
        probeRig = o;
        probeWait = 3000;
        ui.removeCallbacks(probeTick);
        ui.postDelayed(probeTick, firstMs);
    }

    private void probeOnce() {
        final String o = probeRig;
        if (o == null || !onOfflinePicker()) { probeRig = null; return; }
        new Thread(new Runnable() {
            public void run() {
                final boolean up = reachable(o);
                ui.post(new Runnable() {
                    public void run() {
                        if (!o.equals(probeRig) || !onOfflinePicker()) return;
                        if (up) {
                            probeRig = null;
                            web.evaluateJavascript("window.pxaRigBack && window.pxaRigBack()", null);
                        } else {
                            probeWait = Math.min(30000, probeWait * 2);
                            ui.postDelayed(probeTick, probeWait);
                        }
                    }
                });
            }
        }, "rig-probe").start();
    }

    /** the scheme://host[:port] of an http(s) address, or null */
    static String origin(Uri u) {
        if (u == null || u.getHost() == null) return null;
        String sch = u.getScheme() == null ? "" : u.getScheme().toLowerCase();
        if (!sch.equals("http") && !sch.equals("https")) return null;
        return sch + "://" + u.getHost() + (u.getPort() > 0 ? ":" + u.getPort() : "");
    }

    private boolean onOfflinePicker() {
        String u = web.getUrl();
        return u != null && u.startsWith(HOSTS) && u.contains("offline=1");
    }

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        web = new WebView(this);
        setContentView(web);
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);          // the remembered rigs (localStorage) and the page's own state
        s.setAllowFileAccess(true);            // the bundled host picker only
        s.setAllowContentAccess(false);
        s.setMediaPlaybackRequiresUserGesture(true);
        s.setUserAgentString(s.getUserAgentString() + " PXA-Control-Android/4.0-pre");
        CookieManager.getInstance().setAcceptCookie(true);   // Control's sign-in cookie (set from ?token=)
        web.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView v, WebResourceRequest r) {
                Uri u = r.getUrl();
                String sch = u.getScheme() == null ? "" : u.getScheme().toLowerCase();
                if (sch.equals("file")) return !u.toString().startsWith("file:///android_asset/");   // the bundled picker only
                String o = origin(u);
                if (o != null) {
                    if (!r.isForMainFrame()) return false;           // a frame never becomes a rig and never gets the mic
                    String cur = v.getUrl();
                    if (cur != null && cur.startsWith(HOSTS)) rigs.add(o);   // the picker opens the rig the person chose
                    if (rigs.contains(o)) return false;               // a rig's own pages stay in the app
                }
                // any other site (a link, a redirect to another origin) and every other scheme: the phone's browser
                try { startActivity(new Intent(Intent.ACTION_VIEW, u)); } catch (ActivityNotFoundException e) { }
                return true;
            }

            @Override
            public void onPageStarted(WebView v, String url, Bitmap icon) {
                // the picker is the only page that leaves for a new origin by itself (its own code, a navigation without a
                // gesture may skip shouldOverrideUrlLoading): what it opens is the rig the person chose
                String o = origin(Uri.parse(url));
                if (o != null && shown != null && shown.startsWith(HOSTS)) rigs.add(o);
                shown = url;
            }

            @Override
            public void onReceivedError(WebView v, WebResourceRequest r, WebResourceError err) {
                if (!r.isForMainFrame()) return;               // a failed image or script is the page's business
                String o = origin(r.getUrl());
                if (o == null || !rigs.contains(o)) return;      // only a rig gets the offline screen and the probe
                v.stopLoading();
                v.loadUrl(HOSTS + "?offline=1&from=" + Uri.encode(o));
                startProbe(o, 3000);
            }
        });
        web.setWebChromeClient(new WebChromeClient() {
            @Override
            public boolean onShowFileChooser(WebView v, ValueCallback<Uri[]> cb, FileChooserParams p) {
                if (pending != null) pending.onReceiveValue(null);
                pending = cb;
                try {
                    startActivityForResult(p.createIntent(), PICK_FILE);
                } catch (ActivityNotFoundException e) {
                    pending = null;
                    return false;
                }
                return true;
            }

            @Override
            public void onPermissionRequest(final PermissionRequest r) {
                boolean audio = false;
                for (String res : r.getResources()) if (PermissionRequest.RESOURCE_AUDIO_CAPTURE.equals(res)) audio = true;
                String ro = origin(r.getOrigin());
                if (!audio || ro == null || !rigs.contains(ro)) { r.deny(); return; }   // rigs only: never the picker, never another site
                if (Build.VERSION.SDK_INT < 23 || checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                    r.grant(new String[] {PermissionRequest.RESOURCE_AUDIO_CAPTURE});
                    return;
                }
                if (micAsk != null) micAsk.deny();
                micAsk = r;
                requestPermissions(new String[] {Manifest.permission.RECORD_AUDIO}, ASK_MIC);
            }

            @Override
            public void onPermissionRequestCanceled(PermissionRequest r) {
                if (r == micAsk) micAsk = null;
            }
        });
        if (state != null) {
            ArrayList<String> kept = state.getStringArrayList("pxa.rigs");
            if (kept != null) rigs.addAll(kept);
            web.restoreState(state);
        } else web.loadUrl(HOSTS);
    }

    @Override
    protected void onActivityResult(int req, int res, Intent data) {
        if (req == PICK_FILE && pending != null) {
            pending.onReceiveValue(WebChromeClient.FileChooserParams.parseResult(res, data));
            pending = null;
            return;
        }
        super.onActivityResult(req, res, data);
    }

    @Override
    public void onRequestPermissionsResult(int req, String[] perms, int[] res) {
        if (req != ASK_MIC || micAsk == null) return;
        PermissionRequest r = micAsk;
        micAsk = null;
        if (res.length > 0 && res[0] == PackageManager.PERMISSION_GRANTED) r.grant(new String[] {PermissionRequest.RESOURCE_AUDIO_CAPTURE});
        else r.deny();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (probeRig != null) startProbe(probeRig, 500);         // back in the app (maybe on the LAN again): retry at once
    }

    @Override
    protected void onPause() {
        super.onPause();
        ui.removeCallbacks(probeTick);                           // no probing while in the background
    }

    @Override
    protected void onSaveInstanceState(Bundle out) {
        super.onSaveInstanceState(out);
        out.putStringArrayList("pxa.rigs", new ArrayList<>(rigs));
        web.saveState(out);
    }

    @Override
    public void onBackPressed() {
        if (web.canGoBack()) web.goBack();
        else super.onBackPressed();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu m) {
        m.add(0, 1, 0, "Rigs");
        m.add(0, 2, 1, "Reload");
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem it) {
        if (it.getItemId() == 1) { web.loadUrl(HOSTS); return true; }
        if (it.getItemId() == 2) { web.reload(); return true; }
        return super.onOptionsItemSelected(it);
    }
}
