# Android

An Android client app for PXA Control is in testing and is not part of the 4.0 downloads.

It is a remote window onto a PXA Control that is already running on your network: a WebView wrapper with a
rig list (an address and the sign-in for each machine). The phone does not run a model, and there is no
on-device engine. The app has no voice input yet.

Until it ships, open your Control's address in the phone's browser and bookmark it. Control listens on your
network only when you start it with `--lan` (`./pxa --lan`), behind its access token; see
[PXA Control 4.0](user/CONTROL-GUIDE-4.0.md) and [Install PXA](install.md#android).
