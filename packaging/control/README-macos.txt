PXA Control for macOS (@ARCH_TITLE@)  -  @TAG@
=============================================

PXA Control is the control page for PXA: your models folder, Hugging Face downloads, the PXA Store, settings,
the chat page and the access token. This is the Mac client. There is no PXA engine on macOS (no NVIDIA GPU, no
Metal engine), so it cannot start a model server on the Mac. To drive a GPU machine, run Control on that machine with
--lan and open it from the Mac (see "Use it with a GPU machine").

@ARCH_NEEDS@

Start it
--------
1. Unzip. Open the folder.
2. Double-click "Start PXA Control.command" (opens a Terminal window and starts Control; your browser opens signed
   in). Close the window or press Ctrl+C to stop it.
   Or from a Terminal:  ./pxa-control          (add --lan to also serve the page to your network)
The first start unpacks once into ~/Library/Caches/PXA (about 30 seconds); later starts take about a second.

The first time: "cannot be opened because the developer cannot be verified"
-----------------------------------------------------------------------
This download is not signed with an Apple Developer ID and is not notarized, so macOS Gatekeeper asks before it runs
anything from the internet. Nothing is wrong with the file; verify it first by comparing its SHA-256 with the one on
the download page:   shasum -a 256 <the .zip>
Then allow it once, by either way:
  a) Terminal (works on every macOS version), in the unzipped folder:
         xattr -dr com.apple.quarantine .
     then double-click "Start PXA Control.command" again.
  b) Without Terminal: try to open "Start PXA Control.command" once (macOS refuses), open System Settings ->
     Privacy & Security, scroll to Security, click "Open Anyway" next to the message about the file, confirm
     with your password. On macOS 13 and 14 you can also Control-click the file, choose Open, then Open.
You do this once per download. To undo, delete the folder (and ~/Library/Caches/PXA/control-* if you like).

Use it with a GPU machine
-------------------------
On the GPU machine:   pxa-control --lan --port 7777      (it prints the address and the access token)
On the Mac, in a browser:   http://<that machine's address>:7777/?token=<the token>
Anyone on your network can reach that port when --lan is on, but only with the token: keep the token private and
do not forward the port to the internet.
Your own copy of Control on the Mac also keeps its token in ~/.config/pxa/token.

Limits of this build
--------------------
@ARCH_LIMIT@
- Hugging Face downloads and the PXA Store need an internet connection; the models you download here are only
  useful on a machine that can run them.
