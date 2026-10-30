PXA for Windows x64
===================

Needs Windows 10, Windows 11, or Windows Server 2019 or newer, 64-bit, and an NVIDIA driver new enough for CUDA 12 (release 525 or newer). This package does not include a driver.

What you download
-----------------

- pxa-control-win64.exe
  PXA Control. Put this file in a folder of its own.
- pxa-engine-win64-cuda12.zip
  The engine and the CUDA 12 runtime. Unzip it next to pxa-control-win64.exe and rename the unzipped folder to engine, so this path exists:

      engine\bin\llama-server.exe

  Control finds that folder by itself. The CUDA 12 runtime and the Microsoft C++ runtime are already inside bin\.
- The PXQN library is ggml-pxqn.dll. It is published beside the zip, not inside it. Copy it into engine\bin next to ggml.dll to run a PXQN model. Without that file, PXQ models and other GGUF models still run, and a PXQN model is refused with one message.
- LICENSE-ggml-pxqn.txt
  The terms for ggml-pxqn.dll. The same file is in the engine folder next to LICENSE and NOTICE.

This Windows build of the PXQN library does not include the fast CPU math, locked models, or the expert-cache CPU pool. The Linux build does.

Windows SmartScreen
-------------------

These files are not code-signed. The first time you open one, Windows may show "Windows protected your PC". Choose More info, then Run anyway. Windows asks once per file. Before you run a file, compare it to SHA256SUMS, which is published next to the downloads.

Checked on one NVIDIA GPU: a 4B PXQN model answered a short chat, and decode was 92.4 tokens per second over three runs of 63 tokens. A check across more Windows machines is still open.

Run the engine on its own
--------------------------

    bin\llama-server.exe -m model.gguf -ngl 99

Then open http://127.0.0.1:8080

Community: https://discord.pxanetwork.com
Support the project: https://ko-fi.com/shatteredrealms1
