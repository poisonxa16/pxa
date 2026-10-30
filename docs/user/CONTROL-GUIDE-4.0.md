# PXA Control 4.0

PXA Control is the page for your own machine. It finds your cards, starts a model, and gives you chat, benchmarks, and live charts. One Control runs per person. Starting it again opens the one that is already running.

This guide is for someone running Control at home. It follows the pages in 4.0.

Three short pages sit beside it:

- [Beta channel](BETA-4.0.md) — Valued Supporters can try new builds first
- [Report a problem](REPORT-A-PROBLEM.md) — what to send
- [Model quick start](MODEL-QUICKSTART-4.0.md) — one section per model family

## Install and launch

Every package is checked against the `sha256` on the release page before you use it: `sha256sum -c <file>.sha256` on Linux, `shasum -a 256 <file>` on a Mac, `Get-FileHash` on Windows.

**Linux (tarball).** Unpack the tarball into a folder of its own, `cd` into it and run `./pxa`. Control opens in your browser, already signed in. It needs glibc 2.35 or newer and an NVIDIA driver; the package brings its own CUDA runtime, so no toolkit is needed. `./pxa --doctor` checks the cards, the driver and the CPU and starts nothing. Python is not needed for Control itself. The engine is in the same tarball, and Control finds it. A Linux `.run` file (`pxa-<version>-linux-x86_64.run`), when you are handed one, installs the same Control: double-click it, or `chmod +x` it and start it. It unpacks once into `~/.local/share/pxa/<version>`, adds a **PXA Control** entry to your applications menu and opens your browser. Later starts skip the unpack.

**Docker.** `docker pull ghcr.io/poisonxa16/pxa:v4.0` gives the engine image (the same binaries as the tarball, run as a normal user, answering a health check on port 8080). Control in the container is opt-in: add `-e PXA_CONTROL=1 -p 7777:7777`. The tag `latest` moves only with the public release; `v4.0` stays fixed. The full command is in [Install PXA](../install.md).

**Windows.** Two files: `pxa-control-win64.exe` (PXA Control) and `pxa-engine-win64-cuda12.zip` (the engine with the CUDA 12 runtime). Put the exe in a folder of its own, unzip the engine next to it and rename the unzipped folder to `engine`, so `engine\bin\llama-server.exe` exists. Control finds that folder. PXQN files need `ggml-pxqn.dll`, published next to the zip, copied into `engine\bin` next to `ggml.dll`. Without it, PXQN files are refused in one message and PXQ files still run. The files are not code-signed. The first time you open one, Windows SmartScreen shows "Windows protected your PC": choose **More info**, then **Run anyway**. It asks once per file. An NVIDIA driver is required, and a check on more Windows machines is still open.

**Mac.** The Mac download is PXA Control only: there is no engine on macOS (no NVIDIA GPU). It keeps a models folder, downloads models, opens the Store and chats, and it opens a GPU machine's Control over your network: on the GPU machine start Control with `--lan`, then open that address on the Mac. It runs natively on Intel Macs and on Apple silicon Macs (one universal download); Rosetta is not needed. Apple silicon has been built and checked but not yet run on a real Apple silicon Mac; Intel was tried on macOS 13. The download is not signed with an Apple Developer ID and is not notarized, so macOS asks once. Unzip it, double-click **Start PXA Control.command**, and when macOS says it cannot verify the developer, allow it either way: in Terminal, in the unzipped folder, run `xattr -dr com.apple.quarantine .` and double-click the file again; or open System Settings, **Privacy & Security**, scroll to Security and press **Open Anyway** next to the message, then confirm with your password. (On macOS 13 and 14, Control-click, **Open**, **Open** also works.) A Terminal window opens, Control starts and your browser opens signed in. Close the window to stop it. The first launch unpacks the app (about 30 seconds on a slow Mac); later launches take about a second.

**Android.** An Android client app is in testing and is not part of the 4.0 downloads. It is a remote window onto a Control that is already running on your network, not an engine: the phone does not run a model. Until it ships, open your Control's address in the phone's browser and bookmark it.

**In the browser.** On this computer, Chrome or Edge can install Control as an app from localhost. On a plain network address, use a bookmark.

No flag is required. These are optional:

| Flag | What it does |
|---|---|
| `--port N` | Listen on port N. The default is the first free port from 7777. |
| `--lan` | Also listen on your network. Other devices need the sign-in address. |
| `--no-browser` | Do not open a browser. |
| `--no-tray` | No tray icon. |
| `--models-dir DIR` | A folder of model files. You can pass it more than once. It is remembered. |
| `--version` | Print the version and exit. |

With no model folder yet, the first run uses `~/PXA/models` and creates it. By default Control answers only on this computer. A tray icon, when the desktop has one, has Open and Quit. With no tray, leave the terminal open, or stop it with Ctrl-C.

Do not paste a sign-in address into chat or a bug report.

## The sidebar

The menu stays put while the page scrolls. The header names the servers that are up.

| Group | Pages |
|---|---|
| Run | Home, Chat, Launch, Servers, Models |
| Make and tune | Encode, Benchmarks, GPU profiles |
| Watch | Live, Rig |
| Extend | Plugins, Store |

The Help button (top right) is Report a problem, the PXA Network Discord, **Share your rig's speed**, Support PXA on Ko-fi, the theme, and Update.

## Home

Home is the first page. It waits until it knows your models and your servers, so a machine that is already serving does not flash a first-run screen.

**Nothing installed yet.** The heading is "Run your first model". Three steps are on the page: get a model, pick cards and start, then chat. Download a ready-made PXA file, or type the folder that holds your files and press **Use this folder**. A download resumes if it stops, and it is kept only when its checksum matches the published one. Supporter files show **Unlock**, which opens the Store. **See every published model** opens Models. Which family to pick is the [model quick start](MODEL-QUICKSTART-4.0.md).

**Models, nothing serving.** The heading is "Run a model". Pick a model and press **Start**. The cards are the ones you used last, or a set the fit check says can hold the file. The fit line reads **fits on these cards**, **fits with system RAM** (part of a mixture-of-experts model stays in the computer's memory), **tight fit**, or **does not fit**. **Change cards or settings** opens Launch. If Start is refused, the reason is on this page, and **See why** opens Launch. Launch has the full plan.

**Loading.** The page says the server is getting ready. A load often takes a minute or two. **Log** shows each step.

**Serving.** The heading is "Your model is serving" when one server is up and healthy. Each running server has **Chat**, **Benchmarks**, **Log**, and **Stop**. The line under the name shows the port, the cards, and decode speed when the server has reported one. **Run another model** starts a second server. **All servers** opens Servers.

## Models

The page reads every `.gguf` file under the folders you name, including names that end in `.GGUF`.

**Model folders.** This card is at the top. Type a path, press **Add folder**, and press **Rescan** after you copy files in. The list is remembered on this machine. The page shows the settings file it uses. The first scan of a large folder on a spinning disk can take about half a minute.

**Library.** One row per file: codec, type, parameter count, size, family, whether it can think, and **Fits on**. Fits on is a memory check of the weights plus a cache estimate for one 4,096-token slot, on cards that are idle. The badges are **fits**, **tight**, and **no**. A set that only fits by keeping part of the model in system memory is marked **+ RAM**. Click a row to launch that file.

**Available from PXA.** Published PXA files, with a download into the first folder above. The list is remembered on this machine. Beside **Check PXA models** you see when it was last read, such as "checked 40 min ago". Control reads the network again on its own only when that list is older than a day, or when you have no model yet. Press **Check PXA models** to read it now. A download picks up where it stopped, checks the published checksum, fetches every part of a split file, and places the expert-counts file next to the model. The model file is not changed apart from that.

Supporter models need a Hugging Face key. Open **Supporter models: your Hugging Face key**, paste the key, and press **Save key**. The key stays on this machine and is sent only to Hugging Face. The page never shows it back. **Forget** removes it.

**Expert map.** For a mixture-of-experts model that does not fit on the card. **Rebuild** starts from the map that shipped with the model and stops when the speed stops rising. The fastest map is kept. The model file is not changed. **Reset learned** clears learned sessions.

## Launch

Launch is the full start page. Pick a saved server, or press **New**. Three steps:

1. **Choose your cards.**
2. **Choose a model.** The fit line uses the same check as Models. Thinking can be auto, on, or off, with a budget.
3. **Start serving.** **Stop** is beside Start. The note under the buttons says the rest is automatic.

**Show plan** prints the command and does not start anything. **Copy** puts it on the clipboard. **Restart** stops and starts the selected server.

**Advanced settings** is closed until you open it. Context `0` is automatic, and **What would auto pick?** shows that number. Parallel slots, KV cache type, and port work the same way: empty or `0` means automatic. KV cache types run from f16 down to q4_0. Split mode is auto, layer, or tensor. Flash attention is auto, on, or off. **MTP** turns speculation on when the file has a draft block. **Serve on the network** listens beyond this computer. Presets save the current form under a name you choose.

The log on the right follows the server. Search it, tick **problems only**, or **Download** it.

A card older than Pascal is refused with a clear message. Settings you already put in the environment of the shell or the service are kept, and the log says which ones.

## Servers

Every model server on this machine is here: ones Control started, ones you started in a terminal, and containers or processes started some other way. Each card shows state, port, cards, memory, and speed. Actions are Start, Stop, Restart, Log, Edit, Chat, Benchmarks, and Live, for the servers Control is allowed to control.

A server found on the machine is watched only, until you **Adopt** it. Adopt asks for a name. **Allow Stop / Start / Restart** is off unless you tick it. Each of those actions still asks you to type the container's name. Stopping a server you started in a terminal asks you to type `stop`.

**New server** opens Launch. The strip under the cards shows which server holds how much of each card.

## Live

Live draws the cards and the servers as charts. A sample arrives about every 2 seconds. The range buttons are 5 min, 15 min, and 1 h. When older history exists, 6 h, 24 h, 7 d, 30 d, and 90 d appear too.

Charts cover decode, prefill, requests, and the hardware: memory, load, temperature, and power for each card, then this machine's memory and CPU. A tile for a server shows whether it is idle, reading the prompt, or writing.

**Running only** sits at the top. It is on by default and remembered on this machine. Stopped servers fold into one line, "3 stopped (show)" (the number is how many are stopped). Press **(show)** to open them. A finished run stays one click away, and its charts stay. Turn the switch off to keep every stopped server as its own tile.

History is recorded while this page is closed. The line under the title says how long samples are kept, and offers a CSV of this range for cards, servers, requests, and this machine. If history is off, the charts keep about the last hour in memory only.

## Benchmarks

This page was called Speed. The heading is now **Benchmarks**.

Pick the server in the menu at the top. The range is 1 h, 24 h, or 7 d. Two charts show decode and prefill, in tokens per second, for each request. A table lists medians by model and prompt size. When a run finishes, the charts and the tables redraw on their own.

**The server strip.** Under the heading, a strip names the server you are measuring: a status dot, the state, the model, the port, and the cards. **Start** and **Stop** act on that server, so you can measure again without leaving the page. **Edit** opens model, MTP, and split mode (auto, layer, or tensor).

- If the server is stopped, **Save** stores those settings. Press **Start** to run it.
- If the server is running, the button says **Save and restart**. Saving stops it and starts it again. If the model does not fit, you land on Launch with the reason.
- While a benchmark is running, Start, Stop, and Save wait until it ends.

**Benchmark my rig.** Set warm-up seconds (default 10, up to 120) and reps (default 3, up to 6), then press **Run benchmark**. The run uses three prompt classes: prose, edit, and long. The history stores a greedy512 value for that run, so you can compare it across builds. While it runs, the button reads **running ...** and the lines of the run appear under it. Afterward you may see the card temperatures and clocks from that run.

**History.** Runs are listed newest first, 20 per page, with **Prev** and **Next**. The search box matches model, cards, sha, and date. Columns are When, Model, Cards, MTP, Split, prose dec t/s, edit dec t/s, long dec t/s, long prefill t/s, and greedy512. A run stored before MTP and Split were recorded shows a dash in those columns. An empty search on an empty list says "no benchmark yet". A search with no hit says "no benchmark matches that search".

After a finished run, **Check the community high-score board** asks the public board about your setup. Nothing is sent until you press **Submit to the board**. You can read the exact text first. An optional name is at most 32 characters. **Not now** skips it.

## Share a benchmark

Help, then **Share your rig's speed**, opens `/share.html`. You can also open that address on this Control.

The page turns one run from your benchmark history into an image: your cards, the model, and the speeds you measured. Nothing is uploaded. You save it or post it yourself.

Pick a run. **Show my board name** adds the name you use on the community board. Then:

- **Save image (PNG)**
- **Copy image**, for a paste into Discord. If the browser cannot copy an image, use Save image.
- **Copy text for Discord**
- **Save SVG**

If you have not run a benchmark, the page says to run one on the Benchmarks tab first.

## Chat

Chat talks to a server that is already up. Start one on Home or Launch if the page says there is no conversation yet. The mode chip is **Chat**, **Assistant**, **Researcher**, or **Coder**. The server chip picks which model answers. Enter sends. Shift+Enter is a new line. Esc stops. Ctrl+K starts a new chat.

Chats and memory stay on this machine. The list on the left searches, renames, deletes, and exports a chat. **Incognito** is not saved and is not searched. Closing the page drops that chat. **Memory** is what the assistant remembers about you. You can turn memory off for one chat.

The gear is **Settings** for this chat: system prompt, sampling, thinking, and tools. **Advanced** adds stop sequences, a JSON schema, and a grammar. **Reset to preset** puts the mode's own settings back. Web search is the magnifier. It can stay on the built-in search, or you can point it at your own search. A key you paste for a search service stays on this computer and is not shown back. **Test search** runs one query with what is on the screen.

**Attach** takes a text file or an image. You can also drop a file on the box, or paste a file from the clipboard.

- Images are PNG, JPEG, GIF, or WebP. Four images is the limit on one message. Over 1 MB is refused. A damaged file, a wrong type, or a file that is not a real image is refused. The reason pops up, and the words you typed come back into the box.
- A text file longer than 100,000 characters is cut. A notice names the full size, says the first 100,000 were added, and says how many were left out. A line in the box records the cut. A file over 200,000 characters is refused as too big to paste. A short file is added whole, with no notice.

Replies render Markdown, code, and math. Edit a message to branch. The **<** and **>** buttons move between versions of that message. On a phone the layout stacks. A long reply stays smooth: the page updates about 25 times a second, and a very long answer does not redraw the whole page on every word.

One device can have 3 chats answering at once. A fourth is told to wait, or to press Stop on one. Across the whole Control, 6 answers at once is the limit, with the same idea: wait a moment. A setting on this machine changes the per-device number, and 0 turns that cap off.

A system prompt, a memory block, or attached chats that are too big for the window are shortened, or refused, and the notice includes the numbers. A message that cannot fit even alone is refused before it is sent, again with the numbers. The reply is saved before the stream is marked finished, so a reload or an export right after an answer still shows it.

A chat file that cannot be read stays in the list, marked broken, with **Export raw** and **Delete**. Opening it shows a plain explanation.

**If Control restarts while you are in a chat.** The page says "Reconnecting to PXA Control..." and keeps trying. It does not give up after a minute.

- When Control is back with the same sign-in, the saved chat opens. A turn that was cut says it was interrupted because Control stopped, and offers **Resend**.
- When the sign-in changed, the page says you are signed out and that this chat is saved. Paste the new address from the terminal (it is the link Control prints) and press **Sign in again**. You return to the same tab. **Reload** tries the current page.
- An incognito chat is not on disk. A restart during that answer says the private chat's answer was lost.

## Encode

Encode turns a Hugging Face model, or a folder or a GGUF file on this computer, into a PXQ or PXQN file for your cards. The steps are Source, Target, Checks, Run, and Done.

**Get the encoder** if none is installed.

- **Free** makes the classic PXQ tiers. No key.
- **Pro** is a supporter feature. It adds the PXQN tiers. **Unlock with supporter coins** opens the Store, where coins from a Ko-fi donation unlock it. You can still support PXA on Ko-fi and ask in the PXA Network Discord with `/encoder`. The bot gives you a key. Paste it and press **Download Pro**. The key stays on this computer.

Pick the model, press **Look it up**, choose the cards and a tier, run the checks, and start. The plan names a tier that fits the GPU memory it sees. A running job can be resumed after you free disk space. The folders of a job that has already started stay with that job. For a different work folder, start a new encode and pick it in step 2, under Advanced.

## The library updater

On Encode, under the encoder, a **Licensed library** row updates the PXQN library without replacing the whole engine. If this build has no updater, the row says so and the library you have keeps working.

The row shows the installed version and the channel (`stable` or `beta`). **Check now** looks for a newer one.

- **Install** asks you to confirm. The copy you have now is kept, so you can roll back.
- **Roll back** returns to the previous library, or to the one this engine shipped.
- **Install library updates without asking** installs a newer library on its own.
- **Beta library channel** is for Valued Supporters. It is early and it may change. If your key is not on that channel, you stay on stable.

Stop running servers before an install. If one is running, the row says so, and Install waits. If the update server cannot be reached, or the key is refused, the installed library keeps running and the row says why.

That row is the library. The engine and Control themselves update from the Help menu. See Settings.

## Settings

Settings are in three places.

**Help menu.** Theme is System, Light, or Dark. The sun-and-moon button on the header switches light and dark for this visit. **Update** opens **Install update**.

- The dialog names the engine channel: Stable or Beta.
- The Beta card is how Valued Supporters install a beta build. Beta access is included and spends no coins. The steps, the wallet, and **Back to stable** are in [Beta channel](BETA-4.0.md).
- A wallet that is not linked to a Valued Supporter sees **The beta channel is for Valued Supporters**. If you are one, run **/beta** in the PXA Discord: the bot sends you personal download links, no coins.
- The card ends with: **v3.1 / beta 1 users: run /beta in Discord.** PXA 3.1 and the first beta have no Beta card; **/beta** works on any version.
- **Join the beta channel** shows for Valued Supporters. The beta installs in its own folder. **Release notes** shows the notes once you are on that channel.
- **Back to stable** returns you to the stable release.
- Installing a build the dialog offers is a slider: slide fully, then hold 5 seconds. Control does not restart a running server. Stop servers first when the dialog says to.
- From a terminal, `pxa-update channel stable` and then `pxa-update apply` does the same return to stable.

**Chat gear.** The Settings drawer for the chat you have open. See Chat.

**Launch, Advanced settings.** Context, slots, cache type, port, split, flash attention, MTP, network serving, presets, and Home Assistant.

**Home Assistant** is off until you turn it on. It publishes the numbers Control already shows, over MQTT, to a broker on your network. The entities show up in Home Assistant ready to use.

**Report a problem** is in the Help menu. You see the exact text. Home paths, host names, addresses, user names, and anything that looks like a key are already removed. You can edit it. Nothing is sent until you press **Send**. An optional contact is a Discord name so a reply can find you. On a beta build, say **beta** and the version, or write the same in Discord **#support**. The full steps are in [Report a problem](REPORT-A-PROBLEM.md).

## GPU profiles

GPU profiles make cards quieter, cooler, or faster. Every change shows what will happen first, and you can undo it. **Benchmark mode** keeps the cards you are measuring free. **New profile**, **Export**, and **Import** manage profiles. Schedules and auto-start are on the same page. **Changes** lists every change, who made it, and when.

## Rig

Rig is a live reading of this machine. Nothing on the page changes a setting. Cards refresh about every 3 seconds. **Live charts** jumps to Live. **Doctor** runs the same check as the launcher's doctor and lets you **Run again**. A machine with no GPU still opens Control. This page is where you see what was found.

## Plugins

Plugins are built by the PXA team from requests on Discord. Each plugin says what it can reach before you turn it on, and you can turn it off later. **Request a plugin** opens Discord.

## Store

The Store is donation-based. Nobody has to donate to use PXA. Control and the PXQ files stay free. A donation on Ko-fi becomes supporter coins. The **PXA Coins** panel shows your balance, history, ways to earn, and the rewards you can unlock. Coins are for extras such as your own PXQN encodes and supporter plugins. The extras are rewards for donating. The beta channel is a Valued Supporter reward and spends no coins. The panel's **Beta access** line says whether you have it, and ends with the same note as the Beta card: v3.1 / beta 1 users: run **/beta** in Discord. See [Beta channel](BETA-4.0.md).

## Troubleshooting

**The browser did not open.** Start **PXA Control** from the menu again, or run the program with no extra flags. A second start opens the one already running and signs the browser in.

**The page says you are signed out.** Control restarted and made a new sign-in. Paste the new address from the terminal and press **Sign in again**. The chat is saved. See Chat for the interrupted turn and **Resend**.

**"Reconnecting to PXA Control..." stays up.** The page is still trying. Leave it. It does not stop trying on its own. When Control is back, the note goes away.

**Start refused.** Read the line on Home. Open Launch and press **Show plan**. The plan is the reason. A model that does not fit needs fewer cards, a smaller file, or a different context. A card older than Pascal is refused in plain words.

**The model is missing.** On Models, add the folder and press **Rescan**. Files that end in `.GGUF` are included. On Home, **Look again** does the same refresh.

**The PXA list looks old.** Press **Check PXA models**. The age is the "checked … ago" line. It refreshes by itself after a day, and when no model is installed yet.

**A download stopped.** It resumes. You can cancel it and start it again. If Hugging Face cannot be reached, Home says so and offers **Try again**, or you can point at a folder you already have.

**Live is a wall of old servers.** **Running only** should be on. Stopped servers are one line, "N stopped (show)". Press **(show)** for a finished run.

**Benchmarks has no rows.** Start a server from the strip, or from Home, then press **Run benchmark**. History is 20 runs to a page. Clear the search box if a filter is hiding them.

**Share says to run a benchmark first.** The card is built from a run on this machine. Run one on Benchmarks, then open `/share.html` again.

**An image was refused.** Use a PNG, JPEG, GIF, or WebP under 1 MB, and at most four on one message. The notice is the reason. Your typed words are put back. Remove the image and send again.

**A pasted file was cut.** Over 100,000 characters, the notice names the size and the box keeps the first 100,000 plus a line about the rest. Over 200,000 characters, the file is refused. Split it, or attach a shorter extract.

**"This device already has 3 chats answering at once."** Wait, or press Stop on one of them. The same applies when Control itself is at 6 answers. The per-device cap is a setting on this machine. Set that value to 0 to turn the cap off.

**A chat vanished from the list, or will not open.** A file that cannot be read is marked broken. It stays in the list with **Export raw** and **Delete**.

**The library will not install.** Stop the running servers. The installed library keeps working until the install finishes. **Roll back** if the new one misbehaves. If the row says this Control has no library updater, this build does not include one.

**Update will not apply.** Stop servers when the dialog says to. Slide the control fully and hold. **Back to stable** leaves the beta channel. The beta channel is for Valued Supporters. See [Beta channel](BETA-4.0.md).

**Encode says no quantizer was found.** Press **Get the encoder** and download Free, unlock Pro with supporter coins in the Store, or paste a supporter key for Pro.

**Something is wrong and you want a person.** Help, **Report a problem**. You see the text before it is sent. Or write in the PXA Network Discord **#support**; on a beta build, say **beta** and the version. For a crash, a wrong answer, or a speed that looks low, name the card, the model file, and what you pressed, or just send the Control report. See [Report a problem](REPORT-A-PROBLEM.md).
