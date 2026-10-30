# Changelog

All notable user-facing changes to the PXA engine, PXA Control and the PXQN quantization line.
Newest first. Format follows Keep a Changelog. Each entry says what changed and by how much.

Conventions: "opt-in" = a lever that is OFF unless its environment switch is set; "default" = on without any switch.
Every number is a same-box measurement from the ledger; entries marked NOT-QUIET ran with other work on the box
(load 10-50) and are lower bounds. "a popular community fork" is the same-box rival we measure against.

## [Unreleased]

<!-- New entries land here; the release moves this block into the version section. -->

### Models

- Nemotron support, ported from mainline llama.cpp: nemotron (Nemotron-4 / Minitron / Nemotron-Mini), nemotron_h (Nemotron-H, Nano v2, Nemotron 3 Nano 4B; Mamba-2 hybrid) and nemotron_h_moe (Nemotron 3 Nano 30B-A3B trunk). Runs on CPU and CUDA (sm_60/61/70). PPL matches mainline on the same GGUF (Mini-4B 9.5114 vs 9.5113, Nano-4B 12.3789 vs 12.3737, 30B-A3B 7.99 vs 7.96). Not yet: the MTP head.
- Nemotron-H decode on CUDA is faster by default; Nemotron 3 Nano 4B on one V100, 128-token decode: Q8_0 73.5 -> 135.3 tok/s, Q4_K_M 83.9 -> 181.0 tok/s (NOT-QUIET).
- Nemotron-H prompt processing is faster by default on V100 and P100, with bit-identical outputs (greedy 512-token output unchanged). Nemotron 3 Nano 4B Q8_0 on one V100: pp4096 5259 -> 5688 tok/s (+8.1%).
- MoE models with experts kept in system RAM (`-ot exps=CPU`) prefill faster by default, with bit-identical outputs (greedy 512-token output unchanged). One V100 + 18 CPU threads, NOT-QUIET: Nemotron 3 Nano 30B-A3B Q8_0 pp512 131-140 tok/s vs 63-97 before (mainline with experts on CPU: 116); Nemotron 3.5 Bolt 74B-A9B Q4_K_M pp512 103 vs 57 before (mainline 82); one-token decode is unchanged (25.9 tok/s).
- Nemotron conversion from Hugging Face: Nemotron-4 / Minitron, Nemotron-H / Nemotron 3 Nano (dense and MoE) and the Nemotron 3.5 Bolt text model convert with the bundled converter; output is byte-identical to upstream llama.cpp's converter on Mini-4B, Nano-4B and Nano-30B-A3B, and Bolt's metadata matches NVIDIA's own GGUF (upstream's converter cannot read the Bolt checkpoint).
- Nemotron against upstream llama.cpp, same box and files: one V100, Nano-4B Q8_0 decode +5% and prefill +15%; one P100, Mini-4B decode +51%; Nemotron 3 Nano 30B-A3B with its experts in system RAM on one V100, decode +26% and prefill +9%.
- Nemotron 3.5 Bolt 74B-A9B: the text model loads and runs from files made with the bundled converter (per-layer expert counts and widths, latent MoE); on the same file it matches upstream llama.cpp token for token (PPL within 0.05% on a test trunk).
- Nemotron 3.5 Bolt 74B-A9B at Q4_K_M runs on one 16 GB V100 with its experts in system RAM: decode 22.5 tok/s and prefill 60.6 tok/s, 29% and 31% faster than upstream llama.cpp on the same file.
- Gemma 4 now takes images: the server loads a Gemma 4 vision projector (mmproj) and answers questions about pictures (4 of 4 test images right on both the PXQ and the PXQN files, about 300 image tokens, roughly 1.5 s to read an image on a P100). The launcher finds a matching mmproj file next to the model by itself; --no-mmproj turns that off, an explicit --mmproj wins.
- Qwen3.8-27B PXQ4 (free) ships with a better draft head: +3.8% prose and +4.2% code greedy decode on one P100 with MTP on (default in the 4.0 file; the model weights themselves are unchanged).
- On by default: Gemma 4 26B-A4B decode on one P100 runs its expert router on a faster kernel: 77.2 -> 79.8 tok/s (+3.4%), greedy text unchanged (NOT-QUIET quick probe).
- GLM-5.3-Flash speculative decoding with its built-in draft head (--spec-type mtp) now runs instead of stopping the server; on two 16 GB V100s with experts in host memory it ties plain decoding (about 10 tok/s either way), so automatic speculation stays off for that layout.
- Speculative decoding (MTP) on Nemotron Super now reproduces plain decoding exactly at long context and at draft depth 2-3 on P100 (verify batch identical to one-token decode)
- Nemotron 3.5 Super: the shipped multi-token-prediction head can be attached from its own file for draft decoding (off unless the head file is given; a plain load is unchanged), and the draft check on P100 cards now takes the same arithmetic path as one-token decoding
- Qwen3.6-35B-A3B (PXQN4) on 4x P100: the tuned multi-card decode stack and long-prompt prefill now switch on by default for any set of two or more P100 cards, not only two (4x P100 decode 74 -> 127 t/s, +72%; pp16384 +14%)
- Flint 39.8B MoE runs on PXQN encodes; server decode 1.39x mainline on 2x P100 and 1.22x on 2x V100
- Nanbeige4.3-33B: chat prompts now tokenize exactly as the model's own tokenizer (system/user/assistant turns, thinking on/off, tool calls), so turning thinking off no longer leaks a stray end-of-thinking tag

### Known

- Nemotron-H decode on one P100 is faster: Nano-4B Q8_0 tg128 85.9 tok/s vs mainline 63.5 (35-57 on the first port; NOT-QUIET). Open: Nano-4B Q4_K_M 69.6 tok/s vs mainline 68.4 and a popular community fork 87.0; a fix is built and stays off by default until it is measured on a P100.
- Nemotron 3 Nano 30B-A3B with experts in system RAM: load the model from RAM, not by mmap from a slow disk. Page-cache eviction made one run three times slower.
- Nemotron-4 Mini 4B PXQN4 decodes 4% slower than upstream llama.cpp's Q4_K_M on one V100 (155 vs 162 tok/s; prefill is 15-22% faster); on one P100 PXQN4 is the fastest Mini tier (120 tok/s).
- NVIDIA's own Nemotron 3.5 Bolt GGUF (NVFP4 weights) does not load yet; convert the BF16 checkpoint with the bundled converter instead.

### Long-context prefill

- One card that streams MoE experts from system RAM now picks `-ub 4096` by default: Flash-Next on one V100, 28k prompt, 324 -> 597 tok/s.
- Lean long-prompt attention for Pascal, on by default for Qwen3.5/3.6 MoE PXQN4 on two P100s (next line; one setting turns it off): 35B-A3B on two P100 reaches 1803 tok/s at 16k with it and the default long-prompt prefill below (NOT-QUIET).
- On by default: Gemma 4 26B-A4B on one 16 GB card reserves less memory on every card type (one P100, peak at 8k/16k/32k: -426/-278/-492 MiB, greedy text and top-5 probabilities unchanged) and takes the largest prompt batch that still leaves memory free at load. Prompt speed on one V100: 4k 2558 -> 3298 tok/s (+29%), 16k 2282 -> 2893 (+27%); on one P100: 16k 317 -> 568 (+79%), 32k 297 -> 381 (+28%); 32k and 64k (8-bit cache) on one V100 keep their speed, and every context up to 64k boots on both cards (NOT-QUIET).
- Qwen3.5-MoE on all-P100 (sm_60) cards prefills long prompts faster by default: 35B-A3B PXQN4 on 2x P100, pp16384 1150.6 -> 1569.7 tok/s, greedy output identical.
- On by default: Qwen3.5/3.6 MoE PXQN4 files (35B-A3B class) on two P100s use the lean long-prompt attention with a stable prompt-batch shape: prompt processing +43.5% at 16k and +22.1% at 4k tokens. Output stays correct (12/12 identical repeats, 16k needle found, divergence inside the engine's own noise band) but is not bit-identical to the old path; one setting turns it off.
- 27B-class dense Qwen3.5/3.6 on a P100 pair: the P100 long-prompt attention is now on by default (as for the MoE class): pp16384 +5.5%, pp4096 +1%. Not bit-identical to the old path; the off switch is documented.
- On by default on V100: Gemma 4 long prompts with an 8-bit or 4-bit KV cache keep the fused attention kernel and that kernel works through larger key blocks: 26B-A4B 64k prompt with an 8-bit cache 1491 -> 2421 tok/s (+62%; +29% and +23% for the two steps), long-prompt recall and text checks pass; results differ from before at the last bits (summation order), with an off switch for each step.
- On by default: Gemma 4 26B-A4B PXQN on one V100 with a 16-bit cache tries a 2048-token prompt batch first when the server sizes it: prompt speed +12% at 4k, +8% at 8k and 16k where it fits, output bit-identical.
- On by default: Qwen3.6-35B-A3B PXQN4 on one 16 GB V100 with part of its experts in system RAM keeps more experts on the card: decode about 9% and prompt processing about 20% faster than v3.1 (quick probe, NOT-QUIET). Greedy text can differ from earlier builds; each run is deterministic.

### Tensor split

- Flash-Next on a P100 pair now auto-picks `-sm tensor` with its split settings; a V100 pair stays on layer split. The Flash-Next presets also carry a bit-identical speedup (see Overdrive / Flash-Next).
- Dense models on a P100 pair with tensor split: faster prompt processing, output identical. 27B: pp512 +8%, pp4096 +14%, pp16384 +12%, then a further +4% at 4k and 16k prompts.
- Two-card tensor-split decode is faster by default, with bit-identical outputs: Qwen3.8 27B PXQN4 on two P100, 512-token greedy decode 37.4 -> 39.2 tok/s (+4.5-5.0% over v3.1, NOT-QUIET); multi-token prediction on the same setup gains about 1% more.
- On by default: the Qwen3.8-27B OneCard file on a P100 pair now runs with tensor split instead of layer split: greedy decode about 31% faster (40.1-40.3 vs 30.5-30.6 tok/s, 512-token replies), same greedy output.

### PXA Control

- Assistant chat fixes, covered by a 152-case test suite; a turn cut by a Control stop or restart comes back as interrupted with one-click Resend.
- PXA Control ports: Windows (exe and engine), macOS Intel and Android, and the installable app (PWA).
- Pre-release builds are named 4.0-pre (Android APK version, user agent and file name; Control docs); the 3.2pre name is retired.
- PXA Control: the left menu and the header now stay in place while the page scrolls.
- PXA Control: the KV cache types are listed in order, from f16 down to q4_0 (community request).
- PXA Control: in the encoder's model picker, the folder, model and GGUF tags show as a separate label again instead of running into the names (community report).
- Benchmarks record the average and peak watts the cards drew during a run, and show tokens per second per watt in the history table and a small chart.
- Dictate, a Control plugin prototype: hold a key, and the words it recognizes are typed into the window you are using. This first version knows a small fixed vocabulary; open speech comes later. Speech stays on this computer. It stays off until supporter coins unlock it in the Store.
- Speed tab: the request tables and graphs now redraw by themselves when a benchmark finishes (they used to need a page reload)
- Speed tab: the benchmark history pages through your runs (20 per page) with a search box (model, cards, sha, date) and new MTP and Split columns
- The Speed tab is now called Benchmarks and has a server strip on top: status, Start/Stop and a basic edit (model, MTP, split mode) for the server you are measuring, so you no longer leave the page between runs
- Supporter plugins arrive as signed packages from the Store, only on a computer that holds the plugin's unlock, and one check decides whether any supporter plugin runs: an unlock made for this computer, never a copied wallet file. Dictate is the first.
- Scribe, a supporter plugin that takes live notes of a call on this computer (running transcript, then summary, decisions and action items), is built as a prototype and stays hidden until its package is listed.
- Bug reports from Control now send correctly on machines whose name matches the product name
- Chat: a reply whose save failed no longer leaves the chat waiting (every message 3 seconds late, branching refused); the failure is logged.
- Models tab: a damaged remembered Hugging Face list reads as no list instead of an error.
- A shell variable kept over the launcher's pick is shown as hidden in the server log when its name says it holds a credential.
- Chat: a web page that sends its text very slowly can no longer hold a chat; one page gets 45 seconds in all, and Stop ends the fetch at once.
- Chat: a damaged chat file no longer slows the chat list down (its title is read from the file's head).
- Chat: a command's output is never written to the Control log; the log only notes that the output named the chat's folder.
- The share page asks you to sign in like every other Control page when it is opened without the session.
- Benchmark repeat runs are honest: the warm-up uses its own prompt, a server that cannot forget its replies gets a fresh prompt per run, and each class shows the first pass next to the later runs, with a warning when later runs replayed an earlier answer.
- Encode tab: right after installing the Pro encoder the page could still show the Free one when a slow background refresh finished last; a refresh that began before a settings change no longer replaces the newer result
- PXA Control no longer writes Python bytecode cache folders into the install directory.
- Android app: the microphone is given only to your own rigs, and links to other sites open in the phone's browser
- The rig picker's offline screen no longer opens an address it does not know by itself
- Submitting a new high score now closes the high-score frame and shows the board's answer; a result you already sent is not offered again, also after a reload (community report).
- Update now checks for a new build every time it opens and says Up to date when there is none; it asks you to stop servers only when an update is waiting and a server really runs (community report).
- The footer with the Discord and Ko-fi links, the PXA Control version and the Update button now shows at the bottom of every page, Home included, not only on Servers.
- Store: entitlement records stay reliable when several PXA programs use the store at once, and a computer whose clock ran ahead recovers on the next store sync.
- PXQN encodes take 10 coins each, charged only when the encode completes; a failed encode costs nothing. The Encode tab says so.

### Docs added or changed

- This rolling changelog and the 4.0 highlights layer.
- The release package ships the gate record bench/gate/LAST-RUN.md with build-machine paths removed.
- docs/VLLM.md documents the opt-in field patches for vLLM on V100 SXM2 at tensor parallel 4 and credits the community field report docs/FIELD-REPORT-V100-SXM2-TP4.md.
- PXA Control 4.0 includes a home guide: how to install and launch, what each tab does, benchmark history, the share card, chat, and the library updater.
- 4.0 beta 2 user pages: who gets the beta (a Valued Supporter reward), how to get it through Discord, where the key goes in PXA Control, and how the wallet and the updater fit in

### Speculation / MTP

- MTP-armed servers on Qwen3.5/3.6 MoE files on P100 now decode at plain speed when no draft is taken (35B-A3B on 2x P100: 91 -> 124 tok/s with drafting off; plain decode +3%, 121 -> 125 tok/s), and the first reply after boot is byte-identical to later ones. Default, no switch.
- MTP verify on a P100 pair is faster by default and bit-identical: 35B-A3B MTP verify step 11.98 -> 10.74 ms at 2 tokens and 14.61 -> 13.10 ms at 3, MTP tok/s +8..12% in all 8 measured cells (NOT-QUIET).
- Two further P100 MTP verify speedups are now on by default and bit-identical (35B-A3B on 2x P100: about 0.4-0.5 ms less per verify step).
- On by default: a bare launch of a Qwen3.5/3.6 MoE MTP file (PXQN4) on a P100 pair with layer split picks the measured P100 decode settings and MTP n_max 3 p_min 0.85: 35B-A3B plain decode 72.7 -> 125 tok/s, MTP bare launch 74 -> 138 / 151 / 130 tok/s; not bit-identical to before but correct (decode KLD 2.5e-3 against the engine's own width noise 1.8e-3, prefill identical, determinism 12/12, needle 28k pass).
- On by default: when a launch with no speculation flags arms MTP by itself and the draft context does not fit on the card (Qwen3.8 27B PXQN4 on one V100 at 32k context), the server decodes plain instead of exiting (41 tok/s, needle pass); an explicit `--spec-type mtp` now does the same (see the server fix below). 8k, 16k and 24k keep MTP (80 / 70 / 63 tok/s on a 128-token check, NOT-QUIET).
- On by default: Qwen3.8 27B (PXQN4) on one V100 with no speculation flags drafts deeper per step (V100 only; other card types keep the earlier setting until measured). Against v3.1 on the same box, stock file, 512-token replies (NOT-QUIET): greedy prose +14.8%, code +38.4%, chat +11.9%; sampled, on the exact draft rule, +12.6% / +36.0% / +11.8% (v3.1's sampled row still uses its older draft rule). Greedy text is identical to plain decode.
- Sampled requests (temperature > 0) with MTP draft depth above 1 and a draft floor or depth cutoff now emit exactly the model's distribution; before, the draft floor leaned output toward the most likely tokens (toy TV up to 0.05-0.18 on a 3-token joint). Greedy output unchanged.
- MTP now fits longer contexts on one V100: the Qwen3.8 27B PXQN4 MTP file keeps MTP at 32k, 40k and 48k context where a launch without speculation flags used to fall back to plain decode (42 -> 60 tok/s, identical greedy output); its draft context needs 172 MB less VRAM, and a server that cannot fit MTP returns the 161 MB it had set aside. At 40k/48k the server picks a smaller prompt batch by itself (prompt processing of a 38k-token prompt 831 -> 685 tok/s with MTP on); passing a batch size keeps yours.
- Nemotron-H MoE files larger than the card (experts streamed from host memory) no longer turn on MTP speculation by themselves: there it measured about 10% slower than plain decode. Asking for speculation explicitly still turns it on; an off switch is documented.
- Qwen3.6 35B-A3B MTP decode on two P100: faster draft-token pick and a faster verify attention step, with the two-card output head: MTP decode +8.8% prose, +9.5% code, +6.1% chat over the same build without the three; on by default

### Overdrive / Flash-Next

- The opt-in async cold path of Overdrive (default off) now gives the same reply sequence on two boots.
- Flash-Next is about +7% faster by default at the shipping level, bit-identical.
- Nemotron 3 Nano / Super and Nemotron 3.5 Bolt larger than the card now use the fast expert placement by default instead of a slower fallback (Nemotron 3 Nano 30B on one 16 GB V100: 7.3 -> 37.5 tokens/s; Nemotron 3.5 Bolt on one 16 GB P100: 4-7% faster).
- The Flash-Next Overdrive setup is now carried and chosen by the PXQN library and applied by the engine itself at startup; the launcher and the preset files carry only the on/off switch and the box rows. Same seat, same output.
- Flash-Next on one V100 with experts offloaded to host RAM: the expert cache now adapts to what the model is routing while it generates, and the output stays deterministic; prose +9.7%, code +6.5%, chat +15.4% ahead of the fastest other engine on the same card.

### Quantization / PXQN

- PXQ and PXQN encodes of Nemotron-H (Mamba-2 hybrids) are closer to the Q8_0 original: Nemotron-3-Nano-4B KLD against its Q8_0 0.0193 -> 0.0160 (PXQ4) and 0.0162 -> 0.0132 (PXQN4) for +0.26 GB.
- PXQN4 and PXQ4 for Nemotron: Mini-4B PXQN4 has half the KL divergence of Q4_K_M (0.023 vs 0.044) and decodes at 120 tok/s on one P100; Nano-4B PXQ4 / PXQN4 files are supported (tools/pxa_pad_ssm_in.py pads an existing GGUF).
- The PXQN encoder pipeline now reads each layer's width from the model itself and checks every calibration file against it before encoding; an encode that crashes can no longer leave an unfinished file under the final name

### Removed / deprecated / behaviour changes

- On the 4.0 line: sliding-window models (Gemma and other SWA files) that rewind a sequence further back than the window now re-process that sequence instead of hitting an attention assert or a wrong token. Default, no switch.
- On the 4.0 line: the per-card CUDA graph environment switches treat 0 or an empty value as off; before, any value, including 0, turned them on.
- A process that loads more than one model (a draft model, a reload, a benchmark over several files) no longer carries the Nemotron 3 Nano expert-cache settings into the other models; each load keeps its own choice. Nemotron 3 Nano output and speed are unchanged.
- On/off environment switches now read on, true and yes as on (they used to read as off); 0, off, false and no stay off, an empty value means the default, and numbers keep their meaning.
- The updater installs an engine package built for a narrower set of GPUs only when that set includes the GPUs in this machine; otherwise it keeps the standard package.

### Engine (CUDA)

- P100 fast decode: new opt-in exact mode makes the fast MoE decode path's output bit-identical to the standard path (KLD 0, greedy text identical); about 5-6% slower decode, so it is off by default.
- llama-cli, llama-bench, llama-perplexity and llama-imatrix now apply the same measured per-card/per-model defaults as llama-server; one environment switch turns the defaults off for every tool (server batch/ub picks included). Only the model given with -m sets the defaults (a second model gets a warning per differing pick); gate tools take lever settings only (never KV, context or batch picks); llama-cli takes a context pick only without a user -c and never below the model default; llama-bench applies only the picks shared by every test instance.
- One P100 with a PXQN file gets two more bit-identical decode speedups by default.

### Engine kernels, Volta (sm_70)

- On by default on V100: a new MTP verify path for Qwen3.8 27B (PXQN4); correct but not bit-identical to the old verify (determinism 12/12, needle pass, KLD 0.0097 against the engine's own 0.0095 band). A second verify change that goes with it is bit-identical.
- On by default: Gemma 4 26B-A4B decode on one V100 is about 12% faster than v3.1 (131.4 vs 117.5 tok/s in the quick test), greedy text unchanged.
- On by default: Gemma 4 PXQN prompt processing on one V100 is faster (4k prompt +5.5%, 16k +3.1%), output identical.
- On by default on V100: Gemma 4 26B-A4B PXQN prompt processing runs the mixture-of-experts layers on the fast PXQN expert kernels with tensor cores: 4k prompt 661 -> 4307 tok/s, 16k 1467 -> 3628 (NOT-QUIET).
- Llama-architecture models on one V100 decode about 5% faster by default (Nemotron-Nano-8B: 130 vs 124 tokens/s), output unchanged bit for bit

### Engine kernels, Pascal (sm_60)

- Faster PXQN4 multi-token verify on Pascal, on by default, output bit-identical; 27B on 2x P100 with MTP: greedy +7.3/+10.5/+10.2% vs before (prose/code/chat), +6.7/+7.0/+9.4% vs v3.1 same box, same prompts
- Faster single-token decode is now on by default for one P100 with a dense 27B PXQN4 file, output bit-identical; server decode +4.6% prose / +4.8% code vs before, +4.9/+5.3% vs v3.1 same box.
- Pascal decode no longer risks a hang on a launch that cannot fit on the card: it falls back to the standard path; no change on the measured cells (output identical).
- On by default: Gemma 4 PXQN prompt processing on one P100 uses a larger prompt batch when it fits (4k prompt +5.5%, 16k +1.9%), and Gemma 4 decode on one P100 is about 12% faster (68.3 -> 76.8 tok/s).
- Qwen3.6 35B-A3B on two P100 (layer split): the output head now runs on both cards, plain decode +8.6% (output unchanged); on by default

### Decode speed

- 27B plain decode (MTP off) on one V100 runs at 43.2 tok/s on code and 43.5 on prose (server greedy, load-noisy box), output bit-identical, on by default for Qwen3.5-class models on one V100. The same-bracket 3.1 comparison is pending.
- Decode keeps full speed when the host runs short of memory: on fully offloaded models the small set of weights kept in system RAM no longer has to be re-read from disk mid-generation (27B on two P100s with MTP under memory pressure: 22-36 t/s -> 46-62 t/s; no change otherwise, output identical).
- Fixed: the copy of host-kept weights into locked memory now applies only when every layer is on the GPUs, and an invalid size cap turns it off instead of lifting the limit.
- Nemotron 3 Nano 30B on one 16 GB card: decode about twice as fast by default when CPU cores are free (one V100: 45 -> 89 tokens/s prose, 38 -> 88 code, 2.3x a popular community fork); when other work keeps the CPU busy at start-up it takes the GPU read instead of slowing down (50 tokens/s under load where the CPU-only choice dropped to 3.4) and keeps that choice for the whole run, so greedy output is the same from request to request (switching during a run is opt-in)
- GLM-5.3-Flash: six new opt-in decode paths for the 2x V100 + host-experts layout (fewer kernel launches, graph reuse, experts split per layer between the cards and the CPU); off by default until measured
- Qwen 3.5 / 3.8 on two V100 cards with tensor split: about 5-7% faster decode by default, same output bit for bit

### Fixes

- A greedy request whose output row has no finite logit (all NaN / -inf) now takes the same fallback token as the host sampler and counts toward the softfail stop, instead of sampling a stale row.
- Fixed: long-running servers no longer lose prefill speed after about a dozen requests (reported by WebWalker3D, #11). On two P100s with 12-24k-token prompts, prefill now holds about 1,690 t/s through 40 requests where it used to fall from 1,676 to 933 t/s (-44%) from request 13 on; short-prompt serving with speculative decoding no longer loses about 18% after some 30 requests. Default on, same output.
- GLM multi-token prediction (MTP) with up to 4 draft tokens no longer aborts with heap corruption on the CPU: a CPU memory buffer could report up to 16 bytes more than it owned, and a plan that grew by exactly that much was placed past its end. Output unchanged.
- GLM: repeating the same greedy request now gives the same answer every time; before, a request could come out different depending on the requests that ran before it on the same server. Output now always matches a fresh server's first answer.
- Fixed a crash on V100 with some models during speculative decoding
- Gemma 4 (and other MoE models): a short raw prompt right after the server starts no longer decodes differently from the same prompt later in the session; answers no longer depend on which requests ran before (default on, no speed cost).
- Reference mode (every speed lever off, for comparisons) again turns off all the newest speed paths; normal runs are unchanged.
- Test suite: the encoder and Encode-tab tests now pass inside the CUDA build image and under parallel test load (they read the host's CUDA library path and raced a process start); the encoder's CPU matmul check accounts for the fast CPU path
- Test suite: the live Home Assistant publish check tolerates memory moving on a busy machine and checks the published units against the sample itself
- BERT-style embedding models (WordPiece vocab): capital letters are lowercased again instead of becoming unknown tokens, so their tokens match the reference tokenizer; upper-case contractions (I'M, DON'T) also split like the reference on GPT-2/Llama-3 style vocabularies.
- CPU-only runs: -fa on runs the CPU flash-attention kernel again (it had silently run the -fa off path); a sequence's output no longer depends on where its KV cache cells start. GPU runs unchanged; -fa off restores the previous CPU behaviour.
- Built-in chat formats (used without --jinja) now match the model's own template for GLM-4-0414, GLM-Edge, phi-4, Granite and YandexGPT 5 Lite (new built-in names phi4, glmedge, yandex); GLM-4.5 and later are unchanged.
- The full CPU test suite passes offline: the eval-callback test runs on a generated tiny model instead of a download, and tests that need optional Python packages report skipped (not failed) when those are missing.
- Nemotron 3.5 Bolt (and other Nemotron-H MoE files whose experts live partly in host RAM): greedy output is now identical from request to request; before, the engine could move decode between the CPU and the GPU read of the host experts mid-run when the box got busy, which changed the text (2 to 5 different greedy outputs over 6 runs of one prompt). The path picked at start-up now holds for the whole run; switching during a run is opt-in
- Perplexity, KL-divergence and embedding runs on a card too small for the whole model no longer fail with 'a smaller context may fit': expert placement now leaves room for the full logits and the output-layer matmul of runs that need every token's output (server and chat runs keep the same placement)
- Fixed a crash while loading (CUDA illegal memory access) with large mainline-quantized MoE models that do not fit on one GPU, seen with Nemotron 3.5 Bolt 74B Q4_K_M on a single V100.
- Flash-Next on Overdrive with two P100 cards: greedy output is now the same on every run (it could change from run to run), and decode is faster (prose 19 -> 41 tok/s, chat 36 -> 52 tok/s).
- Long prompts repeated on a tight card no longer crash the server: the cached GPU replay plans give way when memory runs low (Flash-Next 30k-token prompts on two P100 cards ran out of memory on the third request).
- A plugin may serve on a free port on this computer only, and the consent card shows it
- PXA Control on Windows now checks whether a server or plugin process is still running correctly, so Stop and the run list no longer mistake a running process for a stopped one or the reverse
- Store items now say they unlock with PXA coins
- A plugin limited to named websites keeps its network wall when it also uses PXA Control's own web fetch
- Plugins that ask a model to chat now get a clear reason when the server is loading, the prompt is too long, the server refuses, is down or is slow
- Nemotron-3-Nano and other MoE models no longer produce NaN output on CUDA when one token's expert activation goes past the half-precision range: the affected rows are scaled by an exact power of two around the expert matrix multiply (outputs that never overflowed keep their exact bits)
- PXA Control now starts even when a record left by an earlier run cannot be read; the record is set aside instead.
- Gemma 4: no more garbage text with flash attention off (long-documents mode, FA off in PXA Control, max mode, -fa off).
- Hybrid MoE models (Qwen3.5/3.6 MoE and similar) with layers left on the CPU (-ngl 0 or partial offload): perplexity and output no longer jump with batch size, perplexity mode or from run to run. A fused GPU norm kernel could overwrite the copy of a CPU-held weight it was still reading. Qwen3.6 35B at -ngl 0 (2 chunks of 2048) now gives 8.66 at batch 512 and 8.61 at batch 256 in both modes, close to the full-GPU path (8.49) and the CPU alone (8.62); before: 14.53 or 8.66 at batch 512 and 8.85 or 8.94 at batch 256. Full-GPU output is unchanged.
- Speculative decoding no longer crashes the server with a GPU out-of-memory at long context: the per-width verify graphs are sized at load for the memory actually free, widths that do not fit verify on the normal decode graph (same output), and a verify graph that cannot be allocated later falls back to the normal graph instead of aborting.
- The Fusion2-35B PXQ6-MTP bench file did not load on 4.0 (its MTP block used a retired weight type). The re-published file holds the same values in a current type and loads on 4.0 and 3.1; re-download it if you have the old copy.
- Chat templates: \\U escapes above U+10FFFF and lone \\u surrogates give U+FFFD instead of invalid UTF-8
- Flint: no more '!!!!' output on V100/P100 when expert activations overflowed half precision
- Two V100 cards with the tensor split no longer return garbled output ('!!!!') on prompts longer than about 6000 tokens
- Overdrive (Flash-Next class on one card): under heavy host load some expert results could time out and be dropped, which changed the text from run to run; the engine now waits for them.
- Hemlock-class models with speculative decoding on: repeated identical greedy requests now return the same text, and on Pascal cards the same text as with speculation off. Plain (non-speculative) output of this family can differ slightly from the previous build.
- Volta and newer, attention without flash attention on models without grouped KV heads: long-context decode no longer pays an extra copy of the V cache per layer (7B at 8k context: 34 -> 58 tokens/s on a V100, output unchanged)
- The same half-precision overflow protection now also covers dense matrix multiplies, the fused MoE up/gate path and half-precision GEMM outputs on P100 and RTX 20/30/40-class cards, so one out-of-range activation can no longer turn a layer's output into NaN
- The switch for the Gemma 4 flash-attention-off repair (and one prefill switch) now reads on, true and yes as on; these values used to read as off and quietly disabled the repair. 0, off, false and no still turn it off.
- Nemotron-H models with experts in host RAM: a short burst of other CPU work during model load no longer locks the whole run into the slower zero-copy expert path (Bolt-74B on 2x P100: about 10 t/s instead of 22-24, and different greedy text). The load now re-checks a busy reading over a few seconds before it commits to that path; a quiet box loads exactly as before.
- The op self-test accepts the GPU's slightly different (and for q4_0 slightly more accurate) rounding when copying into q4_0 and iq4_nl caches
- Mixture-of-experts models with unquantized expert weights no longer abort with 'stream is capturing' on RTX 30xx and newer during single-token decode
- The CUDA backend now declines malformed SWIGLU shapes it cannot run instead of asserting; the op self-test checks them with the proper shapes
- pxa bench card: a class whose every run was all thinking (or ended at once) now says no answer: all thinking next to its rate, with a note that the rate is not a speed to compare
- Control: a thinking budget is kept below the context, so a small context still leaves room for the answer; a small context for a thinking model now shows a warning
- Control bench: a class that wrote no answer shows no answer: all thinking (its speed counts the thinking tokens) instead of a made-up speed, and is not sent to the high-score board
- Launcher: two cards of different sizes (for example a 16 GB and a 12 GB P100) serving a model with its vision projector no longer run out of memory at the first prompt; the model is split by each card's room and the measured batch size is kept
- Bench: a chat model that ends a plain-text bench prompt at once is asked again instead of showing 1,000,000 tokens per second
- Server: ignore_eos now blocks every end-of-turn token, so a forced-length request runs to its length on chat models
- Server: a thinking budget larger than the context fits the context, so the model closes its thought and answers instead of thinking until the context runs out

### site

- Website v2 live at https://pxa.pxanetwork.com: one drawing-sheet design on all 7 pages (light and dark, phone width), every number from a measured run, shown with its row id, date and spread, the replay race at measured rates, a calm supporter path; Control screenshots are still the v3 ones.

### Server

- Fixed: a request's token bias or end-of-text ban no longer carries over to the next request on the same slot (public report #9; thanks to the reporter)
- Fixed: a crash at server exit with speculative decoding on. Partly fixed, still open: a heap corruption crash when GLM speculative decoding drafts several tokens per step (one draft token per step passes; four per step can still crash).
- Opt-in: on one card that streams a large MoE model, the automatic prefill batch can step down at load so the MTP draft chain keeps its full depth (decode +1-9% measured, one more batch on some long prompts).
- Chat templates that only accept a system message first (Qwen 3.5 and similar) no longer fail when a client sends a system message mid-conversation: later system messages are merged into the first one. Other templates are unchanged.

### Engine

- Fixed: MoE models with experts kept in host memory no longer abort on long prompts (Gemma 4 QAT and other merged-expert files); confirmed on a Gemma 4 PXQN4 file with its experts on the host and a 3k-token prompt
- Faster CPU matrix kernel for PXQ4 weights computed on the CPU (experts kept in host RAM, CPU fallbacks): 1.5-1.9x per thread on the Nemotron 3.5 Bolt expert shapes, same output bit for bit
- Flash-Next decodes 7-9% faster on two V100 cards with speculative decoding, on by default with the same output.
- Flash-Next on V100 cards drafts deeper when the draft head is on a run: speculative decoding adapts its draft length to how often drafts are accepted, about 3% faster on code with the same output. It applies to the measured setting only and can be turned off.

### Launcher

- pxa-launch finds the GPUs without nvidia-smi (containers / LXC that pass the driver through but not the binary): it reads the NVIDIA management library or the kernel driver's GPU list instead, same card numbering; when nothing can be read, --gpus is trusted as given and the assumed memory is printed (public issue 4).
- --spec mtp on a file without a draft head: the refusal now says which case it is (made without the head, or a header that declares a head the file does not carry) and what to do, instead of one message that blamed every such file (public issue 7).
- A model file without a draft (MTP) head no longer stops the launcher when MTP is requested: the MTP flag is dropped with a one-line note and the server starts without speculation (public issue 7). PXA Control greys out its MTP box for such a file and says why.

### Packaging

- The release package carries a modulefile: on clusters, `module use <package>/modulefiles; module load pxa/4.0` (Environment Modules and Lmod) sets PATH and the library path, picks the compatibility library on CPUs without AVX2, and leaves the GPU choice to the scheduler.
- Three dead cross-links in the packaged documentation are fixed.
- The Docker image can build in the faster LLGuidance grammar engine for JSON-schema and tool-call grammars (pinned Rust in the build stage only; the runtime image carries no Rust).
- The PXQN library's licence file now ships with the Linux package too, next to the MIT licence, and the licensing map points to it
- Release packages no longer carry any add-on; the package check refuses one.
- The Linux release tarball is now named after the GPU architectures it carries (sm 60, 61, 70, 75, 80, 86, 89), with the old -sm60_61_70 name kept as an identical second file so existing updaters keep working.
- Windows package: the engine zip now includes LICENSE and NOTICE (with the CUDA and Visual C++ runtime notices), and a package checker verifies GPU architectures, DLL dependencies, exports and hashes before release.
- Release scan now also checks programs packed inside single-file executables (Windows, macOS, Linux); the Windows PXQN library ships with its own licence file and exports only its public entry points.

### Update

- Beta channel for Valued Supporters: opt in from PXA Control's update dialog to get new engine builds first, read the beta notes, and go back to stable at any time. Downloads resume after a dropped connection.

### Community

- The PXA Discord bot now also answers from our public web pages (main site, changelog, setup guide, bench board) and quotes the page it used; the page index refreshes every 30 minutes. The owner's own links to our sites no longer get a bot reply.

### Launcher and startup

- Fixed: the startup warning about experimental switches no longer lists settings you never set (the engine's own defaults and what the launcher picks for you); it names only the experimental switches you actually put in your environment.

### Release

- Every release package now passes a clean-package scan before it is published: no build-machine paths in the binaries, no internal names or secrets in the shipped files; a failing package is not produced.
- Release packages are checked to carry native kernels for every supported GPU generation (Pascal, Volta, Turing) before upload.
- The public source tree passes the source-export scan with no internal record ids, machine paths or internal notes; the Windows package check now needs its block list named explicitly.

### speculative decoding

- Sampled generation with MTP drafting now follows the model's distribution exactly in every configuration: a minimum-draft-length setting and some cache-driven draft cuts could still lean it toward the drafter's top tokens. Greedy output is unchanged; sampled speed is within noise (one P100, 27B: -2% prose, -4% code at N=6).

### Memory

- MTP speculative decoding no longer holds a large block of host RAM from startup: a 27B model at batch size 8192 on one P100 now uses 0.7 GB of host RAM instead of 17.6 GB, with identical output and the same speed.

### Speed

- Qwen 27B PXQ3 decode on Pascal P100 is about 3% faster by default (the dense up/gate step now uses the half-precision path); an off switch is documented.
- Switching between conversations on one server slot now keeps the speculative drafter warm: the first turn after a switch decodes about 4% faster with higher draft acceptance, output unchanged.
- One P100 with a 27B-class PXQN file: three more bit-identical decode kernels are on by default (about +2.5% server decode).
- Flash-Next on cards it does not fit: the engine now applies the measured Overdrive setup by itself, not only through the launcher or Control (+14..55% decode on P100).
- 27B prompt processing on one P100 is about 3% faster (4k and 16k prompts), with identical output.
- Models whose cold experts run on the CPU no longer stall when other programs keep a CPU busy: on a loaded machine the 35B-A3B decode went from 26-34 to 76-83 tok/s.
- Qwen3.5/3.6 MoE 35B-A3B on one 16 GB V100 (part of the model in system RAM): decode 59-63 -> 72-82 tok/s (prose/code/chat, REPS 6, same bracket as v3.1), about 2x the other engines on the same card.
- Prefill on Volta (V100) cards is faster when the model's fp16 weights fit in spare VRAM: they are converted once and kept, instead of re-converted for every prompt. Nemotron-4-Mini-Hindi-4B on one V100: pp512 +12%, pp2048 +13%, decode unchanged, output identical.
- Flash-Next on four P100 cards in pipelined-prefill mode: the per-batch wait between cards is gone by default, long-prompt prefill up to 24% faster in that mode, output unchanged
- Qwen 27B/35B MTP models on V100: the n-gram drafter now runs together with the MTP head by default (one card and tensor split, also from Control), output stays identical to plain decoding: agent-style file rewrites +47% on one V100 and +68% on two, prose/code/chat unchanged or slightly faster
- Prompt processing on PXQN mixture-of-experts models is about 6x faster on RTX 30-series and newer cards (RTX 3080 Ti, Hy-Memory 30B-A3B: 485 -> 2946 tokens/s at 512 tokens)
- Prompt processing on PXQN mixture-of-experts models is about 1.5x faster again on RTX 30-series and newer cards: the expert layers now run on the tensor cores (RTX 3080 Ti, Hy-Memory 30B-A3B: 3035 -> 4697 tokens/s at 512 tokens; generation speed unchanged)
- Flint on one 16 GB P100: prompt +20% and decode +21% over the previous default
- Flint runs on one 16 GB V100 with no flags: the engine picks the one-card layout itself
- CUDA top-k expert selection for MoE routers is 1.5-2.3x faster per call, output unchanged
- On by default: Flint B (the mixed PXQN4/PXQN5 Flint file) now gets the Flint one-card speedups too: decode +11.9% on one V100, and +3.2% on code prompts on one P100 (prose unchanged); determinism, needle and quality checks pass.
- On by default: the Qwen3.8-27B OneCard file on one P100 now gets the 27B speedups too, with the same greedy output: decode +7.3%, server replies +8.9%, prompt processing +3.3% at 4k and +8.8% at 16k tokens.
- Flash-Next on V100: faster speculative verify (two-phase hyper-connection GEMVs and fused small-kernel chains), same output; 2x V100 +14% to +17% (prose, code, chat)

### Tools

- New load-time fit check for large multi-GPU layouts: allocates everything a real run would without reading the weights, so a layout that will not fit is found in minutes.
- New read-only header audit (pxa-mtp-audit) checks every model file, local or published, for the draft head its name, model card or preset promises, and the reverse; it reads only the GGUF header.

### Multi-GPU

- Four-card tensor-split decode is much faster by default on four P100 cards, with the same greedy output: Qwen3.8 27B on 4x P100, 512-token greedy decode 29.2 -> 40.4 tok/s (+38%) in all three prompt classes; two-card splits are unchanged.

### Build

- Shipped binaries and their startup log no longer carry internal record identifiers in any build configuration; a new test scans every build's binaries for them.
- Windows: the importance-matrix tool and the test suite compile with the Microsoft C++ compiler. Two tests that need Linux file features skip on Windows.
- Windows CUDA builds compile cleanly against the Windows SDK headers without a compiler workaround

### engine

- Two-card P100 decode of 27B hybrid models about 1% faster (fewer small launches per layer, output unchanged)
- Hybrid recurrent models (Qwen3.5/3.6 DeltaNet): speculative decoding no longer crashes or asserts when recurrent states sit in host memory (-ngl 0, partial offload, --no-kv-offload); checkpoint save and restore now find the right backend
- Hybrid recurrent models: per-step speculative rollback is exact again with recurrent states in host memory and long draft batches, and on GPUs that cannot run the recurrent kernel at the model's head size (those fall back to the whole-state checkpoint); outputs of fully offloaded models unchanged
- Auto-config: the explanation for the chosen batch size stays correct when a multi-card layer split is re-planned (it could show a stale or garbled reason)
- Expert cache: auto-adapt no longer aborts when some expert layers are kept in host memory with -ot on a GPU build
- Qwen3.8 27B (PXQN4) on two P100 cards with tensor split: plain decode about 2% faster (39.3 -> 40.0 t/s), output unchanged.
- CPU flash attention now keeps its running sum in full precision, so long-context attention on the CPU (CPU-only runs and layers left on the host) drifts less from the reference; prompt processing on the CPU is about 4% faster at 4k tokens.
- Flint on one P100 card gives the same greedy output on every run again: the fused MoE router step, which could race when the memory planner shared its buffers, is now opt-in (off by default) and guarded when switched on
- GLM-5.3-Flash decodes faster with the same output, token for token: about 1.3x v3.1 and 2.5x mainline on six cards (P100 + V100), and about 2.1x mainline on two V100s with experts in system RAM, where it is also ahead of v3.1.
- A chat template that fails to load no longer crashes the server when a client reads the server properties or the thinking support is probed at startup; the template accessors answer safely and chat requests get a clear error
- P100: attention outputs no longer depend on where a sequence sits in the KV cache when the attention runs without the fused flash-attention kernel (fixes 4 KV placement test failures); long-context decode with flash attention off is about 6% slower on P100 (the flash-attention path is unchanged).
- V100 and newer cards: a request's logits no longer depend on the KV cache cell it starts at (they could differ by up to about 0.013 at offsets that are not a multiple of 4); Pascal unchanged
- Nemotron-H decode on one RTX 30xx/40xx/A-series card: the input copies no longer stall the GPU and the decode graph runs as a single launch per token (Nemotron-3-Nano-4B on an RTX 3080 Ti: about 157 -> 241 tokens/s decode, same output).

### Speed (P100)

- Qwen3.6 35B MTP on P100: fixed a stale plan reuse that could make the exact verify mode roll back speculative drafts from the wrong state; default output unchanged.

### cuda

- P100 (sm_60): faster prompt processing for PXQN mixture-of-experts models at short prompts (Qwen3.6-35B-A3B on 2x P100: pp512 about +8%, pp4096 and pp16384 about +2%); outputs unchanged
- P100 (sm_60) MTP speculative decoding on PXQN mixture-of-experts models: two further verify-step speedups now on by default (Qwen3.6-35B-A3B on 2x P100: verify step about 0.4-0.5 ms faster per draft batch); outputs unchanged
- V100 (sm_70): fixed a possible 'illegal memory access' crash in flash attention prefill at long context; the mask scan no longer reads past the mask (outputs unchanged)

### Control

- Report a problem now works for a server you started yourself: Control lists servers it finds on this computer, or takes the server's address.
- Model downloads check the sha256 while downloading, so finishing no longer re-reads the whole file; a resumed download shows how far the check is and the time left.
- Benchmarks: choose the KV cache type (as served by default, or f16/q8_0/q4_0); the result and the history show the KV each run used.
- Beta wording for the Valued Supporter beta: pxa-update, the Beta card and the PXA Coins panel say the beta is for Valued Supporters (no coins) and point 3.1 / beta 1 users to #support for the one-time beta 2 handover.
- Valued Supporters on any version get the beta with /beta in the PXA Discord; Control, pxa-update and the beta guide point there.
- Chat: the Run command tool no longer shows the model the chat folder's absolute path in its output
- Chat: a chat whose file is damaged shows as a marked Broken row with Export raw and Delete, and opens to a plain explanation; an image the server refuses now pops up the reason and gives your typed words back
- Chat file errors show the model a folder-relative path instead of the machine's full path
- Chat web_fetch now connects to the exact address it checked, so a changing DNS answer cannot reach your own network
- Chat: one device can have only 3 chats answering at once by default (a settings value changes it, 0 turns it off); the next one is told to wait
- Chat: an image that is the wrong type or is damaged is now refused with the reason instead of being quietly left out
- Chat: a chat file that cannot be read now shows in the list as 'could not be read' and can still be exported raw or deleted, instead of vanishing
- Chat: fast streaming replies send about 25 updates a second instead of one per word, so a long answer keeps the page smooth
- Chat: a system prompt, memory or attached chats too big for the model's window are now trimmed (memory and attached chats) or refused with the numbers (an oversized system prompt) instead of ending in a context error
- Assistant Advanced settings: stop sequences, a JSON schema and a grammar can now be set for the reply
- Chat: a stray system message in the middle of a conversation is merged to the front before it reaches the model, so strict chat templates do not refuse it
- Chat: the live stream of a reply now ends only after the chat is saved, so reloading or exporting right after an answer always shows it
- Store > Wallet: gift coins to another supporter by wallet ID or Ko-fi email, with a preview to confirm before anything moves; gifts show in both wallets' history
- Models tab: the model folders (add, rescan) now sit above the Library, and the list of PXA models is no longer re-read from Hugging Face on every visit: it is remembered with its age ("checked 40 min ago") and a Check PXA models button reads it on demand; it re-reads on its own only when older than a day or when no model is installed yet.
- Live tab: a Running only switch (on by default, remembered on this machine) folds stopped servers into one line, "N stopped (show)", so servers that come and go no longer pile up as tiles; a finished run is one click away and its charts stay.
- Flash-Next on a pair of P100 cards now keeps its measured tensor split: Control passes along the engine's own settings for the split it picked, so the seat no longer falls back to the slower layer split without saying so (+7-10% decode).
- Overdrive on Flash-Next now also switches on the fused hyper-connection mixer (bit-identical, about +7%) on builds that do not default it yet.
- The Linux and macOS Control builds now name their cache folder after the real version (for example v3.1) instead of a garbled tagv3.1.
- Control now refuses a card older than Pascal (compute capability below 6.0) with a clear message instead of planning it as 'expect it to work'.
- A PXA_* switch you set in your own shell or service now wins over the value the launcher would have picked for it, and the log says which of yours was kept.
- Model files named in upper or mixed case (.GGUF) are now found by the model scan and accepted when you launch them.
- Flash-Next seats that run from system RAM with Overdrive on no longer have speculative decoding forced off, so the expert cache and the MTP head work together as in the measured Overdrive runs.
- Store > Wallet: Link Discord shows a one-time command so /wallet on the PXA Discord shows your coins and /gift sends coins to a member; Redeem a voucher puts a voucher's coins into your wallet
- New Wayhouse plugin (preview): a project workspace in Control with your own documents and answers that cite them, side-by-side compare of two running servers, and a prompt library, all on your local model.
- Plugins can now have their own tab in PXA Control; Wayhouse opens there. New supporter reward: Wayhouse Plus (watched folders, agent tasks with web research and cited reports, notes), 50 coins.
- Store > Wallet: a Ko-fi code box. Paste the code PXA emailed you after a Ko-fi donation and its coins land in your wallet
- Plugins: Control now serves a plugin's OBS overlay itself, keeps plugin sign-in keys in a per-plugin key store (Forget saved keys on the card), and lets a plugin reach the internet only through a door limited to the hosts you allowed; on Linux such a plugin runs with no other network at all.
- PXA Control on the network now announces itself on the home network (mDNS / Bonjour), so tools on other computers can find it; the access token is never in the announcement
- Router (supporter plugin): a Router tab in Plugins to add the rigs in the house, scan the home network for them (computers running Control on the network announce themselves), see every model they serve, make keys for your tools and watch activity
- PXA Control now says why no cards are listed when the NVIDIA driver answers but reports no GPUs, instead of showing an empty card list.
- Control: new N-gram drafting setting (auto / on / off; auto follows the engine), and the built-in speed test measures first-pass prompts only
- PXA Control shows load progress (GB read, rate, time left) while a big model loads from a slow disk, and no longer calls a busy first load stuck

### Security

- Engine log: an unknown setting in the environment is named without its value.

### Benchmarks

- llama-bench -o json/csv/md/sql: stdout now carries only the results; every status line goes to stderr.

### Performance

- One V100, Qwen3.8 27B: long-prompt prefill is faster again (20.5k-token prompt: about 1025 tokens/s, was 930; 3.1 read 1000).
- CUDA graphs under low free VRAM: graphs that recur every token are kept instead of being re-captured every token (multi-GPU tensor split).
- P100 long-context decode without flash attention is back to full speed with the exact KV band on: about 7% faster than the previous build on Gemma 4 26B at 6k context, same output.
- The overflow guard for MoE models on the f16 cuBLAS route reads each activation column once; the guard's prefill cost on P100 drops from about 1.1% to 0.4% (output unchanged).
- Attention without flash attention: when every row of the attention mask opens at the first cache cell, the exact-band bookkeeping is skipped, with output unchanged (Nano 8B at 6k context on a V100: +3% decode, now level with the band switched off)
- Overdrive: the CPU part of each expert step is quicker; answers are unchanged
- One-card Overdrive on a two-socket box: the host copy of the cold experts now stays on the CPU threads' memory node even when that node is short of free RAM at load (same output; such boots were up to 15-20% slower, about 5% on code in a forced test)

### PXA Control, Store

- The PXA Coins panel shows the Store's live list of ways to earn and offers a one-time report code so a confirmed bug report on GitHub or Discord earns coins in your wallet (4.1, not in 4.0)

### control

- PXA Coins: the wallet panel, the nav badge and the Store cards show one balance and follow the wallet while the Store tab is open (a donation, a confirmed report, a hold or a redeem from another computer appears within a minute); the saved signed statement follows every wallet change; a double redeem says nothing was charged; browser click-through check tests/coins-gui-check.py
- PXA Coins panel: one store read serves every open tab (20 s), the panel backs off while the store is unreachable, a failed join keeps the offline balance, a saved-statement error never breaks the wallet view
- Plugins: Co-host (a supporter reward) reads Twitch and YouTube live chat, answers out loud as a persona of your own with captions in an OBS browser source, and holds back anything the block list or its safety check refuses.
- Plugins: manifests can ask to hear what the computer plays, to accept connections from the home network on named ports, to reach named internet hosts only, and to show a read-only overlay page on this computer.
- Store: unlocks on this computer now refresh on their own about twice a day, when the computer wakes and after an unlock; if the Store says this computer is no longer linked or an unlock was withdrawn, the plugins it covered stop at that sync.
- Plugins: each plugin now runs as its own program next to PXA Control; its page talks to it only through PXA Control's own sign-in checks, it gets only what you allowed it, it restarts if it stops answering, and a supporter plugin stops when its unlock on this computer ends.
- Shared speech service for voice plugins: one warm local process for speech-to-text (incremental), text-to-speech (first clause first) and voice activity detection, one pinned model download for every plugin.
- Plugins tab: the plugins on this computer with signature, consent and unlock state, a consent card in plain words (every permission, tick boxes), Unlock with N coins, and each plugin's own page in a sealed frame with a postMessage bridge
- Android app: when a rig does not answer it shows the rig picker naming that rig with Try again, and reopens the rig by itself once it answers; rig pages can use the microphone after Android asks (https rigs)
- macOS Intel build: a build-mac-control.sh driver, the macOS certificate-bundle fix (https to the Store and Hugging Face works from the compiled app), a smoke test and a zip with start script and notes
- macOS: one universal2 file for Intel and Apple silicon Macs (PyInstaller, fused wheels, every native file checked), with a smoke script that runs on the Mac itself

### Tests

- The switch-resolution test now expects the Gemma 4 single-P100 router default that already ships; no change to what runs
- The speculative call-site test now runs in every test pass on a small built-in model and covers both the shipped lossless acceptance default and the opt-in relaxed rule.
- New test: speculation on vs off gives bit-equal token probabilities at every position

### kernel

- P60 decode stack (35B-A3B PXQN4 on a P100 pair): the one-token MoE half of the persistent decode kernel is now bit-identical to the normal decode path (a unit test on random inputs and a live bitwise check of 800 calls agree to the last bit); the previous kernel stays available as a rollback. Cost: the first exact form read -5.9% decode (118.2 vs 125.4 t/s, not quiet), the tuned form runs at the previous kernel's speed pending its final run. The stack as a whole still differs from the stack-off path through one of its other members; that bisect is owed.

### models

- Flint-PXQN4 (Recursive's 39.8B coding mixture-of-experts model) is on Hugging Face: quality level with the mainline Q4_K_XL file (KLD 0.326 vs 0.321) at 0.18 GB smaller; runs on one 16 GB P100 or V100 with the rest in system RAM.

### bench

- Bench files: the Fusion2-35B PXQ6-MTP file was re-published with cleaned metadata. The tensor data is unchanged; the checksum lists carry the new hash and name the old one as superseded.

### Plugins

- Co-host: the OBS overlay is now served by PXA Control (the plugin opens no port), YouTube sign-in can be remembered in PXA's key store, and all chat traffic goes through Control's limited internet door.
- Interpreter, a supporter reward in the Plugins tab (prototype): two people who speak different languages talk through one computer; it hears each of them, translates with the model you already run, and speaks the other language, all on your own machine.
- Scribe (supporter reward): live call notes in the Plugins tab. Your microphone and what the computer plays become text on this computer, then a summary, decisions and action items when you stop; Windows and macOS capture helpers included
- A plugin whose folder was opened in Finder or Explorer no longer refuses to start.
- Updating a running plugin now stops it first and starts it again afterwards, and a second install of the same plugin while one is running is turned away with a clear message instead of failing halfway.

### plugins

- Satellite: Home Assistant voice through the plugin host (shared speech service STT/TTS, model.chat agent), S4 manifest validator wired, 4 connections per device, real-HA container bench (NOT-QUIET: STT +1.1 s win, TTS loss), signed .pxp + hidden licd row
- Dictate 0.2.0: hold a key (F7-F12), speak, and the words are typed into the window you are using; speech stays on this computer (PXA Control's speech engine). Windows 10/11 and Linux X11.
- Router and Co-host now install on PXA Control 4.0 (min_host 4.0).

### CUDA

- Turing (sm_75) support: the default CUDA build includes sm_75, Flash-Next PXQN batch sizes for 22 GB Turing and V100 cards, the launcher reads per-card memory when a Turing card is present, and a Turing setup guide (community contribution)
- Fixed: on RTX 30xx / 40xx and newer cards (where graph replay is on by default) a replayed decode graph could reuse stale kernel arguments when only a small per-op setting changed between steps (pad offsets, clamp bounds, norm epsilon); the replay now checks those settings too
- Flash attention on RTX 20xx/30xx/40xx (sm_75+): fixed a long-prompt attention error on sliding-window models (Gemma 3/4, gpt-oss and similar) and on large prefill batches, where part of the attention could be counted twice. Output is unchanged wherever the bug was not hit; Pascal and Volta cards were never affected.

### Correctness

- Speculative decoding on Pascal: draft-verify attention now matches plain decoding bit for bit when a verify step crosses a KV-cache padding boundary at short context.

### Install

- The one-command installer now accepts Turing cards (compute capability 7.5: T4, RTX 20xx, GTX 16xx), which the release builds now carry kernels for; before, a host with only Turing cards was refused.

### Store

- Store rewards that are add-ons now arrive as a verified download after you unlock them with coins; they install, update and remove from PXA Control, and new rewards appear without a new release.
- Plugin packages are checked on the store server before they are listed, and a busy store server no longer uses up a download link.
- The store is live with plugins: redeem one with coins in PXA Control and it installs, checks its files and starts, on Linux and Windows.

### Windows

- Windows CUDA engine build for Pascal and Volta (sm 60/61/70) with the CUDA 12 and Microsoft C++ runtimes inside; Windows CPU tests 88/88
- Windows RC2 engine package (Pascal/Volta) passes the release checks; Windows tests 98/98

### Speculative decoding

- Qwen 3.x 27B with MTP on two P100s now produces exactly the same greedy text as plain decoding (code prompts used to drift after a few hundred tokens); speed unchanged.
- Gemma 4 with its assistant drafter: long-prompt decode on one P100 is about 4% faster and now matches decoding with no drafter loaded.

### server

- When the MTP draft context does not fit next to the model (Qwen3.6-35B-A3B MTP on one 16 GB V100 at -ngl 30 -c 4096), a launch with --spec-type mtp now starts and decodes without speculation, with one warning that says how much memory was free and what to change, instead of exiting; the output equals a launch without speculation.
- GLM-5.3: turning thinking off now works. A request with enable_thinking=false (or a server started with reasoning off) gets the answer as content instead of a reasoning trace with an empty reply; thinking on, the default, is unchanged.
- First start on a GPU without native kernels for its architecture (for example H100, B200 or RTX 50) no longer gets the server stopped by the stall watchdog while the driver compiles the GPU kernels (a cold first request took 266 s on the test card and now completes); the launcher and PXA Control also give the driver a 4 GiB kernel cache so later starts skip the compile (first token 0.8 s warm).

### Launcher and PXA Control

- A card that drives your display is no longer refused as busy because of the desktop: on Windows (and on Linux with an active display) the desktop's share of the card is allowed and the model is sized from the memory that is actually free; a card running another compute program is still refused, and the message now says which program holds it and how to start anyway (tick Allow busy cards in PXA Control, or --allow-busy).

### Docs

- User docs follow the 4.0 release: install steps for Linux, Docker, Windows and Mac (SmartScreen and Gatekeeper first-start notes), Control screenshots retaken on the 4.0 interface, the Compose file and tutorials point at the 4.0 image, the model quick start says Super is a PXQN2-only text model, and the known-issues page leads with the 4.0 limits.

### encoder

- Encoding mixture-of-experts models: one expert with out-of-range calibration values no longer aborts the encode; that expert alone gets a safe fallback, with a warning. Encodes that never hit this case are byte-identical.
- Encoding: out-of-range values in the calibration data no longer abort an encode; the affected tensor gets a safe fallback, with a warning. Encodes that never hit this case are byte-identical.

### speculation

- Qwen3.8-27B with its MTP head on two V100 under -sm tensor: faster speculative decoding by default, ahead of v3.1 on the same cards in prose, code and chat, greedy output unchanged

### launch

- pxa-launch: Stop now ends a seat's process when the seat has no command (stop by pid, SIGTERM then SIGKILL); a failed stop is shown and printed, not swallowed

### CPU

- MoE models with PXQN experts kept in system RAM (-ot exps=CPU, --cpu-moe) process prompts faster on the CPU, with bit-identical output (greedy 512-token output unchanged).

### updater

- pxa-update now accepts the engine tarball named after the GPU architectures it is built for (for example -sm60_61_70_75_80_86_89) and prefers it over the older -sm60_61_70 name when a release carries both; only the chosen file and its own checksum are downloaded.

### Fixed

- The test suite builds and passes on native Windows again, and the CUDA prefill integer-GEMM test now runs in the CUDA test set on every card (its RTX 30 series route check skips cleanly on older cards).

### Overdrive

- Overdrive on V100: new option to make experts computed on the CPU give exactly the card's results, so answers no longer depend on where an expert runs; on by itself only when experts move between card and CPU during a run, since it costs some CPU time

### engine/xcache

- Single-card Overdrive on Flash-Next can keep about 1,000 more experts resident by lending the idle prefill buffer to the expert cache (lever, off by default; prose decode +10.9% in a first probe).

## [4.0.0] - (date TBD)

> ## PXA 4.0
>
> - **+42% over a popular community fork on a 27B, same P100 pair:** 26.67 vs 18.78 tok/s tg128.
> - **35B-A3B MoE with MTP on two P100: 138.1 tok/s greedy,** +4.5% over the fork in the same bracket and +38.6% over our previous MTP build.
> - **16k-prompt prefill on the MoE: 1803 tok/s with the opt-in long-prompt preset,** +23% over the fork (NOT-QUIET, fork in the earlier bracket).
> - **28k prompts on one V100 (Flash-Next): 324 -> 597 tok/s,** 1.84x, from an automatic ubatch choice.
> - **27B with MTP on one V100: +14.8% prose / +38.4% code over 3.1** (stock file, greedy, same box).
> - **Flash-Next presets +7.2% prose / +5.9% code on 2x V100,** bit-identical.
> - **One V100 on Flash-Next Overdrive: +19.5% prose / +31.5% code** (38.8 / 47.5 vs 32.4 / 36.1 tok/s; quiet rerun owed).
> - **PXA Control now ships a Windows exe, a macOS Intel build and an Android client,** with an agentic Assistant chat covered by a 152-case test suite.

Numbers are same-box measurements. Most arms ran NOT-QUIET (box load 10-50) and are lower bounds.

### What's new

#### Pascal (sm_60)

- **35B-A3B decode on two P100 goes from 75.9 to 133.4 tok/s** (tg128, +76%), on by default (75.9 is our earlier build; v3.1 cannot run this model).
- **PXQN4 files get the same P100 decode speedup** for MoE, gated DeltaNet and attention layers.
- **MTP verify on P100 is 38.6% faster** (99.6 -> 138.1 tok/s greedy). On by default.
- **A further opt-in verify speedup adds +4.4% / +3.6%** over the fork on the MTP board cell.
- **One-card 27B gains +5.3% prose / +5.2% code over 3.1** (25.18 -> 26.53 and 25.08 -> 26.40 tok/s, same box, bracket and prompts), identical output.
- **Single-token PXQN4 decode on P100 reads 17% more bytes per second** (ffn_down 0.1181 -> 0.1008 ms), byte-identical; on by default for one P100 with a dense 27B PXQN4 file (see Engine kernels, Pascal).
- **MoE prompt processing on P100 is faster:** pp512 440 -> 1028 on the fork's own file in the reproduction. Opt-in.
- **Same-box yardstick:** a popular community fork's P100 patch set was reproduced on our cards (tg128 65.8 -> 119.6) so every comparison above is same-bracket.

#### Volta (sm_70)

- **27B MTP verify is up to 1.46x at width 8** (pp8 155 -> 226 tok/s in the faster opt-in mode; the bit-identical opt-in mode gives 164). Opt-in, widths >= 4.
- **Plain 27B decode on one V100 is +3.0%** (40.3 -> 40.9 tok/s, identical output); the card is GPU-bound now.
- **MTP on V100 is +1.2% faster** from a lower per-token host cost.
- **Single-token decode on V100 is +0.8% faster (tg128)** by default, bit-identical.
- **One V100 decodes +1.5% faster** now that an old per-card default that turned a decode speedup off is gone.
- **Overdrive on V100: prose +20.5% in the close bracket** with an opt-in setting (loaded box).
- **A Flash-Next speedup is in every preset:** 2x V100 layer split +7.2% prose / +5.9% code, bit-identical.

#### Speculation / MTP

- **Depth presets per context** raise 27B 1x V100 8k prose from 57.5 to 62.6 tok/s (+9.1%): `mtp:n_max=4,p_min=0.5`. 16k uses a ramp; 32k n3 is gated-owed.
- **Short-context 27B on a V100 reaches 62.4 prose / 84.4 code** (+5% / +29% vs depth 1) at depth 4, p_min .5.
- **Flash-Next prose stops locking at a shallow depth** with a confidence floor: `mtp:n_max=3,p_min=0.75`.
- **Identical text across restarts is possible on 1x P100:** 24/24 identical over two boots with the determinism setup.
- **35B-A3B MTP acceptance is .774** with det 12/12 and needle 16k + 32k passing.
- **An opt-in Overdrive mode trades output for speed;** it is off by default.

#### Overdrive / Flash-Next

- **A learned expert map makes the cache faster from the start:** 28.50 -> 32.59 prose, 31.59 -> 35.07 code on one P100; the curated map beside the model is worth +16% / +25%.
- **Deterministic Overdrive works:** with the determinism setup, 12/12 and 24/24 identical texts on V100 with no speed cost.
- **The 32 GB file gets smaller and faster:** the per-layer table shrinks by 19 / 29 GB (KLD +0.0043), and 2x V100 is +20.1% faster.
- **28k prompts on one V100 prefill 1.84x faster** (324 -> 597 tok/s): the server picks `-ub 4096` when one card streams experts; default on.
- **Two Overdrive speedups add +6.6% code and +6.0% code.**
- **The one-P100 preset is confirmed by A/B:** 29.52 / 37.81 vs 26.29 / 32.88 tok/s on card 0.
- **Control picks the measured Overdrive setup** by reading the Flash-Next header.
- **Less RAM pressure:** an opt-in direct-read mode for the per-layer table, and the launcher guards against a rotational disk for it.
- **An opt-in thread-placement setting** pins the decode and cold-path threads.
- **16 GB cards boot reliably:** q8_0 KV, no `-b 8192`, and expert maps are found under every file name.

#### Tensor split

- **Flash-Next runs under `-sm tensor` on two 16 GB cards** (it ran out of memory at boot before): 2x P100 MTP prose 36.85 vs 33.98 (+8.4%), code +5.3%.
- **The launcher picks the split for you:** a P100 pair gets `-sm tensor` with its split settings; a V100 pair stays on layer, where it ties.
- **A Flash-Next speedup on both cards** adds +12.7% prose / +18.4% code over before; bit-identical.
- **Planner tests cover uneven pairs and 4-way splits.**

#### Long prefill

- **35B-A3B prefill on two P100 reaches 1803 tok/s at 16k** (+23% over the fork; the arm without the lean attention gave 1730 in a closer bracket). NOT-QUIET.
- **Long-prompt prefill is faster on a P100 pair:** pp4096 1595 -> 1953, pp16384 1257 -> 1730 (+22% / +38%); on by default for Qwen3.5-MoE on all-P100 cards.
- **Lean long-prompt attention adds +5.2% at 16k** (29.7 vs 36.1 ms per attention call); engages from 8192 tokens of context. On by default for Qwen3.5/3.6 MoE PXQN4 on two P100s; one setting turns it off.
- **A second attention setting adds +4.4% at 4k, +11.7% at 16k;** on by default in the same two-P100 cell (one setting turns it off).
- **Flash-Next 28k prefill is 1.84x** with decode unchanged and needle 28k passing.

#### Quantization / PXQN

- **PXQN3 and PXQN2 are 2.5x and 3.0x faster than a community quant at the same bytes** (29.39 vs 11.59, 34.26 vs 11.54 tok/s) with lower KLD at 3 bits (0.0364 vs 0.0631).
- **PXQN4 beats uniform PXQ4 at equal bytes:** 25.5 vs 20.1 tok/s on one P100.
- **PXQN4 files for the 27B and the 35B-A3B MTP file use the newer encode path by default** (19.52 GB, boots 42/42 on two P100).
- **gguf-py reads every PXQN type again** (830 failures -> 0 on two real files).
- **The encoder writes routing counts beside every MoE file.**
- **PXQN revision 2 files** are supported by the closed library.

#### Models

- **Qwen3.6-35B-A3B (qwen35moe) with its MTP block is supported** end to end: encode, boot, 138 tok/s on two P100.
- **Flash-Next fits more hardware:** one V100 or P100 with RAM expert streaming, two 16 GB cards under tensor split.
- **Nemotron, Nemotron-H and Nemotron-H MoE load** on CPU and CUDA (ported from mainline llama.cpp).
- **GLM-5.3-Flash has an N2 file;** no 4.0 speed number yet.
- **Ornith 1.5 35B-A3B uses the same MoE paths** as the Qwen3.6 MoE.

#### PXA Control

- **One download runs everything:** a compiled single file (18.8 MB, 0.5 s start) and a Linux `.run` that carries the engine.
- **The UI is rebuilt:** Home for first run / ready / serving, grouped navigation, Plugins and Store pages, tier badges.
- **Start tells you why it refuses,** using the launcher's own reason.
- **The launcher reads the box:** sockets, NUMA, each card's node, the CPU pool; every decision is named in the plan.
- **The Assistant is an agent:** tools, sub-agents (`spawn_agent`), memory across chats with incognito, recall, `@` references, auto-compaction on exact token counts.
- **A 152-case suite over Control's HTTP API tests the agent loop,** found three fixes, plus 28 unit tests.
- **Chat renders math (KaTeX), switches between edited message versions,** and fits a phone.
- **The licensed library updates by itself** as its own signed, rollback-able artifact, shown beside the encoder's.
- **History is kept on disk** and Home Assistant MQTT is opt-in.
- **Hardening H1-H10 ships** (loopback session protection, CSRF and rebinding guards, request limits, content-security policy).
- **One-click model downloads, the live header names every server,** and the log shows its backlog.

#### Platforms

- **Windows x64 Control exe and engine build:** the engine is `llama-server.exe`, the PXQN library is `ggml-pxqn.dll`. Real-GPU run owed.
- **macOS Intel Control build script,** with docs that state only what was verified.
- **Android client APK** (a WebView wrapper that remotes to your Control host; reproducible command-line build; API 34 emulator smoke test).
- **Installable web app (PWA)** with manifest, shell service worker, host picker, icons.
- **Container fixes:** the cache volume is owned by the `pxq` user and the server runs as pid 1.

#### Licensing / store / sidecar

- **The PXA Store runs in licd:** credits, vouchers, signed catalogue, machine-bound grants, grant-bound downloads. Live since 10-14.
- **A Ko-fi donation becomes a voucher (Store credits)** end to end in the local test (store 21/21, base licd 100/100 tests).
- **Control talks to the real store:** catalogue, link, voucher, redeem, refresh, unlink, supporter-model downloads.
- **The vLLM sidecar ships in the release tarball** and runs without Docker.
- **The sidecar serves PXQN4/PXQN5/PXQN3:** V100 TP2 PXQN4 48.0 tok/s; P100 TP2 +8.6% over vLLM PXQ4; PXQN5 top-1 99.84% vs the engine.
- **`pxa-update` fails closed,** needs the exact tarball name, stages in the install directory and rolls back.
- **The release package carries** the library updater and its verifier, the expert-map tool, the Grafana dashboard and the Overdrive preset.

### Upgrade notes

**Defaults that changed**
- One card that streams MoE experts now picks `-ub 4096`.
- One V100 keeps a decode speedup that an earlier per-card default turned off.
- Flash-Next on a P100 pair starts with `-sm tensor`; a V100 pair stays on layer.
- P100 MoE decode and MTP verify settings are on by default.
- Every Flash-Next preset includes a bit-identical speedup.
- A bit-identical single-token decode speedup is on by default. Every default above can be turned off; docs/LEVERS.md lists the master switches and the documented off switches.

**Levers that need a switch (all off by default)**
- Volta: two faster MTP verify modes (one exact, one faster) and the Overdrive verify graph.
- Prefill: long-prompt attention and prefill settings.
- Speculation: rollback, depth cutoff and confidence settings, and a determinism setup for identical text.
- Overdrive: a direct-read mode, thread placement and a skip mode that changes output.

**Behaviour changes**
- Plain-greedy device-side argmax is opt-in (off by default).
- Texts can differ from 3.1 at near-ties because accumulation order changed; identical text across restarts needs the determinism setup.
- Two unused Overdrive cache settings are removed.

**Known issues**
- Flash-Next is not supported on the vLLM sidecar.
- Windows: the PXQN DLL builds but a real-GPU run is owed; no code signing.
- macOS: Intel only; no Apple Silicon engine or Control build.
- Android: remote client only; no on-device engine.
- Gates owed: the 32k n3 speculation gate; 35B-A3B v3.1 cell and KLD; quiet rerun of the V100 1x Overdrive headline; the same-bracket 3.1 pair for 27B plain decode on one V100.
- Four-P100 Flash-Next greedy text is not stable across boots with speculation on (use `--spec-type none`); V100 logits depend slightly on the KV cell offset.
- GLM-5.3-Flash: no 4.0 speed number yet; a short-prompt NaN on multi-backend execution is narrowed but open; speculative decoding at a draft depth of 4 can still crash with a heap corruption (open).

## [3.1.x] - 2026-10-08 .. 2026-10-12

Anchor: tag `v3.1-rc1` (2026-10-08). The 91 commits after it are the 3.1 patch line (release notes, packaging, Control updater, hot swap, chat web search, adaptive expert map, learned counts, Overdrive preset). 4.0 builds on top; details are in docs/RELEASE-3.1-HANDOFF.md.

- Server sleeps when idle and reloads on the next request (`--sleep-idle-seconds`); hot swap keeps a continuation, refuses sliding-window models, restores shift counters.
- Control chat gets built-in web search (SearXNG / Brave / Tavily / reader), host access (off by default, allowlisted, audited).
- Adaptive expert map with stop-when-flat, learned counts saved without touching the curated file, docs for the one-P100 Overdrive recipe.
- Signed, rollback-able licensed-library updater; `pxa-update` fails closed.
- Release image and tarball fixes (Grafana dashboard, llguidance build, entrypoint pid 1).
