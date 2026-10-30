# Model quick start (4.0)

One section per model family in 4.0. Start each of them from Control the same way.

1. Open **Home**.
2. If the file is not on this computer yet, download it there, or add the folder and press **Look again**.
3. Pick the file and press **Start**. Leave context, split, cache type, and flash attention on automatic unless a section below says otherwise.
4. Read the fit line. **fits on these cards** means the file and a short cache fit on the cards. **fits with system RAM** means a mixture-of-experts model keeps part of itself in the computer's memory. That is expected for the large ones on a 16 GB card. **tight fit** has little room. **does not fit** needs fewer other programs on the card, a smaller file, or another card. On the Models table the same case shows as **+ RAM** next to the card set.

Free files are PXQ. Supporter files are PXQN, a reward for donating on Ko-fi: they unlock with Store coins, and nobody has to donate to use PXA. Home offers free files first. A locked supporter file shows **Unlock**, which opens the Store. Valued Supporters can also try new builds first on the beta channel: [Beta channel](BETA-4.0.md).

Speeds below were measured on the beta 2 build. The table in the 4.0 release notes is the final one, and where a figure there differs from a figure here, the release notes win. A busy machine reads lower. Compare a slow run with the row for your card and your prompt, then [report it](REPORT-A-PROBLEM.md) if you are clearly under that row.

Speculation in Control is the **MTP** switch on Launch. Turn it on only when the section for that family says the file has a draft block. The Nemotron families do not use a draft block in 4.0.

## Qwen 27B

Qwen3.8 27B. A dense model, and a good first one on a single 16 GB P100 or a single V100.

One card is enough. Two or four P100 cards work too. **Start** uses the cards you used last, or a set the fit check accepts.

Both the free PXQ file and the supporter PXQN file run. Prefer a file that includes a draft block if you want speculation, and leave **MTP** on. If the file has no draft block, leave **MTP** off. The server tells you when the file cannot do it.

Beta 2 build: about 26 tokens per second decoding on one P100. On one V100 with speculation on, about 60 to 90 tokens per second, depending on the prompt (prose, code, or chat).

## Qwen 35B-A3B

Qwen3.6 35B-A3B. A mixture-of-experts model. 3.1 does not run it. 4.0 does.

Two 16 GB P100 cards are the comfortable setup. One 16 GB V100 can run it with **fits with system RAM**. That is slower than two P100 cards.

Download the file that includes the draft block if you want speculation, and leave **MTP** on.

Beta 2 build, two P100, speculation on: about 130 to 151 tokens per second (prose, code, and chat sit in that range). Without speculation, decode on the same pair is about 125 tokens per second. Reading a long prompt is faster than the earlier build on that pair.

## Gemma 4

Gemma 4 26B-A4B. A mixture-of-experts model. One 16 GB card is the setup measured for this release. Context lengths through 64k start on both a P100 and a V100.

It can answer questions about pictures. Put the vision projector file in the same folder as the model. Control finds it. From a terminal, `--no-mmproj` runs text only. In chat, attach a PNG, JPEG, GIF, or WebP, up to four images, each under 1 MB.

Leave **MTP** as you like for a file that has a draft block. Automatic settings are the right start.

Beta 2 build: on one P100, decode is about 77 tokens per second, a bit over a tenth faster than 3.1. On one V100, decode is about the same as 3.1, and reading the prompt is much faster (about half again, and about +55% around 4k to 16k). A 64k prompt is the slow one of that set.

## Nemotron Nano 4B

Nemotron 3 Nano 4B. Small. One P100 is enough, and an 11 GB card can hold it.

The free file is PXQ. The supporter file is PXQN. Prefer PXQN when you have it.

Leave **MTP** off. Nemotron draft blocks are not supported in 4.0.

Beta 2 build, one P100: the supporter file decodes at about 122 tokens per second. The free file decodes at about 100.

## Nemotron Nano 8B

Llama 3.1 Nemotron Nano 8B. A small dense model. One 16 GB card is enough, and a smaller Pascal card can hold a small file.

The 4.0 release notes list its speed in the Nemotron table. If Models does not list a file yet, add the folder that holds your file and press **Look again**, then **Start**.

In Launch, **Thinking** can be auto, on, or off. Auto is the right start. This family can take a detailed thinking mode. Leave **MTP** off, same as the other Nemotron files.

## Nemotron Nano 30B

Nemotron 3 Nano 30B-A3B. A mixture of experts. One 16 GB V100 is the setup the release notes measure. Expect **fits with system RAM**.

It is quickest when the computer's processor is not busy with other work. If the processor is busy it still runs, only slower, and it picks up when the processor is free. You do not set that by hand.

Leave **MTP** off.

Beta 2 build, one 16 GB V100, processor free: about 89 tokens per second on prose and about 88 on code.

## Bolt

Nemotron 3.5 Bolt, 74B-A9B. A large mixture of experts. 4.0 runs it as a text model; its image input is not included.

One 16 GB V100, with part of the model in system memory. Expect **fits with system RAM**. The first reply can take a while because experts are read from the computer's memory.

Use a PXA file, or a file you make with Control's encoder from the BF16 checkpoint. NVIDIA's NVFP4 GGUF of Bolt does not load yet.

Leave **MTP** off.

Beta 2 build, one 16 GB V100: about 22 tokens per second decoding, and about 61 tokens per second while reading the prompt.

## Super

Nemotron 3.5 Super-VL 120B-A12B runs as a text model in 4.0. Its image input is not included.

Super comes as a PXQN2 file only. It is a supporter reward file: you unlock it with Store coins from a Ko-fi donation. It is made for two 16 GB P100 cards with part of the model kept in system RAM, so plan for plenty of free RAM. There is no free PXQ file for Super in 4.0.

Start it from **Home** like the other models and read the fit line before you trust it. Leave **MTP** off. The release notes list its speed in the Nemotron table.

## Flash-Next

A mixture-of-experts model aimed at long prompts. Two P100 cards, or one V100.

Leave **Split** on auto. On a P100 pair the automatic choice is the faster one. On a V100 pair the automatic choice is the other split, and the two splits tie. A file with a draft block can leave **MTP** on.

Beta 2 build, two P100, speculation on: about 41 tokens per second prose and about 47 code. Long prompts on that pair (about 8k, 16k, and 30k) are roughly a fifth to a quarter faster than 3.1. On one V100, a prompt of about 28k tokens is read much faster than in 3.1 (about 600 tokens per second, against about 320 before).

## GLM (experimental)

GLM-5.3-Flash. The 4.0 release notes say whether it is supported or still experimental. Until they say supported, treat a strange answer or a failed short prompt as something to report. A short-prompt failure on more than one device is a known open item.

It is a large mixture-of-experts model. The beta 2 build measured about 16 tokens per second of decode on six cards, on a busy machine. There is no 3.1 number to compare. Fewer cards can start it when the fit line says **fits** or **fits with system RAM**, and it will be slower than that six-card figure.

Leave **MTP** off while you are only checking that it runs.

If it crashes, answers wrongly, or is far under that preview, send a report with the card count, the file name, and **4.0**. See [Report a problem](REPORT-A-PROBLEM.md).

## Which one to try first

| You have | Start with |
|---|---|
| One 16 GB P100 or V100 | Qwen 27B, or Nemotron Nano 4B if you want a small fast model |
| Two P100 cards | Qwen 35B-A3B, or Flash-Next if the chat is long |
| One 16 GB V100 and you want a large mixture of experts | Qwen 35B-A3B, Nemotron Nano 30B, or Bolt |
| Pictures | Gemma 4, with the projector file beside the model |
| A reason to test the newest model | GLM-5.3-Flash, and report what happens |

The Control pages are in [PXA Control 4.0](CONTROL-GUIDE-4.0.md).
