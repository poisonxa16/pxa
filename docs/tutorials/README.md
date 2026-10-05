# Tutorials

Step-by-step guides for people who want to run a language model on an old NVIDIA card and are
not engineers. Every command on these pages was run on a real machine before it was printed —
except [guide 11](11-switching-models-from-your-app.md), which says so at the top of the page.

**New here? Read [00 — Start here](00-start-here.md), then
[01 — Run your first model](01-run-your-first-model.md).** That is fifteen minutes and you will
have a working chat server.

| | | |
|---|---|---|
| **00** | [Start here](00-start-here.md) | what this is, which cards it is for, what you need installed, tarball or container |
| **01** | [Run your first model](01-run-your-first-model.md) | download, start the server, talk to it from the command line and from a chat app |
| **02** | [Pick settings for your cards](02-pick-settings-for-your-cards.md) | one card, two, four, mixed; the tensor split; context length; how to tell a model fits before you wait for it |
| **03** | [Going faster without changing the answers](03-going-faster.md) | speculative decoding in plain words, what "lossless" honestly means, why the first request is slow |
| **04** | [Quantize your own model](04-quantize-your-own-model.md) | a Hugging Face folder to a PXQ file you can run, and how to check you did not break it |
| **05** | [Quantize for the vLLM sidecar](05-quantize-for-vllm.md) | the same job for the other server, and which server you actually want |
| **06** | [Run the vLLM sidecar](06-run-the-vllm-sidecar.md) | starting it, pointing a client at it, the two flags tool-calling needs |
| **07** | [The whole stack with Docker Compose](07-docker-compose.md) | both servers, one command, three variables |
| **08** | [Long chats and many conversations](08-long-chats.md) | the prompt cache, why follow-ups are instant, the one case that is still slow |
| **09** | [Measure your own card honestly](09-measure-your-card.md) | the three traps I fell into myself, and how to get a number you can defend |
| **10** | [When something goes wrong](10-when-something-goes-wrong.md) | the failures people actually hit, how to turn any new feature off, where to ask |
| **11** | [Switching models from your app](11-switching-models-from-your-app.md) | letting an app pick the model by name, with a swap proxy in front of the server |
| | [Glossary](GLOSSARY.md) | every unfamiliar word, two sentences each |

## How these are written

- Every command is in a copy-paste block, followed by what you should see.
- Every section ends with a plain-words paragraph about what just happened.
- Anything likely to go wrong has an "if it went wrong" box with the two or three failures
  beginners actually hit — not an exhaustive list, the common ones.
- **There are no speed numbers in these guides.** They go stale the moment the engine gets
  faster, and a stale number in a tutorial has you debugging a healthy machine. The current
  measured numbers are in the project README with the exact command and file behind each one.
  Guide 09 shows you how to measure your own.

## Getting help

Discord: <https://discord.gg/EqazvV9tf> — [guide 10](10-when-something-goes-wrong.md) lists the
five things to paste so the first reply is a useful one.

If this saved you buying a new card and you would like to chip in:
<https://ko-fi.com/shatteredrealms1>. Entirely optional, and it gates nothing.
