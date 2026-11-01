# Writing a PXA plugin: the author's guide

This guide is for people who write tool packs: small programs that add tools to Anibal. It explains what goes in a
pack, how it is signed, what it may ask for, how it is sandboxed, what the owner sees before saying yes, and the rules
for signals. Everything here follows the code in `tools/pxa_plugins/` (manifest checks: `manifest.py`; packages:
`package.py`; sandbox: `sandbox.py`, `seccomp.py`; signals: `safe_kill.py`). If this guide and the code ever
disagree, the code wins. Section 8 has a complete example pack; a test loads it with the real loader.

## 1. What a pack is

A pack is a folder with a `plugin.json` manifest and your code. Anibal starts it as a separate process inside a
sandbox and talks to it over stdin/stdout, one JSON message per line. The SDK (`tools/pxa_plugins/sdk/pxa_plugin_sdk.py`,
standard library only) handles that protocol for you. Copy it into your pack's `py/` folder.

```
mypack/
  plugin.json
  py/pack.py
  py/pxa_plugin_sdk.py
```

## 2. The manifest (plugin.json)

`plugin.json` is JSON, at most 64 KiB. These fields are required:

| field | rule |
|---|---|
| `schema` | must be `1` |
| `plugin_id` | lower-case letters, digits and dashes |
| `name` | 1 to 64 characters |
| `version` | like `1.2.3` (an optional `-suffix` or `+suffix` is allowed) |
| `publisher` | your publisher name: lower-case, starts with a letter, 3 to 32 characters of letters, digits and dashes. `pxa` is reserved |
| `edition` | `free`, `paid` or `supporter`. Third-party packs must be `free` |
| `min_host` | like `1.0` |
| `entry` | how to start the pack (below) |

Optional fields that matter to a tool pack:

| field | rule |
|---|---|
| `kind` | `toolpack` for you (third parties may only ship tool packs). Default `app` is first-party only |
| `max_host` | like `1.x` or `1.4` |
| `platforms` | a list of strings, default `["any"]` |
| `summary` | free text |
| `entry.toolpack.cmd` | the only entry a tool pack has: a list starting with `{python}`, then paths inside the pack (no `..`, no absolute paths), e.g. `["{python}", "py/pack.py"]` |
| `tools` | 1 to 64 tools. Each: `name` (lower-case letters, digits, `_`; 2 to 41 characters, starting with a letter), `description` (up to 500 characters), `tier` (`T0` to `T3`, default `T0`), `outbound` (true/false), `params` |
| `tools[].params` | at most 28 parameters, each `{"type": "string" or "number" or "integer" or "boolean", "optional": ..., "description": ...}` and nothing else |
| `permissions` | see section 4 |
| `net_hosts` | plain host names your pack may fetch from (needs `net.http`) |
| `egress_domains` | host names or `*.domain.tld` patterns (needs `net.http`; `*.d` covers subdomains only, never `d` itself) |
| `secrets_kinds` | the kinds of stored secrets your tools may use through the host broker (needs `net.http`; owner-only kinds are refused) |
| `tier_overrides` | maps your tool names to a tier; it can only raise a tier |
| `max_tier` | no tool may end up above it |
| `env` | names of environment variables your pack may see (not `PXA_*`, `LD_*` or `PYTHON*`) |
| `consent` | `{"summary", "revoke", "data_stored": [...]}` |

`listen`, `exec` and `browser` are first-party only. If `net.http` is in `permissions`, you need at least one
`net_hosts` or `egress_domains` entry, and no tool may say `outbound: false`.

A tool's effective tier is the highest of: its own `tier`, its `tier_overrides` entry, and the floor its pack's
permissions imply (`fs.workspace.read` T0, `net.http` T1, `fs.workspace.write` T2). Higher tiers make Anibal ask the
user before a call.

## 3. Signed packages

Users install a `.pxp` file. It is a gzipped tar that starts with:

* `MANIFEST.json`: the sha256 of every file in the pack, your key id (`kid`), `plugin_id` and `version`, in canonical
  JSON;
* `MANIFEST.sig`: your Ed25519 signature over it;
* `PUBLISHER.cert`: your publisher certificate.

How you get there:

1. Make an Ed25519 key (a 32-byte seed) and keep it offline.
2. Send PXA your public key and publisher name. PXA returns a publisher certificate, signed with its release key,
   naming your publisher, your key id, your public key, the kinds you may ship (`toolpack`) and an expiry date.
3. Build: `python3 -m pxa_plugins.pack mypack OUT --seed-file me.seed --kid <your kid> --cert me.cert.json`.

On install and on every load the host checks: the certificate chains to a pinned PXA key, has not expired and has
not been withdrawn; the manifest is signed with the key the certificate names; `plugin.json`'s `publisher` is the
certificate's publisher; the pack's kind is one the certificate allows; and every file matches its sha256. Archives
with absolute paths, `..`, links or special files are refused, and size limits apply while unpacking, before the
signature is checked. A withdrawn publisher's packs stop loading even if already installed.

For trying things out, the repository has a DEV publisher key and certificate (`tools/pxa_plugins/keys/`). They are
trusted only when `PXA_PLUGIN_TRUST_DEV_KEY=1` is set. Never ship with them.

## 4. Permissions

Third-party packs may ask for three permissions, and nothing else:

| permission | what the owner reads | what it allows |
|---|---|---|
| `fs.workspace.read` | read files in Anibal's workspace folder | read the workspace (read-only mount or rule) |
| `fs.workspace.write` | change files in Anibal's workspace folder (asks first) | write the workspace; every tool becomes at least T2 |
| `net.http` | fetch web pages from the sites it names, through PXA (no other network) | `ctx.http_get(url)` to your declared hosts, over https, through the host's broker; every tool becomes at least T1 |

Your pack never opens a socket itself. The sandbox has no network; the host fetches for you and only from the hosts
you declared. Secrets you use (via `secrets_kinds`) are filled in by the broker; your code never sees them.

Other permissions (`proc.spawn`, `net.internet`, `host.shell`, `fs.read`, `fs.write` and the rest) are for first-party
packs only. In particular, a third-party pack can not start other programs.

## 5. The sandbox

Which backend runs your pack depends on the machine. On Linux the host picks bubblewrap if it works, else Landlock,
else rlimit. On macOS it uses a Seatbelt profile; on Windows an AppContainer. A third-party pack starts only on a
strong backend: bubblewrap or Landlock with seccomp on Linux, Seatbelt on macOS, AppContainer on Windows. On anything
weaker it is refused (`sandbox_unavailable`), except under a developer test flag.

Every backend also sets resource limits: 1 GiB of address space, 600 s of CPU, 256 MiB per written file, 256 open
files, and a process-count limit. A call times out after 60 s. The environment is clean: `HOME` and `TMPDIR` point to
a private temp folder.

| backend | protects | does not protect |
|---|---|---|
| **bwrap** (bubblewrap) | Own user, PID, network, IPC, UTS and mount namespaces. Read-only `/usr` and system libraries, your pack folder read-only, a private temp folder, the workspace only as declared. No network at all. Seccomp filter on top. `pid_ns` is always true | Whatever you read from the declared workspace, and what the broker fetches for you |
| **landlock** | Kernel file rules: system folders and your pack read/execute, temp folder read/write, workspace as declared, `/proc/self` read. Network cut by Landlock (ABI 4 or newer) or, on older kernels, by a separate network namespace. Seccomp filter on top | The rest of the process list is visible; there is no PID namespace for packs that can not spawn (seccomp limits their signals to themselves instead) |
| **rlimit + seccomp** | Only resource limits, no_new_privs and the seccomp filter (no sockets, no ptrace, no mounts, no namespaces, no kernel keyrings, no bpf, and more) | No file isolation (the pack can read whatever the user can) and no network namespace. Third-party packs never run here outside the developer test flag |

There is no separate "sandbox user" backend.

**Seccomp.** The filter is an allowlist; anything not on it kills the process. It has no `socket`, `connect` or
`bind`. A pack without `proc.spawn` can not fork or exec: after the interpreter starts, `execve` returns EPERM, and
`fork`/`vfork`/process `clone` return EPERM (threads still work). Its `kill` reaches only itself or its own session
group, `tgkill` only its own threads, and `tkill` is refused.

**`pid_ns`.** The sandbox facts report whether your pack had its own PID namespace:

* `true`: it did. Always under bwrap. Under Landlock or rlimit, a first-party pack with `proc.spawn` runs as pid 2
  under a tiny init (pid 1) in a new user + PID namespace. Inside it, a pid can only name the pack's own tree, and
  when the pack exits, everything it started is killed with it.
* `false`: a pack with `proc.spawn` that was explicitly allowed to run without a PID namespace. This only happens
  when the host builds the plan with `allow_no_pidns=True`; the facts then carry a warning.
* `null`: not needed. The pack can not spawn, and seccomp already limits its signals to itself. Every third-party
  pack and every Workshop skill is here, unless the backend is bwrap (then `true`).

**When spawn packs are refused.** A pack with `proc.spawn` under Landlock or rlimit needs unprivileged user
namespaces. If the system does not have them, the pack is refused with a reason that names the missing PID
namespace, unless the plan explicitly accepts running without one (then `pid_ns` is `false` with a warning).

## 6. What the owner sees: the approval card

Before anything with the highest tier (T3) is turned on, the owner gets an approval card. For skills made in the
Skill Workshop the card is described in full in `docs/GALLERY-UI-CONTRACT.md` ("Skill Workshop T3 card"). The fields:

* `skill`, `summary` (`capability`, `description`, `threat`), `tier`, `hard`, `outbound`, `signed`, `local_author`;
* `permissions`: `egress` (hosts), `tools` (name, tier, outbound), `secrets`, `account`;
* `diff`: every file with its path, size, sha256 and text, so the owner can read the exact code;
* `sha`: the hash of exactly the code that was tested;
* `sandbox`: the facts of the sandbox test run: `backend`, `network_isolation`, `landlock_abi`, `egress` (`none` or
  `broker_only`), `seccomp`, `pid_ns`, plus `ok`, `sha`, the test `cases` and `warning`;
* `sandbox_warning`: present only when isolation was weak, shown next to the Approve button. In order, the first
  that applies: a weak backend (files and network not isolated), network not isolated, no seccomp filter, or
  `pid_ns: false` (child processes not confined to the plugin's own process tree);
* `nonce` and `binding`: the owner's yes is bound to a hash of the code `sha`, the nonce and the sandbox facts
  (including `pid_ns`). If any of them changes while the card is waiting, enabling fails as `stale` and the draft has
  to be tested again.

Text you write (names, summaries, descriptions, file paths and contents, test case names and errors) is shown as
untrusted plain text: escaped, never as HTML or links. Write it for a person reading carefully, not for a renderer.

For tool packs, the owner sees your permissions in the plain words from the table in section 4, and each tool's
effective tier decides whether a call asks first.

## 7. Signals: the safe_kill rules

These rules apply to host code and to first-party packs that start helper processes. A third-party pack can not
start processes, and its signals reach only itself anyway.

* Signal only processes you started yourself and recorded at start (`safe_kill.record(proc)`, or
  `record(proc, group=True)` for one you started in its own session). Use `safe_kill(proc, sig)` or
  `safe_killpg(proc, sig)`, never raw `os.kill`/`os.killpg` on a number.
* Never signal `-1`, `0`, a negative pid, process group 1, your own process or your own group. `safe_kill` refuses
  all of these and sends nothing.
* `safe_killpg` signals a group only if you recorded it as yours and its leader still leads it. Never signal a group
  you did not create.
* Once a process has been reaped, its number may belong to someone else. `safe_kill` refuses to signal it.
* In tests, anything that spawns or signals runs only inside `tests/run-in-pidns.sh` (a fresh user + PID namespace
  on a scratch machine).

**Exit codes.** A pack that runs in its own PID namespace (bwrap, or a spawn pack under Landlock/rlimit) is reported
by its init. If it is killed by signal N, its exit status is 128+N (for example 137 for SIGKILL, 159 for SIGSYS from
seccomp), not -N. Check for both if you read exit codes.

## 8. A minimal example pack

A pack with one tool, no permissions. You do not have to type it: from `tools/`, run

```
python3 -m pxa_plugins.pack new shouter [--dir DIR]
```

It writes `DIR/shouter/` (default: the current folder) with exactly the files below plus `py/pxa_plugin_sdk.py`
(the SDK copy). The name must follow the `plugin_id` rule (2-32 lower-case letters, digits and dashes, starting with
a letter); an existing file or folder of that name, a missing `--dir` and paths with `..` are refused, and nothing
is overwritten. The files come from `tools/pxa_plugins/scaffold.py`; `tests/test-pxa-plugins-pack-scaffold.py`
fails if the blocks below differ from it. `tests/test-pxa-plugins-author-guide.py` reads the manifest and pack
blocks, signs them with the DEV key, installs the package and calls the tool through the real host.

`plugin.json`:

```json
{
  "schema": 1,
  "plugin_id": "shouter",
  "name": "Shouter",
  "version": "1.0.0",
  "publisher": "example-dev",
  "edition": "free",
  "kind": "toolpack",
  "min_host": "1.0",
  "entry": {"toolpack": {"cmd": ["{python}", "py/pack.py"]}},
  "tools": [
    {"name": "shout", "description": "Return the text in capitals with an exclamation mark.", "tier": "T0",
     "params": {"text": {"type": "string"}}}
  ],
  "permissions": []
}
```

`py/pack.py`:

```python
from pxa_plugin_sdk import Pack, ToolError

pack = Pack()


@pack.tool("shout")
def shout(ctx, text):
    if not text:
        raise ToolError("Give me some text.")
    return text.upper() + "!"


if __name__ == "__main__":
    pack.run()
```

`tests/test_pack.py`:

```python
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "py"))
import pack  # noqa: E402
from pxa_plugin_sdk import ToolError  # noqa: E402


class ShoutTest(unittest.TestCase):
    def test_shout(self):
        self.assertEqual(pack.shout(None, "hello"), "HELLO!")

    def test_empty_text_is_an_error(self):
        with self.assertRaises(ToolError):
            pack.shout(None, "")


if __name__ == "__main__":
    unittest.main()
```

Run the test with `python3 -m unittest discover tests` inside the pack folder.

Notes: return a string; raise `ToolError("message")` for an error the user should read; never print to stdout (it
is the protocol; use `ctx.log(...)`). Arguments are checked against `params` before your function runs.

## 9. Check a pack before signing: `pack lint`

From `tools/`, run

```
python3 -m pxa_plugins.pack lint DIR
```

It applies the loader's and installer's own rules to the folder and prints one line per problem:
`<level> <file>: <what>; fix: <how>`. It exits 0 when clean (`ok DIR: no problems`), 1 for warnings only and 2 for
errors.

Errors:
- `plugin.json` fails the loader's validation (section 2);
- a third-party pack asks for a permission outside `fs.workspace.read`, `fs.workspace.write` and `net.http`, and
  `proc.spawn` in particular;
- an entry path points outside the pack, or names a file that is missing;
- a link (file or folder) anywhere in the pack;
- a file, the file count or the total size over the installer's limits;
- `py/pxa_plugin_sdk.py` with a different `SDK_VERSION` (the protocol number) from the current SDK.

Warnings:
- no `tests/test_*.py`;
- `py/pxa_plugin_sdk.py` differs from `tools/pxa_plugins/sdk/pxa_plugin_sdk.py` but has the same `SDK_VERSION`
  (copy the current one over it).

`python3 -m pxa_plugins.pack SRC OUT ...` runs the same lint first. On errors it signs nothing; warnings are printed
and the package is still built. `--vigil` lints the staged Core tree too and skips only the warnings: errors block
it like any other pack (permissions are judged by the pack's real publisher, so a first-party `proc.spawn` is fine).
What `pack new` writes lints clean.

The SDK version is not only a lint rule: the host refuses a pack whose bundled `py/pxa_plugin_sdk.py` has a
different `SDK_VERSION`, at install and at every load, with the message "This pack was built for SDK n, this host
speaks SDK m; rebuild with the current SDK." on its card. The host reads that line from the file without running
it (`tools/pxa_plugins/sdk_compat.py`, the one helper lint uses too).
