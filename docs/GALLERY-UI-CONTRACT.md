# Connector gallery: Control UI contract (server side: plugins/gallery-control 24e017c270)

The server API lives in tools/pxa_plugins/gallery_api.py. The page belongs to UX (ui/*): this file is the contract, and no UI file was changed.
Auth: these are ordinary Control ROUTES, so they get the same Host check, the same LAN token (cookie, X-PXA-Token or
Bearer) and the same rule that a POST must come from the same origin as every other /api/plugins/* route.

## GET /api/plugins/gallery
```json
{"ok": true,
 "gallery": {"state": "ok|empty|refused", "why": null | "missing|unsigned|bad_signature|unknown_kid|bad_entry|...",
             "issued": 7, "kid": "..."},
 "entries": [{
   "connector": "github", "sha": "<64 hex>", "tier": "T3", "hard": true, "outbound": true,
   "transport": "http|stdio", "package": "@scope/pkg@1.2.3" | null,
   "hosts": ["api.githubcopilot.com"],
   "secrets": [{"kind": "api_token", "name": "github"}],
   "name": SHOW, "notes": SHOW, "upstream": SHOW, "license": SHOW,
   "tools": [{"name": "search_issues", "tier": "T1", "floored": false, "description": SHOW}],
   "status": "available|pending|installed|disabled|needs_consent"}]}
```
SHOW = `{"untrusted": true, "text": "<HTML-escaped, control/bidi chars removed, capped>", "fenced": "「text」", "truncated": bool}`.
If the gallery is unsigned, tampered, signed with an unknown key or missing, `entries` is `[]` and `gallery.state` explains why. The UI should say
"No connectors available" and must not show `why` as if it were an error.

## POST /api/plugins/gallery/install-request
Request: `{"connector": "github", "sha": "<sha from GET>", "workspace": "/path"?}` (`workspace` is only for entries
whose command has a {workspace} slot).
Response: `{"ok": true, "card": "gal-…", "connector": "github", "sha": "…", "state": "pending"}`.
Errors (`ok: false`, `error`):
- `bad_request`: the body carries approve, approved, source, card_hash, approval_id or id. Approval never travels in this request.
- `unknown`: the entry is not in the signed gallery.
- `stale`: the sha is not the entry's current sha; reload the gallery.
- `approval_unavailable`: the owner card path is not connected, for example because Core's vault is not open.

## The card (the existing owner card UI)
The request puts a card on the existing desk. It shows up in `GET /api/plugins/cards` like every other card and is
answered with `POST /api/plugins/cards/answer {"card": <desk id>, "approve": true|false}`, from this machine only and
once per card. To match it to the request, use `kind == "mcp_install"` and `request == <card from install-request>`.
Card fields:
- `kind`, `tier`, `hard`, `outbound`, `connector`, `sha`, `request`, `hosts`, `tools`, `secrets`, `transport`, `package`
- `name` and `notes` (SHOW)
- `title` ("Connect <escaped name>?")
- `rows` ([[label, value]…]: connector, the hosts it talks to, the highest tier, one row per tool with its tier, the secrets it needs, the package)

The current plugins.js renders `title` plus `rows` as plain text. With hard == true (a T3 tool), the UI should ask for
deliberate confirmation. The install runs only when the answer is exactly approved == true, from ui, passkey or cli,
for this sha. If the gallery changed while the card was open, the request ends as `stale` and nothing is written.
No answer within the card timeout counts as a decline.

## GET /api/plugins/gallery/status[?request=gal-…]
`{"ok": true, "connectors": {"github": "disabled", ...}, "request": {"connector", "sha", "state", "error"?}}`
The request state is one of pending, installed, declined, stale, failed or unknown. Poll every 1–2 s while it is pending.
What each connector status means:
- available: not installed.
- pending: a card is open.
- disabled: installed, but `x-pxa.enabled` is not true. A fresh install is always written disabled.
- installed: installed and enabled.
- needs_consent: the entry changed since consent, the vetted tool pins differ, drift was flagged, or the config no
  longer runs the signed entry. Offer "Review again", which is a new install-request with the new sha.

## Untrusted fields
`name`, `notes`, `upstream`, `license` and `tools[].description` are untrusted. Use only `.text` (already escaped), insert it with
textContent, never innerHTML, and never decode the entities. Label it as text from the connector's publisher.
Everything else is validated by the signed-gallery loader: `connector` and tool names match [A-Za-z0-9_-], hosts are
validated host names, tiers are T0–T3, package names are pinned specs, secret kind/name pairs match a regex, and sha is hex.
Even so, render them as text. Secret values are never returned. `why`, `error` and `state` are fixed codes.

# Skill Workshop T3 card (server side: plugins/skill-workshop c68253999e)

Produced by `DraftShop.card(name)` (`tools/pxa_plugins/skills/drafts.py`) and handed to the owner's approve callback by
`DraftShop.enable(name, sha, approve)`. The examples below are the fixtures in `tests/fixtures/workshop_card/`, built
from the real code path; `tests/test-pxa-plugins-workshop-card-contract.py` fails if the code, the fixtures or this
section drift apart. In the examples, hashes, nonces and timestamps are placeholders (`<SHA256>`, `<NONCE>`,
`<BINDING>`, `<TIMESTAMP>`); real values are lower-case hex (sha256: 64 chars, nonce: 32 chars).

## GET /api/plugins/workshop/card?name=<draft>

Read-only. It returns the draft's current card, `{"ok": true, "card": {...}}` (the shape below). It never tests or
enables anything. Approval stays on `enable()` and the owner's desk. Auth is the same as every /api/plugins/* route
(Host check, LAN token). Errors (`ok: false`, `error`):
* `bad_request`: no name.
* `unknown_draft`.
* `not_tested` / `stale`: show the UI copy below, offer Test.
* `workshop_unavailable`: no Skill Workshop in this Control.

Cards parked on the owner desk (`GET /api/plugins/cards`) keep `sandbox` and `sandbox_warning`. They never carry
`nonce`, `binding` or the code diff.

## card.sandbox

The sandbox the draft's test run actually used: `sandbox.describe()` recorded at the start of that run (not re-read
when the card is built).

| field | type | meaning |
|---|---|---|
| `backend` | string | `bwrap`, `landlock`, `seatbelt`, `appcontainer` (strong) or `rlimit` (weak, only with the test flag) |
| `network_isolation` | bool | the sandbox itself cuts the network (true) or not (false) |
| `landlock_abi` | int or null | Landlock ABI when backend is landlock (net rules need ABI >= 4), else null |
| `egress` | string | `none` (no network from the skill) or `broker_only` (only the broker limits egress) |
| `seccomp` | bool | a seccomp filter was applied |
| `pid_ns` | bool or null | the run had its own PID namespace (true); false = a process-spawning pack accepted without one (warning); null = not needed (skills never spawn: seccomp limits their signals to themselves) |
| `ok` | bool | the sandbox test run passed (always true on a card: no card without a passing run) |
| `sha` | string | sha256 of the draft folder the run tested (equals `card.sha`) |
| `cases` | list of `{name: string, ok: bool, error: string or null}` | the fixture cases of the run |
| `warning` | string or null | same text as `card.sandbox_warning`, null when isolation is strong |

## card.sandbox_warning

Present (and equal to `card.sandbox.warning`) only when isolation is weak; absent otherwise. Show it prominently
next to the Approve button (not hidden in details). Exact texts, first rule that matches:

1. backend not in bwrap/landlock/seatbelt/appcontainer: `Network not isolated by the sandbox; only the broker limits egress. Files are not isolated either (backend rlimit).`
2. `network_isolation` is not true or `egress` is not `none`: `Network not isolated by the sandbox; only the broker limits egress.`
3. strong network isolation but `seccomp` false (the seccomp note): `No seccomp filter in this sandbox run.`
4. `pid_ns` is false (a process-spawning pack accepted without its own PID namespace): `Child processes are not confined to the plugin's own process tree (no PID namespace).`

## card.binding

`binding` = sha256 over canonical JSON of `{sha, nonce, sandbox: {backend, network_isolation, landlock_abi, egress,
seccomp, pid_ns}}` (`drafts.card_binding`). The owner's yes covers exactly that code, that card and those sandbox facts. The
UI does not compute or check it; it must pass the card back unchanged. If the recorded facts change after the card
is shown, `enable()` refuses with `stale`.

## enable() result

| status | meaning | UI |
|---|---|---|
| `ok` | enabled; `{state: "enabled", enabled: true, approved_sha}` | "Enabled" with the skill name |
| `stale` | the code, the card or the sandbox facts changed while waiting (`DraftError` code `stale`) | "This draft changed after you looked at it. Test it again to get a new card." No retry of the same card. |
| `not_tested` | no passing sandbox run, or the run recorded no sandbox (`card()` refuses; there is no card) | "Run the sandbox test first." Offer Test, not Approve. |

Other refusals: `denied` (owner said no), `exists` (already enabled / name taken), `unknown_draft`.
The `message` field is server text for logs; show the UI copy above, not the message.

## Untrusted fields (escape; never HTML, never links)

Written by the drafting model or the skill author, so untrusted: `skill`, `summary.capability`,
`summary.description`, `summary.threat`, `permissions.egress[]`, `permissions.tools[].name`, `diff.files[].path`,
`diff.files[].text` (show as plain preformatted text, no highlighting that executes), `sandbox.cases[].name`,
`sandbox.cases[].error`. Unlike the gallery's SHOW objects these are raw strings: HTML-escape them, strip control
and bidi characters, and insert with textContent. Server-generated: `kind`, `tier`, the booleans, hashes, `nonce`,
`binding`, `sandbox.backend/egress/landlock_abi` and `sandbox_warning` (still insert as text, never as HTML).

## Examples

### strong.json (landlock ABI 4, network isolated, no warning; enable ok)

```json
{
  "card": {
    "binding": "<BINDING>",
    "diff": {
      "base": null,
      "files": [
        {
          "bytes": 265,
          "path": "SKILL.md",
          "sha256": "<SHA256>",
          "text": "---\nname: weather-now\nversion: 0.1.0\ndescription: weather-now (made in the Skill Workshop).\negress: [api.example.com]\nentry: weather_now:TOOLS\ntools:\n  - {name: weather_now, tier: T2, outbound: true}\n---\n# weather-now\n\nDescribe when the model should use each tool.\n"
        },
        {
          "bytes": 275,
          "path": "fixtures.json",
          "sha256": "<SHA256>",
          "text": "{\n \"cases\": [\n  {\n   \"name\": \"now\",\n   \"tool\": \"weather_now\",\n   \"args\": {},\n   \"reads\": {\n    \"https://api.example.com/weather\": {\n     \"status\": 200,\n     \"body\": \"{\\\"t\\\": 21}\"\n    }\n   },\n   \"expect\": {\n    \"would_send\": 0,\n    \"result\": {\n     \"t\": 21\n    }\n   }\n  }\n ]\n}"
        },
        {
          "bytes": 211,
          "path": "weather_now.py",
          "sha256": "<SHA256>",
          "text": "\"\"\"weather\"\"\"\nimport json\n\n\ndef weather_now(call, **args):\n    r = call.http(\"GET\", \"https://api.example.com/weather\", {}, None)\n    return json.loads(r[\"body\"] or \"null\")\n\n\nTOOLS = {\"weather_now\": weather_now}\n"
        }
      ]
    },
    "hard": true,
    "kind": "skill_draft_enable",
    "local_author": true,
    "nonce": "<NONCE>",
    "outbound": true,
    "permissions": {
      "account": null,
      "egress": [
        "api.example.com"
      ],
      "secrets": [],
      "tools": [
        {
          "name": "weather_now",
          "outbound": true,
          "tier": "T2"
        }
      ]
    },
    "sandbox": {
      "backend": "landlock",
      "cases": [
        {
          "error": null,
          "name": "now",
          "ok": true
        }
      ],
      "egress": "none",
      "landlock_abi": 4,
      "network_isolation": true,
      "ok": true,
      "pid_ns": null,
      "seccomp": true,
      "sha": "<SHA256>",
      "warning": null
    },
    "sha": "<SHA256>",
    "signed": false,
    "skill": "weather-now",
    "summary": {
      "capability": "current weather",
      "description": "weather-now (made in the Skill Workshop).",
      "threat": "reads one public weather endpoint"
    },
    "tier": "T3"
  },
  "case": "strong",
  "enable": {
    "approved_sha": "<SHA256>",
    "enabled": true,
    "state": "enabled",
    "status": "ok"
  }
}
```

### weak.json (rlimit under the test flag, egress broker_only, with sandbox_warning; enable ok)

```json
{
  "card": {
    "binding": "<BINDING>",
    "diff": {
      "base": null,
      "files": [
        {
          "bytes": 265,
          "path": "SKILL.md",
          "sha256": "<SHA256>",
          "text": "---\nname: weather-now\nversion: 0.1.0\ndescription: weather-now (made in the Skill Workshop).\negress: [api.example.com]\nentry: weather_now:TOOLS\ntools:\n  - {name: weather_now, tier: T2, outbound: true}\n---\n# weather-now\n\nDescribe when the model should use each tool.\n"
        },
        {
          "bytes": 275,
          "path": "fixtures.json",
          "sha256": "<SHA256>",
          "text": "{\n \"cases\": [\n  {\n   \"name\": \"now\",\n   \"tool\": \"weather_now\",\n   \"args\": {},\n   \"reads\": {\n    \"https://api.example.com/weather\": {\n     \"status\": 200,\n     \"body\": \"{\\\"t\\\": 21}\"\n    }\n   },\n   \"expect\": {\n    \"would_send\": 0,\n    \"result\": {\n     \"t\": 21\n    }\n   }\n  }\n ]\n}"
        },
        {
          "bytes": 211,
          "path": "weather_now.py",
          "sha256": "<SHA256>",
          "text": "\"\"\"weather\"\"\"\nimport json\n\n\ndef weather_now(call, **args):\n    r = call.http(\"GET\", \"https://api.example.com/weather\", {}, None)\n    return json.loads(r[\"body\"] or \"null\")\n\n\nTOOLS = {\"weather_now\": weather_now}\n"
        }
      ]
    },
    "hard": true,
    "kind": "skill_draft_enable",
    "local_author": true,
    "nonce": "<NONCE>",
    "outbound": true,
    "permissions": {
      "account": null,
      "egress": [
        "api.example.com"
      ],
      "secrets": [],
      "tools": [
        {
          "name": "weather_now",
          "outbound": true,
          "tier": "T2"
        }
      ]
    },
    "sandbox": {
      "backend": "rlimit",
      "cases": [
        {
          "error": null,
          "name": "now",
          "ok": true
        }
      ],
      "egress": "broker_only",
      "landlock_abi": null,
      "network_isolation": false,
      "ok": true,
      "pid_ns": null,
      "seccomp": true,
      "sha": "<SHA256>",
      "warning": "Network not isolated by the sandbox; only the broker limits egress. Files are not isolated either (backend rlimit)."
    },
    "sandbox_warning": "Network not isolated by the sandbox; only the broker limits egress. Files are not isolated either (backend rlimit).",
    "sha": "<SHA256>",
    "signed": false,
    "skill": "weather-now",
    "summary": {
      "capability": "current weather",
      "description": "weather-now (made in the Skill Workshop).",
      "threat": "reads one public weather endpoint"
    },
    "tier": "T3"
  },
  "case": "weak",
  "enable": {
    "approved_sha": "<SHA256>",
    "enabled": true,
    "state": "enabled",
    "status": "ok"
  }
}
```

### stale.json (sandbox facts rewritten while the card waited: enable stale; plus card() of an untested draft: not_tested)

```json
{
  "card": {
    "binding": "<BINDING>",
    "diff": {
      "base": null,
      "files": [
        {
          "bytes": 265,
          "path": "SKILL.md",
          "sha256": "<SHA256>",
          "text": "---\nname: weather-now\nversion: 0.1.0\ndescription: weather-now (made in the Skill Workshop).\negress: [api.example.com]\nentry: weather_now:TOOLS\ntools:\n  - {name: weather_now, tier: T2, outbound: true}\n---\n# weather-now\n\nDescribe when the model should use each tool.\n"
        },
        {
          "bytes": 275,
          "path": "fixtures.json",
          "sha256": "<SHA256>",
          "text": "{\n \"cases\": [\n  {\n   \"name\": \"now\",\n   \"tool\": \"weather_now\",\n   \"args\": {},\n   \"reads\": {\n    \"https://api.example.com/weather\": {\n     \"status\": 200,\n     \"body\": \"{\\\"t\\\": 21}\"\n    }\n   },\n   \"expect\": {\n    \"would_send\": 0,\n    \"result\": {\n     \"t\": 21\n    }\n   }\n  }\n ]\n}"
        },
        {
          "bytes": 211,
          "path": "weather_now.py",
          "sha256": "<SHA256>",
          "text": "\"\"\"weather\"\"\"\nimport json\n\n\ndef weather_now(call, **args):\n    r = call.http(\"GET\", \"https://api.example.com/weather\", {}, None)\n    return json.loads(r[\"body\"] or \"null\")\n\n\nTOOLS = {\"weather_now\": weather_now}\n"
        }
      ]
    },
    "hard": true,
    "kind": "skill_draft_enable",
    "local_author": true,
    "nonce": "<NONCE>",
    "outbound": true,
    "permissions": {
      "account": null,
      "egress": [
        "api.example.com"
      ],
      "secrets": [],
      "tools": [
        {
          "name": "weather_now",
          "outbound": true,
          "tier": "T2"
        }
      ]
    },
    "sandbox": {
      "backend": "rlimit",
      "cases": [
        {
          "error": null,
          "name": "now",
          "ok": true
        }
      ],
      "egress": "broker_only",
      "landlock_abi": null,
      "network_isolation": false,
      "ok": true,
      "pid_ns": null,
      "seccomp": true,
      "sha": "<SHA256>",
      "warning": "Network not isolated by the sandbox; only the broker limits egress. Files are not isolated either (backend rlimit)."
    },
    "sandbox_warning": "Network not isolated by the sandbox; only the broker limits egress. Files are not isolated either (backend rlimit).",
    "sha": "<SHA256>",
    "signed": false,
    "skill": "weather-now",
    "summary": {
      "capability": "current weather",
      "description": "weather-now (made in the Skill Workshop).",
      "threat": "reads one public weather endpoint"
    },
    "tier": "T3"
  },
  "card_not_tested": {
    "message": "the draft has no passing sandbox run; no card",
    "status": "not_tested"
  },
  "case": "stale",
  "enable": {
    "message": "the draft or its card changed while waiting for approval",
    "status": "stale"
  }
}
```
