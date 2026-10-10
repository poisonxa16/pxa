# PXA Control chat plugins

PXA Control's Chat tab has two views.

- **Classic** is `chat.js`. It exposes `window.PXAChat`. Its plugins are `chat-<name>.js` in this directory.
- **Assistant** (the default) is `tools/pxa_chat/ui/agent.js`. It exposes `window.PXAAgent`. Its plugins are `agent-<name>.js` next to `agent.js`.

Features are added by plugins: plain scripts that run after the core and use only the API below.
Everything runs in the browser, offline. There are no frameworks, no CDN and no network fetches except to PXA Control itself.

Both views share `window.PXAMarkdown`, a list of functions that run on the DOM after that view's own renderer.
Register one with `PXAChat.addMarkdownPostProcessor` or `PXAAgent.setMarkdownPostProcessor` (both call `PXAMarkdown.add`).
The function is `fn(root, ctx)`. `ctx.view` is `"classic"` or `"assistant"`. `ctx.live` is true while tokens are still arriving and false on the last paint of that reply.
Classic also passes `ctx.msg`, the message object. Assistant also passes `ctx.idx`, the turn index.
A thrown post-processor is logged and skipped. With none registered, the rendered DOM is unchanged.
`agent.js` creates `window.PXAMarkdown`. `chat.js` creates it only when `agent.js` did not.

## Files

| File | Role |
|---|---|
| `chat.js` / `chat.css` | the core: state, events, registries, the SSE reader, layout slots |
| `chat-history.js` | conversation list, search, rename, delete, export/import |
| `chat-markdown.js` | Classic-only renderer tweaks. Shared post-processing goes through `window.PXAMarkdown` |
| `chat-actions.js` | per-message actions: copy, edit, regenerate, delete |
| `chat-stream.js` | streaming comforts: smooth rendering, scrolling, continue |
| `chat-attach.js` | attachments: images for vision models, text files |
| `chat-params.js` | sampling parameters and per-chat settings |
| `chat-context.js` | context-window use and trimming |
| `chat-ux.js` | keyboard shortcuts, focus, small comforts |

Each plugin has a same-named `.css` file that is already linked. The page loads every file in the order of this table.
PXA Control serves only the files it knows: a new file needs a line in `STATIC` in `tools/pxa_control.py` and a
`<script>`/`<link>` tag in `index.html`. The `CHAT_PLUGINS` tuple there handles all of the files above.

A plugin starts with this line and touches nothing outside its own file:

```js
(function(){ const C = window.PXAChat; if (!C) return; /* plugin */ })();
```

The page's own helpers are globals that plugins may use: `$(sel, root)`, `$$(sel, root)`, `el(tag, attrs, ...kids)`
(attrs: `class`, `text`, `on<event>`, anything else becomes an attribute), `toast(msg, bad)`, `store.get/set` (small
per-browser settings), `fmtNum`, and the CSS variables (`--accent`, `--line`, `--dim`, `--bad`, `--s1`..`--s6`, `--r1`, `--r2`, `--fs-0`..`--fs-4`, `--mono`).

## Data model

```js
conv = {id, title, created, updated, settings: {}, messages: [msg, ...]}
msg  = {id, role: "user" | "assistant" | "system" | "tool", content, reasoning, meta: {}, parentId}
```

- `content` is the text. A plugin may set `msg.parts` (OpenAI content parts, e.g. `[{type:"text",...},{type:"image_url",...}]`): it is sent instead of `content`.
- `meta` holds `text` (the line under the bubble), `timings`, `usage`, `finish_reason`, `error`, `stopped`, `model` and `ts`. Plugins may add keys.
- `parentId` is the previous message. History is linear; a branching plugin can use it.
- `settings` belongs to plugins (for example a per-chat system prompt or sampler values).
- The request carries user turns and the assistant turns that produced text. A failed, empty reply is not sent.

## API: `window.PXAChat`

**State**
- `C.conv`: the current conversation.
- `C.target`: the server being talked to (`{sid}` or `{port}`, from the "Talk to" picker).
- `C.busy`: true while a reply streams.
- `C.store`: the storage API (below).

**Events.** `C.on(evt, fn)` returns a function that removes the handler. An error in a handler is logged and does not stop the chat.

| Event | Arguments | When |
|---|---|---|
| `init` | `(C)` | once, after every plugin has loaded (a late handler still runs once) |
| `beforeSend` | `(body, conv)` | before every request; mutate `body` (it may be async; handlers run one after another) |
| `delta` | `(chunk, msg)` | every parsed SSE chunk (raw OpenAI JSON); `msg` already holds the new text |
| `messageDone` | `(msg, conv)` | a reply ended: finished, stopped or failed |
| `error` | `(err, msg)` | a reply failed (HTTP error, unreachable, or an error object mid-stream) |
| `convChange` | `(conv)` | `newChat()` or `loadConv()` |
| `render` | `(msgEl, msg)` | a message element was rendered: when a message is added, when a reply is complete, and on load. It does not fire on every delta. |

**Actions**
- `C.send(text?, opts?)`: sends `text`. With no text it sends the prompt box. `opts.parts` and `opts.meta` go on the user message. It resolves to the assistant message.
- `C.regenerate(msgId)`: replaces that assistant reply. For a user message, it generates a new reply after it. Later messages are dropped.
- `C.stop()`, `C.newChat()`, `C.loadConv(id)`.
- `C.streamCompletion(body, onDelta, signal)`: one request to the current target. It resolves `{response, thinkNotes, timings, usage, done}` and throws on HTTP errors and on mid-stream errors. It handles chunks split anywhere, CRLF, `data:` with or without a space, comments, multi-line events, `[DONE]` and a missing `[DONE]`.
- `C.history(conv?)`: the messages as they would be sent (without the system prompt).
- `C.rerender(msg)`, `C.view(msgId)` (returns `{m, body, think, meta, actions}` elements), `C.toggleSidebar(open?)`.

**Registries**
- `C.addMessageAction({id, label, icon?, iconOnly?, title?, when(msg, conv), run(msg, conv, view)})`: a button in each matching message's `.msg-actions`. A message's actions render when it completes.
- `C.addSettingsSection(id, title, renderFn(box, C))`: a block under the chat settings. Returns `box`.
- `C.addSidebarPanel(id, renderFn(box, C))`: a panel in the left sidebar, `#chat-sidebar`. The sidebar and its "Panels" toggle appear once a panel exists. On narrow screens it starts closed.
- `C.addComposerButton({id, label, icon?, iconOnly?, title?, onClick(conv, event)})`: a button in the toolbar above the prompt box.
- `C.addSlashCommand(name, fn(args, conv))`: `/name args` typed in the box runs `fn` instead of sending. If `fn` returns a string, that string is sent. A `/word` that is not registered is sent unchanged, so model soft tags such as `/no_think` keep working.
- `C.setRenderer(fn(text, msg) -> html)`: replaces the markdown renderer and re-renders every message. `null` restores the default. **The renderer must escape the text**: its output goes to `innerHTML`. `C.md`, `C.esc` and `C.inline` are the defaults, for reuse.
- `C.addMarkdownPostProcessor(fn(root, ctx))`: adds `fn` to the shared `window.PXAMarkdown` list. It then runs in both views, after the renderer, on the DOM. Do not assign `innerHTML` from the message text; the renderer already escaped it.

**Storage.** `C.store` provides `get(id)`, `put(conv)`, `list()` (returns `[{id, title, created, updated, count}]`, newest first) and `delete(id)`.
Each may return a value or a Promise. The default keeps conversations in memory and in one browser `localStorage` entry
(`pxa-control.chat.convs`, at most the newest 100). Every access is guarded, and a full quota drops the oldest
conversations first. To keep them elsewhere (for example IndexedDB), a plugin assigns `C.store = {get, put, list, delete}`
in its body. The core saves a conversation after each user message and each finished reply.

## Layout slots

`#chat-sidebar` (left column, hidden until a panel exists), `#chat-settings-sections` (in the Settings card),
`#chat-composer-tools` (above the prompt box), and `.msg-actions` in every `.msg`. The existing ids (`#msgs`, `#c-in`, `#c-send`, `#c-stop`,
`#c-sys`, `#c-temp`, `#c-max`, `#c-think*`, `#c-target`, `#c-attach`) remain. Plugins should add to them and not replace them.

## Example plugin

```js
// chat-wordcount.js: shows the word count of every reply and adds a /clear command.
(function(){ const C = window.PXAChat; if (!C) return;
  C.on("messageDone", msg => {
    const v = C.view(msg.id); if (!v || !msg.content) return;
    v.meta.textContent += " · " + msg.content.trim().split(/\s+/).length + " words";
  });
  C.addSlashCommand("clear", () => { C.newChat(); });
  C.on("beforeSend", body => { body.top_k = 40; });
})();
```

## Testing

The chat can be tested without a GPU against a small mock OpenAI server. The mock streams content and reasoning, and can
be told to be slow, to split chunks, to fail mid-stream or to return HTTP 500. A headless browser runs the checks.
Each plugin should add its own small check file next to the core checks: the plugin's check sends a message, asserts
what the plugin shows, and leaves no console errors.

## Assistant view: `window.PXAAgent`

The Assistant is the view people see first. `agent.js` is the core: the transcript, the tool loop, approvals, saved chats and memory.
Plugins extend it. They do not replace it. A plugin that registers nothing leaves the page as the core drew it.
Empty composer and settings slots stay hidden, so a stub does not move the layout. Features fill these files and must keep using this API.

### Files

| File | Role |
|---|---|
| `agent.js` / `agent.css` | the core, and `window.PXAAgent`. Edit, regenerate, and the `< 1/3 >` version list live here |
| `agent-markdown.js` | shared highlighting and KaTeX `$` / `$$` math (vendored, MIT, offline), registered on `window.PXAMarkdown` |
| `agent-attach.js` | paste and drag of files and images |
| `agent-actions.js` | Continue from the last reply |
| `agent-params.js` | sampling parameters, system-prompt presets, snippets |
| `agent-context.js` | token meter: the server's `/tokenize`, or chars/4 labelled estimated |
| `agent-ux.js` | slash commands, shortcuts, voice, mobile layout, and the offline app shell |
| `agent-stream.js` | retry, a second server, per-reply metrics |
| `agent-memory.js` | auto-compact marker, `/compact`, threshold. Memory retention, the recall tools, `@` references, and sub-agents (`spawn_agent`) live in the Assistant core |

`index.html` loads `agent.js`, then these plugins, then `chat.js` and the Classic plugins.
PXA Control serves the Assistant files from `pxa_chat.register` (`AGENT_PLUGINS` in `tools/pxa_chat/__init__.py`).
A new Assistant plugin needs a name in that tuple, a `<script>` and `<link>` in `index.html`, and the two files under `tools/pxa_chat/ui/`.

A plugin starts with this line and touches nothing outside its own file:

```js
(function(){ const G = window.PXAAgent; if (!G) return; /* plugin */ })();
```

The same page helpers Classic uses are available: `$(sel, root)`, `$$(sel, root)`, `el(tag, attrs, ...kids)`, `toast(msg, bad)`, `store.get/set`.

With no plugin registered, Assistant behaves as it did before this surface. The same requests go out, and the same bubbles, buttons and drawer come back.
`#ag-composer-tools` and `#ag-settings-sections` are in the page and hidden while empty, so they do not shift the composer or the drawer.

### API

**State**
- `G.version` is `1`. `G.ready` is true after boot (presets loaded, layout built, a saved chat opened or the welcome shown).
- `G.sessionId`: the open chat, or `null`.
- `G.state`: a snapshot `{session, run, turns, servers, selected, preset, advanced, view, title, web, useMem}`. `turns` is a copy.
- `G.el`: `{root, col, scroll, input, send, composerTools, drawer, sections}`. Slots exist once `ready` is true.
- `G.md`: the Assistant renderer, `{render, inline, esc, splitThink}`. `render` escapes its input.

**Events.** `G.on(evt, fn)` returns a function that removes the handler. A handler that throws is logged and skipped. `beforeRun` handlers may be async and run one after another.

| Event | Arguments | When |
|---|---|---|
| `init` | `(G)` | once, after boot. A handler added later still runs once. |
| `sessionChange` | `(sessionId, G)` | a chat was opened, created or cleared. `sessionId` is `null` for a new chat. |
| `beforeRun` | `(body, G)` | before `POST /api/chat/run`. Mutate `body` (the message, the server, the preset, sampling, `rewind`). |
| `delta` | `(ev, msgEl)` | every SSE event of the current run, after the core has applied it. `ev` is `{type, seq, data, t}`. |
| `messageDone` | `(msg, G)` | the run ended (`run.done`, `run.error` or `run.cancelled`). `msg` is `{role, idx, end, data, el, text}`. |
| `render` | `(el, msg)` | a user bubble was added, or an assistant footer was written. Not on every token. |

**Actions**
- `G.send(text?)`: sends `text`, or the prompt box when `text` is omitted.
- `G.stop()`, `G.newChat()`, `G.openSession(id)`.
- `G.resend(idx, text?)`: drops that turn and everything after it, then sends `text` (or the saved turn text).
- `G.sessions()`: resolves to a copy of the saved-chat list.
- `G.insertMarker(text, opts?)`: a note row in the transcript. `opts` is `{kind, id, before}`. It is on screen only. It is not saved and it is not sent. Returns the element.
- `G.addMessageAction({id, label, icon?, title?, when(msg, G)?, run(msg, G)})`: a button in each matching message's toolbar, including messages already on screen. `icon` is a name the Assistant already draws (for example `"copy"`), or omit it and the label is shown. Built-in copy, edit and regenerate stay.
- `G.addComposerButton({id, label, icon?, iconOnly?, title?, onClick(G, event)})`: a button in `#ag-composer-tools`.
- `G.addSettingsSection(id, title, render(box, G))`: a block at the bottom of the settings drawer.
- `G.addSlashCommand(name, fn(args, G))`: `/name args` in the box runs `fn` instead of sending. If `fn` returns a string, that string is sent. If it returns nothing, nothing is sent. A `/word` that is not registered is sent unchanged, so model soft tags such as `/no_think` keep working.
- `G.slashCommands()`: the registered names.
- `G.addRunEventType(type)`: also deliver this SSE event name to `on("delta")`. The core already delivers `run.*`, `status`, `step`, `text.*`, `think.*`, `tool.*` and `approval.*`. `compact` is already registered, for the compaction marker.
- `G.setMarkdownPostProcessor(fn)`: the shared post-processor described at the top. Assistant calls pass `ctx.idx`.

### Example

```js
// agent-wordcount.js: a /words command and a count on each finished reply.
(function(){ const G = window.PXAAgent; if (!G) return;
  G.addSlashCommand("words", () => { G.insertMarker("Word count is on"); });
  G.on("messageDone", msg => {
    if (msg.end !== "run.done" || !msg.text) return;
    G.insertMarker(msg.text.trim().split(/\s+/).length + " words");
  });
})();
```
