"""PXA Control: the HTTP API -- one r_<name>(app, body, query) per route, the ROUTES table, the static page map
(STATIC) and the chat agent's registration. The shapes the page uses are a contract: see ROUTES-AUTH.md."""
from .core import *  # noqa: F401,F403
from .auth import *  # noqa: F401,F403
from . import core as _core


# ---------------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------------
STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/index.html": ("index.html", "text/html; charset=utf-8"),
          "/encode.js": ("encode.js", "application/javascript; charset=utf-8"),
          "/encode.css": ("encode.css", "text/css; charset=utf-8"),
          "/live.js": ("live.js", "application/javascript; charset=utf-8"),
          "/live.css": ("live.css", "text/css; charset=utf-8"),
          "/profiles.js": ("profiles.js", "application/javascript; charset=utf-8"),
          "/profiles.css": ("profiles.css", "text/css; charset=utf-8"),
          "/mqtt.js": ("mqtt.js", "application/javascript; charset=utf-8"),
          # the 2026-10 UX revamp: Home, the Plugins / Store sections, their store adapter + preview fixture
          "/home.js": ("home.js", "application/javascript; charset=utf-8"),
          "/plugins.js": ("plugins.js", "application/javascript; charset=utf-8"),
          "/plugin-host.js": ("plugin-host.js", "application/javascript; charset=utf-8"),
          "/plugin-host.css": ("plugin-host.css", "text/css; charset=utf-8"),
          "/store.js": ("store.js", "application/javascript; charset=utf-8"),
          "/wallet.js": ("wallet.js", "application/javascript; charset=utf-8"),
          "/wallet.css": ("wallet.css", "text/css; charset=utf-8"),
          "/store-mock.json": ("store-mock.json", "application/json; charset=utf-8"),
          "/mark.png": ("mark.png", "image/png"),
          "/favicon.png": ("mark.png", "image/png"),
          # the installable app (PWA): manifest, shell service worker, its registration, icons and
          # the host picker (also the offline page and the Android app's start page)
          "/manifest.webmanifest": ("manifest.webmanifest", "application/manifest+json; charset=utf-8"),
          "/sw.js": ("sw.js", "application/javascript; charset=utf-8"),
          "/pwa.js": ("pwa.js", "application/javascript; charset=utf-8"),
          "/hosts.html": ("hosts.html", "text/html; charset=utf-8"),
          "/icon-192.png": ("icon-192.png", "image/png"),
          "/icon-512.png": ("icon-512.png", "image/png")}
# the Chat tab's Classic view: chat.js (the PXAChat core) and its plugins chat-<name>.js / .css
# (tools/pxa_control_ui/CHAT-PLUGINS.md). The Assistant view's agent-<name> plugins are registered
# by pxa_chat.register, not here.
CHAT_PLUGINS = ("history", "markdown", "actions", "stream", "attach", "params", "context", "ux")
for _n in ("chat",) + tuple("chat-" + p for p in CHAT_PLUGINS):
    STATIC["/" + _n + ".js"] = (_n + ".js", "application/javascript; charset=utf-8")
    STATIC["/" + _n + ".css"] = (_n + ".css", "text/css; charset=utf-8")


def r_thinking(app, body, query):
    return app.thinking_info(query)


def r_thinking_override(app, body, query):
    body = body or {}
    return {"model": body.get("model"), "saved": app.set_thinking_override(body.get("model"), body.get("thinking"))}


def r_thinking_table(app, body, query):
    t = TH.load_table()
    return {"version": t.get("version"), "note": t.get("note"), "answer_reserve": t.get("answer_reserve"),
            "families": [{k: f.get(k) for k in ("id", "label", "mechanism", "default", "levels", "budget", "source")}
                         for f in t["families"]]}



def _pxa_update_bin():
    """the pxa-update binary shipped next to this install, or none."""
    env = os.environ.get("PXA_UPDATE_BIN")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    for c in (os.path.join(HERE, "pxa-update"), os.path.join(os.path.dirname(HERE), "bin", "pxa-update")):
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _pxa_update_check(bin):
    p = subprocess.run([bin, "check"], capture_output=True, text=True, timeout=40)
    line = ""
    if p.stdout:
        lines = [x for x in p.stdout.splitlines() if x.startswith("current=")]
        line = lines[-1] if lines else ""
    info = {"ok": p.returncode == 0, "binary": True, "update": False}
    for part in line.split():
        if "=" in part:
            k, v = part.split("=", 1)
            if k in ("current", "latest"):
                info[k] = v
            elif k == "channel" and v in ("stable", "beta"):
                info["channel"] = v
                _ENGINE_CH["last"] = v
            elif k == "update":
                info["update"] = v == "yes"
    if p.returncode != 0:
        info["error"] = ((p.stderr or "could not check for an update").strip())[:300]
    return info


# The engine beta channel (Valued Supporters): may this key use it? Asked of the licence server through `pxa-update channels`,
# cached 6 h (10 min after a failure), so opening the update dialog does not cost a round trip each time.
_ENGINE_CH = {"t": 0.0, "beta": False, "ttl": 0, "last": "stable"}
_ENGINE_CH_LOCK = threading.Lock()


def _pxa_update_beta_allowed(bin, force=False):
    with _ENGINE_CH_LOCK:
        if not force and time.time() - _ENGINE_CH["t"] < _ENGINE_CH["ttl"]:
            return _ENGINE_CH["beta"]
    try:
        p = subprocess.run([bin, "channels"], capture_output=True, text=True, timeout=40)
        ok = p.returncode == 0
        beta = ok and any(part == "beta=yes" for line in p.stdout.splitlines() for part in line.split())
    except (OSError, subprocess.TimeoutExpired):
        ok, beta = False, False
    with _ENGINE_CH_LOCK:
        _ENGINE_CH.update(t=time.time(), beta=bool(beta), ttl=6 * 3600 if ok else 600)
    return bool(beta)


def _servers_running(app):
    """the servers this Control started whose process is alive now (Popen.poll), the same test the apply guard uses"""
    return len([s for s in list(app.seats.values()) if s.running()])


def r_pxa_update(app, body, query):
    """GET /api/update: ask the pxa-update binary. Python does not download anything. `channel` is what the install follows
    (stable unless it chose beta); `beta_allowed` says whether to offer the Engine channel row (Valued Supporters only)."""
    b = _pxa_update_bin()
    if not b:
        return {"ok": True, "binary": False, "update": False, "servers_running": _servers_running(app)}
    info = _pxa_update_check(b)
    info["servers_running"] = _servers_running(app)       # counted after the check (it can take 40 s): live when the page reads it
    info.setdefault("channel", "stable")
    info["beta_allowed"] = _pxa_update_beta_allowed(b)
    return info


def _pxa_update_run(args, timeout, what):
    b = _pxa_update_bin()
    if not b:
        e = Invalid("pxa-update is not in this install")
        e.status, e.code = 404, "no_updater"
        raise e
    try:
        p = subprocess.run([b] + args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        e = Invalid("The %s took too long and was stopped. Try again." % what)
        e.status, e.code = 504, "update_timeout"
        raise e
    if p.returncode != 0:
        msg = ((p.stderr or p.stdout or "%s failed" % what).strip())
        msg = msg[len("pxa-update: "):] if msg.startswith("pxa-update: ") else msg
        e = Invalid(msg[:400])
        e.status, e.code = (403, "not_valued") if "Valued Supporters" in msg else (500, "update_failed")
        raise e
    return p


def r_pxa_update_channel(app, body, query):
    """POST /api/update/channel {channel: stable|beta}: record the engine channel. It installs nothing; the page calls
    /api/update/apply next. Beta is refused (403) unless the licence server calls the key Valued."""
    ch = str((body or {}).get("channel") or "")
    if ch not in ("stable", "beta"):
        raise Invalid("channel must be stable or beta")
    _pxa_update_run(["channel", ch], 60, "channel change")
    with _ENGINE_CH_LOCK:
        _ENGINE_CH["last"] = ch
    return {"ok": True, "channel": ch}


def r_pxa_update_notes(app, body, query):
    """GET /api/update/notes: the release notes of the installed (or offered) beta, shown only after pxa-update has checked
    them against the signed release. Plain text: the page renders it as text, never as HTML."""
    p = _pxa_update_run(["notes", "--json"], 120, "notes download")
    try:
        d = json.loads((p.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise Invalid("pxa-update did not return the notes")
    return {"ok": True, "version": str(d.get("version") or ""), "build_id": str(d.get("build_id") or ""),
            "notes": str(d.get("notes") or "")[:256 * 1024]}


def r_pxa_update_apply(app, body, query):
    """POST /api/update/apply: run pxa-update apply. Refuses while a server is running."""
    if any(s.running() for s in app.seats.values()):
        e = Invalid("Stop the running servers first. PXA will not restart them.")
        e.status, e.code = 409, "servers_running"
        raise e
    b = _pxa_update_bin()
    if not b:
        e = Invalid("pxa-update is not in this install")
        e.status, e.code = 404, "no_updater"
        raise e
    try:
        # a beta comes from the licence server through a tunnel (a 1.5 GB tarball): give it longer than a GitHub stable
        p = subprocess.run([b, "apply"], capture_output=True, text=True, timeout=2400 if _ENGINE_CH["last"] == "beta" else 600)
    except subprocess.TimeoutExpired:
        e = Invalid("The update took too long and was stopped. Try again.")
        e.status, e.code = 504, "update_timeout"
        raise e
    if p.returncode != 0:
        e = Invalid(((p.stderr or p.stdout or "update failed").strip())[:400])
        e.status, e.code = (409 if p.returncode == 2 else 500), "update_failed"
        raise e
    return {"ok": True, "output": (p.stdout or "").strip()[:400]}


# The Beta card (a beta release downloads straight from PXA Control when the wallet holds the coins): the
# release, its cost in thank-you coins and the wallet balance come from licd through the install key; the one button redeems
# (POST /api/update/beta/redeem) and the page then records the beta channel and installs it with pxa-update (above). The Docker
# image archive is a second file of the same release: downloaded through a personal link, resumed, kept only when its sha256
# matches the signed file list. Download only: nothing is pushed to or pulled from a registry.
_BETA_DL = {"state": "idle", "done": 0, "size": 0, "name": "", "path": "", "error": ""}
_BETA_DL_LOCK = threading.Lock()
# licd limits a computer's wallet calls (120 an hour). The card reads the wallet at most once a minute (or after a wallet change),
# the download's progress (?dl=1) never calls licd, and a 429 keeps the last good card and backs off until licd's Retry-After.
BETA_VIEW_TTL, BETA_BACKOFF_S = 60.0, 300.0
_BETA_VIEW = {"card": None, "at": -1e18, "epoch": -1, "until": 0.0}
_BETA_VIEW_LOCK = threading.Lock()
_beta_clock = time.monotonic


def _beta_client():
    base = _store.store_url()
    if not base:
        e = Invalid("the store is not configured on this computer")
        e.status, e.code = 409, "store_off"
        raise e
    return _store.LicdClient(base, config_dir())


def _beta_card(view):
    b = (view or {}).get("beta") or {}
    coins, cost = int(view.get("coins") or 0), int(b.get("coins") or 0)
    access = bool(b.get("access"))
    return {"store": True, "wallet": view.get("wallet_id"), "coins": coins, "release": b.get("release") or None, "access": access,
            "how": b.get("how"), "cost": cost if not access else 0, "short_by": 0 if access or not b.get("release") else max(0, cost - coins),
            "kofi_linked": bool(view.get("kofi_linked")), "valued_supporter": bool(view.get("valued_supporter")),
            "valued_only": bool(b.get("valued_only"))}     # licd's Valued-only beta: not Valued = release hidden, no coin price


def r_update_beta(app, body, query):
    """GET /api/update/beta: what the Beta card shows: release, cost in coins, balance, whether this wallet has it already.
    ?dl=1: only the Docker download's progress (local, no licd call): the page polls that while a download runs."""
    if not _store.store_url():
        return {"store": False}
    with _BETA_DL_LOCK:
        dl = dict(_BETA_DL)
    if str(_q(query, "dl", "")) == "1":
        return {"docker": dl}
    t = _beta_clock()
    with _BETA_VIEW_LOCK:
        c, fresh = _BETA_VIEW["card"], t - _BETA_VIEW["at"] < BETA_VIEW_TTL and _BETA_VIEW["epoch"] == _store.WALLET_CHANGES[0]
        wait = t < _BETA_VIEW["until"]
        retry = max(1, int(_BETA_VIEW["until"] - t) + 1) if wait else 0
    if c is not None and (fresh or wait):
        return dict(c, docker=dl, cached=True, busy=wait, retry_in=retry) if wait else dict(c, docker=dl, cached=True, busy=False)
    if wait:
        return {"store": True, "busy": True, "retry_in": retry, "docker": dl}
    epoch = _store.WALLET_CHANGES[0]
    try:
        v = _beta_client().wallet("view")
    except Invalid as e:
        if getattr(e, "code", "") == "no_wallet":
            return {"store": True, "wallet": None}
        if getattr(e, "code", "") == "store_offline":
            return {"store": True, "offline": True}
        if getattr(e, "status", 0) == 429:
            ra = getattr(e, "retry_after", None)
            with _BETA_VIEW_LOCK:
                _BETA_VIEW["until"] = t + max(30.0, float(ra) if ra else BETA_BACKOFF_S)
                retry = int(_BETA_VIEW["until"] - t) + 1
            return dict(c, docker=dl, cached=True, busy=True, retry_in=retry) if c is not None else {"store": True, "busy": True, "retry_in": retry, "docker": dl}
        raise
    card = _beta_card(v)
    with _BETA_VIEW_LOCK:
        _BETA_VIEW.update(card=card, at=t, epoch=epoch, until=0.0)
    return dict(card, docker=dl)


def r_update_beta_redeem(app, body, query):
    """POST /api/update/beta/redeem: spend the coins for the current beta (one licd transaction: coins out, entitlement in). Redeeming
    again costs nothing: an account that has the beta keeps it."""
    r = _beta_client().wallet("beta")
    card = _beta_card(r.get("wallet") or {})
    with _BETA_VIEW_LOCK:
        _BETA_VIEW.update(card=card, at=_beta_clock(), epoch=_store.WALLET_CHANGES[0], until=0.0)
    return dict(card, redeemed=bool(r.get("redeemed")))


def _beta_download_dir():
    return os.path.join(config_dir(), "downloads")


def r_update_beta_docker(app, body, query):
    """POST /api/update/beta/docker: download the release's Docker image archive in the background (GET /api/update/beta shows it)."""
    try:
        import pxa_engine_update as EU
    except Exception:  # noqa: BLE001
        e = Invalid("this install has no engine updater (tools/pxa_engine_update.py)")
        e.status, e.code = 404, "no_updater"
        raise e
    base = _store.store_url()
    if not base:
        e = Invalid("the store is not configured on this computer")
        e.status, e.code = 409, "store_off"
        raise e
    with _BETA_VIEW_LOCK:
        want = ((_BETA_VIEW["card"] or {}).get("release") or None)     # the release the card shows: an older beta's links are not reused
    with _BETA_DL_LOCK:
        if _BETA_DL["state"] == "running":
            return dict(_BETA_DL)
        _BETA_DL.update(state="running", done=0, size=0, name="", path="", error="")

    def prog(done, size):
        with _BETA_DL_LOCK:
            _BETA_DL.update(done=int(done or 0), size=int(size or 0))

    def run():
        try:
            path, f = EU.fetch_file("docker", _beta_download_dir(), base=base, progress=prog, cfg_dir=config_dir(), want_version=want)
            with _BETA_DL_LOCK:
                _BETA_DL.update(state="done", path=path, name=f["name"], done=f["size"], size=f["size"])
        except Exception as ex:  # noqa: BLE001 - shown on the card; a network drop keeps the .part and the next press resumes
            with _BETA_DL_LOCK:
                _BETA_DL.update(state="failed", error=str(ex)[:300])

    threading.Thread(target=run, daemon=True, name="pxa-beta-docker").start()
    with _BETA_DL_LOCK:
        return dict(_BETA_DL)


def r_lib_update(app, body, query):
    """GET /api/lib/update: the licensed library's update state. ?check=1 asks the licence server now, otherwise the
    cached answer is returned and the poller refreshes it in the background."""
    app.lib.touch()
    if str(_q(query, "check", "")).lower() in ("1", "true", "yes", "on"):
        return app.lib.check(force=True)
    return app.lib.status()


def r_lib_update_apply(app, body, query):
    """POST /api/lib/update/apply: install the newest release on the channel. Refuses while a server is running, on a
    refused or unverifiable manifest, and on an engine older than the release asks for."""
    app.lib.touch()
    return app.lib.apply(channel=(_b(body).get("channel") or None))


def r_lib_update_rollback(app, body, query):
    """POST /api/lib/update/rollback: put back the previous library, or the one the engine shipped."""
    app.lib.touch()
    return app.lib.rollback()


def r_lib_update_settings(app, body, query):
    """POST /api/lib/update/settings {channel, auto}: the channel this install follows and whether a newer release is
    applied without asking. Both are off until set; the beta channel is still refused by the server for a key without
    the valued role."""
    app.lib.touch()
    b = _b(body)
    ch, auto = b.get("channel"), b.get("auto")
    if ch is not None and ch not in LIB_CHANNELS:
        raise Invalid("channel must be one of: %s" % ", ".join(LIB_CHANNELS))
    if auto is not None and not isinstance(auto, bool):
        raise Invalid("auto must be true or false")
    # only the fields the body names: the page sends one at a time ({auto} or {channel}), and passing the absent one as
    # None made cfg_set drop it, so turning auto-install on reset a beta channel to stable and vice versa
    app.lib.cfg_set(**{k: v for k, v in (("channel", ch), ("auto", auto)) if v is not None})
    return app.lib.status()


def r_expert_map(app, body, query):
    """GET /api/expert-map: the adaptive map, in plain words, and where it stands."""
    import pxa_expert_map as M
    model = _q(query, "model", "") or ""
    return M.status_dict(model or None)


def r_expert_map_rebuild(app, body, query):
    """POST /api/expert-map/rebuild: start from the curated csv. Calibration is the builder."""
    import pxa_expert_map as M
    body = body or {}
    model = body.get("model") or ""
    curated = body.get("curated") or ""
    dest = body.get("dest") or (M.beside_model(model) if model else "")
    if model and curated:
        try:
            M.curated_start(model, curated, dest)
        except ValueError as e:
            err = Invalid(str(e))
            err.status, err.code = 400, "bad_map"
            raise err
    out = M.status_dict(model or None, dest or None)
    out["ok"] = True
    out["note"] = (
        "The curated map is the starting point. Each calibration pass is measured. "
        "The builder stops when decode speed stops rising and keeps the fastest map. "
        "Leave online adapt on after that."
    )
    return out


def r_expert_map_reset(app, body, query):
    """POST /api/expert-map/reset: drop learned counts. The curated file stays."""
    import pxa_expert_map as M
    body = body or {}
    model = body.get("model") or ""
    if not model:
        err = Invalid("name the model")
        err.status, err.code = 400, "bad_map"
        raise err
    try:
        M.reset_learned([M.learned_beside(model), M.cache_learned(model)])
    except ValueError as e:
        err = Invalid(str(e))
        err.status, err.code = 400, "bad_map"
        raise err
    out = M.status_dict(model)
    out["ok"] = True
    out["note"] = "Learned counts cleared. The map that shipped with the model is unchanged."
    return out


def r_health(app, body, query):
    """GET /api/health: is this Control alive and what can it do. Cheap: no nvidia-smi call, no server probe."""
    seats = [s for s in app.seats.values() if s.running()]
    d = {"ok": True, "control_version": CONTROL_VERSION, "pid": os.getpid(), "uptime_s": round(time.time() - app.started_at, 1),
         "servers_running": len(seats), "nvidia_smi": shutil_which("nvidia-smi") is not None,
         "telemetry": {"available": TEL is not None, "recording": app.telemetry is not None},
         "encode": ENC is not None, "gpu_profiles": None, "errors": []}
    if app.gpuctl is not None:
        try:
            d["gpu_profiles"] = app.gpuctl.health()
        except Exception as e:        # noqa: BLE001 - health answers even when the backend is broken
            d["errors"].append(f"gpu profiles: {e.__class__.__name__}: {e}")
    elif app.gpuctl_error:
        d["errors"].append("gpu profiles: " + app.gpuctl_error)
    return d


def _limit(query, default=200, hi=2000):
    v = _q(query, "limit", str(default))
    if not str(v).isdigit():
        raise Invalid("limit: a whole number")
    return max(1, min(hi, int(v)))


def r_gpu_state(app, body, query):
    return app.gpu_state()


def r_gpu_profiles(app, body, query):
    g = app.need_gpuctl()
    return {"profiles": list(g.store.profiles().values()), "presets": CTL.store.PRESETS}


def r_gpu_profile_save(app, body, query):
    return {"profile": app.need_gpuctl().save_profile(body, who=req_who())}


def r_gpu_profile_delete(app, body, query):
    return app.need_gpuctl().delete_profile(body, who=req_who())


def r_gpu_plan(app, body, query):
    return app.need_gpuctl().plan(body)


def r_gpu_apply(app, body, query):
    return app.need_gpuctl().apply(body, who=req_who())


def r_gpu_reset(app, body, query):
    return app.need_gpuctl().reset(body, who=req_who())


def r_gpu_reserve(app, body, query):
    return app.need_gpuctl().reserve(body, who=req_who())


def r_gpu_maintenance(app, body, query):
    return app.need_gpuctl().maintenance(body, who=req_who())


def r_gpu_quiet(app, body, query):
    return app.need_gpuctl().quiet(body, who=req_who())


def r_gpu_audit(app, body, query):
    return app.need_gpuctl().audit_read(limit=_limit(query))


def r_gpu_export(app, body, query):
    d = app.need_gpuctl().export()
    return Reply(json.dumps(d, indent=1), "application/json; charset=utf-8",
                 extra={"Content-Disposition": 'attachment; filename="pxa-gpu-profiles.json"'})


def r_gpu_import(app, body, query):
    return app.need_gpuctl().import_(body, who=req_who())


def r_gpu_schedules(app, body, query):
    return app.need_gpuctl().set_schedules(body, who=req_who())


def r_gpu_supervisor(app, body, query):
    return app.need_gpuctl().supervisor_action(body, who=req_who())


def r_gpu_undo(app, body, query):
    return app.need_gpuctl().undo(body, who=req_who())


def r_gpu_ui(app, body, query):
    """POST /api/gpu/ui {"mode": "simple" | "advanced"}: the Profiles page's Simple / Advanced switch, remembered in
    control.json (the page also keeps it per browser)."""
    mode = (body or {}).get("mode")
    if mode not in ("simple", "advanced"):
        raise Invalid("mode: simple or advanced")
    with _cfg_lock:
        c = load_config()
        c["ui_mode"] = mode
        save_config(c)
    if app.gpuctl is not None:
        app.gpuctl.settings["ui_mode"] = mode
    return {"mode": mode}


def r_gpu_power(app, body, query):
    return app.need_gpuctl().set_power(body, who=req_who())


def r_gpu_clocks(app, body, query):
    return app.need_gpuctl().clocks(_q(query, "uuid"))


def r_info(app, body, query):
    return {"version": 1, "control_version": CONTROL_VERSION, "lan": app.lan, "port": app.port,
            "companion": app.companion, "idle_exit_s": app.idle_s if app.companion else None,
            "catalog_src": app.catalog_src, "levers": len(app.catalog), "kv_types": KV_TYPES,
            "split_modes": SPLIT_MODES, "config": config_path(), "docker": docker_bin() is not None,
            "extra_flags": sorted(EXTRA_FLAGS), "thinking": True, "token_required": bool(app.token),
            "store": dict(_store_urls())}         # store.js calls only what is announced here


def r_rig(app, body, query):
    return {"static": app.rig_static(force=_q(query, "force") == "1"), "live": app.rig_live()}


def r_rig_live(app, body, query):
    app.live.touch()
    return app.rig_live()


def r_doctor(app, body, query):
    return app.doctor(force=_q(query, "force") == "1")


def r_companions(app, body, query):
    """Drafter next to the chosen model, and whether --hot-model may be offered for these cards."""
    model = _q(query, "model") or ""
    raw = _q(query, "gpus") or ""
    want = set()
    for part in raw.replace(",", " ").split():
        try:
            want.add(int(part))
        except ValueError:
            raise Invalid("card %r is not an index" % part)
    rows, _err = app.gpus()
    sel = [g for g in rows if g[0] in want] if want else []
    drafters = []
    if isinstance(model, str) and model.lower().endswith(".gguf") and os.path.isfile(model):
        sib = app.L.gemma4_assistant_siblings(model)
        if sib:
            drafters.append({"path": sib, "file": os.path.basename(sib)})
    return {"drafters": drafters, "vmm": bool(sel) and app.L.vmm_offered(sel)}


def r_models(app, body, query):
    return app.models(force=_q(query, "force") == "1")


def r_model_dirs(app, body, query):
    dirs = app.set_model_dirs((body or {}).get("model_dirs"))
    return {"model_dirs": dirs, "models": app.models(force=True)}


def r_fits(app, body, query):
    body = body or {}
    ctx = _int(body, "ctx", 0, 1 << 21, 0)
    return app.fit_table(body.get("card_sets"), ctx)


def r_hf_available(app, body, query):
    return app.hf_available(force=_q(query, "force") == "1", cached_only=_q(query, "cached") == "1")


def r_hf_download(app, body, query):
    return app.hf_start(body)


def r_hf_progress(app, body, query):
    return app.hf_progress()


def r_hf_cancel(app, body, query):
    return {"cancelled": app.hf_dl.cancel() if app.hf_dl is not None else False}


def r_hf_token(app, body, query):
    set_now = app.set_hf_token((body or {}).get("token"))
    return {"token_set": set_now}               # the token itself never comes back through here


def r_levers(app, body, query):
    names = app.catalog_names
    every, preset = getattr(names, "every", set()), getattr(names, "preset", set())
    # built-in levers set where Control was started (the Flash-Next preset's env): counted, not named; they reach the
    # server through the inherited environment
    env_on = sum(1 for k, v in os.environ.items() if v and isinstance(names, LeverNames) and names.builtin(k))
    return {"src": app.catalog_src,
            "levers": [dict(r, kind=lever_kind(r), default_state=lever_default_state(r)) for r in app.catalog],
            "builtin": {"hashes": sorted(every), "preset": sorted(preset), "env_on": env_on}}


def r_levers_lint(app, body, query):
    levers, errs = validate_levers((body or {}).get("levers") or {}, app.catalog_names)
    return {"errors": errs, "warnings": lint_levers(levers, app.catalog)}


def r_plan(app, body, query):
    return app.plan(body)


def r_start(app, body, query):
    return app.start(body)


def r_stop(app, body, query):
    sid = (body or {}).get("sid") or _q(query, "sid") or "main"
    return {"stopped": app.stop(sid), "status": app.status(sid)}


def r_restart(app, body, query):
    return app.restart(body if body else None, sid=_q(query, "sid"))


def r_status(app, body, query):
    return app.status(_q(query, "sid"))


def r_log(app, body, query):
    try:
        since = int(_q(query, "since", "0"))
    except ValueError:
        since = 0
    seat = app.get_seat(_q(query, "sid") or "main", create=not _q(query, "sid"))
    q = _q(query, "q", "")
    if q:
        items = seat.search(q[:200], regex=_q(query, "regex") == "1")
        return {"seq": seat.seq, "lines": [x[1] for x in items], "q": q, "matches": len(items)}
    items = seat.lines_since(since)
    return {"seq": items[-1][0] if items else since, "lines": [x[1] for x in items]}


def r_servers(app, body, query):
    prof = app.profiles()
    out = []
    for sid in sorted(prof, key=lambda k: (k != "main", prof[k].get("name", k).lower())):
        st = app.seats[sid].status() if sid in app.seats else Seat(app.L, sid).status()
        out.append({"sid": sid, "name": prof[sid].get("name", sid), "settings": prof[sid].get("settings") or {},
                    "running": st["running"], "phase": st["phase"], "port": st["port"], "pid": st["pid"]})
    return {"servers": out}


def r_server_save(app, body, query):
    body = body or {}
    sid, ent = app.save_profile(body.get("sid"), body.get("name") or "", body.get("settings"))
    return {"sid": sid, "server": ent, "servers": r_servers(app, None, {})["servers"]}


def r_server_delete(app, body, query):
    return {"deleted": app.delete_profile((body or {}).get("sid")), "servers": r_servers(app, None, {})["servers"]}


def r_fleet(app, body, query):
    app.live.touch()
    return app.fleet(max_age=0 if _q(query, "force") == "1" else 2.5)


def r_live(app, body, query):
    """GET /api/live?since=<unix s>&step=<s>: the rolling history behind the Live tab. `since` (default 15 min) keeps
    a poll small: ask only for what is newer than the last row you have; `step` averages old rows into buckets."""
    app.live.touch(fast=True)
    now = time.time()
    since = _num(_q(query, "since"))
    since = None if since is None else min(max(since, now - LIVE_KEEP_S), now)
    return app.live.snapshot(since, max(0.0, min(_num(_q(query, "step")) or 0.0, 600.0)))


def _tel(app):
    if app.telemetry is None:
        if app.tel_cfg.get("enabled") and not app.tel_error:
            raise Invalid("this PXA Control does not record history (it was started without it)")
        raise Invalid("the telemetry history is off" + (f" ({app.tel_error})" if app.tel_error else "") +
                      ": turn it on with POST /api/telemetry/settings {\"enabled\": true} or unset PXA_CONTROL_TELEMETRY=0")
    return app.telemetry


def _tel_args(query):
    keys = [k for k in (_q(query, "key") or "").split(",") if k.strip()] or None
    return keys, _num(_q(query, "since")), _num(_q(query, "until")), _num(_q(query, "step"))


def r_tel_status(app, body, query):
    """GET /api/telemetry: is the history on, where it lives, how big, which series it holds."""
    st = app.telemetry_status()
    if app.telemetry is not None:
        try:
            st["series_list"] = app.telemetry.list_series()
        except Exception as e:        # noqa: BLE001
            st["series_list"] = []
            st["error"] = st.get("error") or str(e)
    return st


def r_tel_series(app, body, query):
    """GET /api/telemetry/series?kind=card|server|host&key=a,b&since=&until=&step=&points=&source=auto|raw|1m"""
    keys, since, until, step = _tel_args(query)
    kind = _q(query, "kind", "card")
    if kind not in ("card", "server", "host"):
        raise Invalid("kind must be card, server or host")
    pts = int(min(max(_num(_q(query, "points")) or 600, 10), 5000))
    src = _q(query, "source", "auto")
    if src not in ("auto", "raw", "1m"):
        raise Invalid("source must be auto, raw or 1m")
    return _tel(app).series(kind, keys, since, until, step, max_points=pts, source=src)


def r_tel_history(app, body, query):
    """GET /api/telemetry/history?since=&until=&points=: cards, servers and the host in the shape the Live tab draws."""
    st = _tel(app)
    _k, since, until, step = _tel_args(query)
    pts = int(min(max(_num(_q(query, "points")) or 600, 10), 5000))
    c = st.series("card", None, since, until, step, max_points=pts)
    s = st.series("server", None, since, until, step, max_points=pts)
    h = st.series("host", None, since, until, step, max_points=pts)
    return {"ts": time.time(), "since": c["since"], "until": c["until"], "step": c["step"], "source": c["source"],
            "cols": {"card": c["cols"], "server": s["cols"], "host": h["cols"]},
            "cards": c["series"], "servers": s["series"], "host": (h["series"] or [None])[0]}


def r_tel_requests(app, body, query):
    """GET /api/telemetry/requests?key=&since=&until=&limit=: every finished request the store kept (raw retention)."""
    keys, since, until, _s = _tel_args(query)
    return _tel(app).requests(keys, since, until, int(_num(_q(query, "limit")) or 2000))


def r_tel_csv(app, body, query):
    """GET /api/telemetry/csv?kind=card|server|host|requests&key=&since=&until=&step=: a spreadsheet download."""
    keys, since, until, step = _tel_args(query)
    kind = _q(query, "kind", "card")
    if kind not in ("card", "server", "host", "requests"):
        raise Invalid("kind must be card, server, host or requests")
    text = _tel(app).csv_text(kind, keys, since, until, step, source=_q(query, "source", "auto"))
    name = "pxa-telemetry-%s-%s.csv" % (kind, time.strftime("%Y%m%d-%H%M", time.localtime()))
    return Reply(text, "text/csv; charset=utf-8", extra={"Content-Disposition": f'attachment; filename="{name}"'})


def r_tel_settings(app, body, query):
    return app.set_telemetry(body)


def r_metrics(app, body, query):
    """GET /metrics: Prometheus text, only when the history's prometheus setting (or PXA_CONTROL_METRICS=1) is on."""
    if TEL is None or not app.tel_cfg.get("prometheus"):
        return Reply("PXA Control /metrics is off: set PXA_CONTROL_METRICS=1 or POST /api/telemetry/settings "
                     "{\"prometheus\": true}\n", "text/plain; charset=utf-8", code=404)
    return Reply(app.metrics_text(), "text/plain; version=0.0.4; charset=utf-8")


def r_mqtt_status(app, body, query):
    """GET /api/mqtt: is the Home Assistant publisher on, is it connected, what has it published."""
    return app.mqtt_status()


def r_mqtt_settings(app, body, query):
    """POST /api/mqtt/settings: turn the publisher on/off and set the broker. The password is stored
    and reported as `password_set` only, never echoed back."""
    return app.set_mqtt(body)


def r_mqtt_test(app, body, query):
    """POST /api/mqtt/test: connect to the broker and prove the link, publishing nothing."""
    return app.mqtt_test(body)


def r_fleet_adopt(app, body, query):
    body = body or {}
    return {"adopted": app.adopt(body.get("key"), body.get("name"), bool(body.get("control")))}


def r_fleet_unadopt(app, body, query):
    return {"unadopted": app.unadopt((body or {}).get("key"))}


def r_fleet_control(app, body, query):
    body = body or {}
    return app.external_control(body.get("key"), body.get("action"), body.get("confirm"))


def r_fleet_log(app, body, query):
    try:
        tail = int(_q(query, "tail", "500"))
    except ValueError:
        tail = 500
    return app.external_log(_q(query, "key"), tail, (_q(query, "q", "") or "")[:200])


def r_engines(app, body, query):
    return app.engines(force=_q(query, "force") == "1")


def r_presets(app, body, query):
    return app.presets()


def r_preset_save(app, body, query):
    body = body or {}
    return {"saved": app.save_preset(body.get("name"), body.get("settings")), "presets": app.presets()}


def r_preset_delete(app, body, query):
    return {"deleted": app.delete_preset((body or {}).get("name")), "presets": app.presets()}


def r_attach(app, body, query):
    return {"attach_port": app.set_attach_port((body or {}).get("port")), "status": app.status()}


def r_bench_start(app, body, query):
    return app.start_bench(body)


def r_bench(app, body, query):
    # the history table pages through up to 500 runs. The temperature trace stays off the history
    # (it lives on the job). The power summary (average W, peak W, tokens/s per watt) stays on the row.
    # A result already sent to the high-score board says so (submitted), so the page does not offer it again (FR#13).
    job = dict(app.bench)
    if isinstance(job.get("result"), dict):
        job["result"] = dict(job["result"], submitted=_core._score_was_sent(job["result"].get("ts")))
    return {"job": job, "history": [{k: v for k, v in r.items() if k != "telemetry"} for r in read_jsonl("bench.jsonl", limit=500)]}


def r_history(app, body, query):
    try:
        since = float(_q(query, "since", "0"))
    except ValueError:
        since = 0.0
    return {"records": read_jsonl("history.jsonl", since=since)}


def r_report_bundle(app, body, query):
    return app.report_bundle(_q(query, "sid"), key=_q(query, "key") or None, url=_q(query, "url") or None)


def r_report_servers(app, body, query):
    return app.report_servers()


def r_score_check(app, body, query):
    return app.score_check()


def r_user(app, body, query):
    return app.user_name()


def r_user_set(app, body, query):
    return app.set_user_name(body)


def r_score_detail(app, body, query):
    return app.score_detail(_q(query, "id"))


def r_score_send(app, body, query):
    return app.score_send(body)


def r_report_send(app, body, query):
    return app.report_send(body)


def _b(body):
    return body if isinstance(body, dict) else {}


def r_enc_state(app, body, query):
    return app.encode_call("state", force=bool(_q(query, "force")))


def r_enc_rescan(app, body, query):
    return app.encode_call("rescan")


def r_enc_use(app, body, query):
    return app.encode_call("use", _b(body).get("path"))


def r_enc_add(app, body, query):
    return app.encode_call("add_encoder", _b(body).get("path"))


def r_enc_key(app, body, query):
    return app.encode_call("set_key", _b(body).get("key"))


def r_enc_get(app, body, query):
    b = _b(body)
    return app.encode_call("get_encoder", b.get("edition"), b.get("key"), bool(b.get("update")))


def r_enc_get_cancel(app, body, query):
    return app.encode_call("cancel_get")


def r_enc_runtime(app, body, query):
    """GET: the GPU runtime (cuBLAS / cuSOLVER) the Pro encoder needs: what is missing, what is installed, what the licence server offers."""
    return app.encode_call("runtime_view", None, True, bool(_q(query, "force")))


def r_enc_runtime_get(app, body, query):
    return app.encode_call("get_runtime")


def r_enc_runtime_cancel(app, body, query):
    return app.encode_call("cancel_runtime")


def r_enc_update(app, body, query):
    return app.encode_call("update_check", force=bool(_b(body).get("force")))


def r_enc_licence(app, body, query):
    return app.encode_call("licence_check")


def r_enc_inspect(app, body, query):
    return app.encode_call("inspect", _b(body).get("source"), force=bool(_b(body).get("force")))


def r_enc_plan(app, body, query):
    return app.encode_call("plan", _b(body))


def r_enc_checks(app, body, query):
    return app.encode_call("checks", _b(body))


def r_enc_start(app, body, query):
    return app.encode_call("start", _b(body))


def r_enc_job(app, body, query):
    since = _q(query, "since")
    return app.encode_call("job_view", _q(query, "id"), log_since=int(since) if since and since.isdigit() else None)


def r_enc_log(app, body, query):
    since = _q(query, "since")
    return app.encode_call("log_view", _q(query, "id"), int(since) if since and since.isdigit() else 0)


def r_enc_pause(app, body, query):
    return app.encode_call("pause", _b(body).get("id"))


def r_enc_resume(app, body, query):
    return app.encode_call("resume", _b(body).get("id"))


def r_enc_cancel(app, body, query):
    return app.encode_call("cancel", _b(body).get("id"))


def r_enc_discard(app, body, query):
    return app.encode_call("discard", _b(body).get("id"))


def r_enc_test(app, body, query):
    return app.encode_test(_b(body))


def r_enc_browse(app, body, query):
    return app.encode_call("browse", _q(query, "path"))


ROUTES = {
    ("GET", "/api/info"): r_info,
    ("GET", "/api/rig"): r_rig,
    ("GET", "/api/rig/live"): r_rig_live,
    ("GET", "/api/doctor"): r_doctor,
    ("GET", "/api/models"): r_models,
    ("GET", "/api/launch/companions"): r_companions,
    ("POST", "/api/models/dirs"): r_model_dirs,
    ("POST", "/api/models/fits"): r_fits,
    ("GET", "/api/models/available"): r_hf_available,
    ("POST", "/api/models/download"): r_hf_download,
    ("GET", "/api/models/download"): r_hf_progress,
    ("POST", "/api/models/download/cancel"): r_hf_cancel,
    ("POST", "/api/models/hf-token"): r_hf_token,
    ("GET", "/api/levers"): r_levers,
    ("POST", "/api/levers/lint"): r_levers_lint,
    ("GET", "/api/servers"): r_servers,
    ("POST", "/api/servers"): r_server_save,
    ("DELETE", "/api/servers"): r_server_delete,
    ("GET", "/api/fleet"): r_fleet,
    ("GET", "/api/live"): r_live,
    ("GET", "/api/telemetry"): r_tel_status,
    ("GET", "/api/telemetry/series"): r_tel_series,
    ("GET", "/api/telemetry/history"): r_tel_history,
    ("GET", "/api/telemetry/requests"): r_tel_requests,
    ("GET", "/api/telemetry/csv"): r_tel_csv,
    ("POST", "/api/telemetry/settings"): r_tel_settings,
    ("GET", "/metrics"): r_metrics,
    ("GET", "/api/mqtt"): r_mqtt_status,
    ("POST", "/api/mqtt/settings"): r_mqtt_settings,
    ("POST", "/api/mqtt/test"): r_mqtt_test,
    ("POST", "/api/fleet/adopt"): r_fleet_adopt,
    ("DELETE", "/api/fleet/adopt"): r_fleet_unadopt,
    ("POST", "/api/fleet/control"): r_fleet_control,
    ("GET", "/api/fleet/log"): r_fleet_log,
    ("GET", "/api/engines"): r_engines,
    ("POST", "/api/plan"): r_plan,
    ("POST", "/api/start"): r_start,
    ("POST", "/api/stop"): r_stop,
    ("POST", "/api/restart"): r_restart,
    ("GET", "/api/status"): r_status,
    ("GET", "/api/log"): r_log,
    ("GET", "/api/presets"): r_presets,
    ("POST", "/api/presets"): r_preset_save,
    ("DELETE", "/api/presets"): r_preset_delete,
    ("POST", "/api/attach"): r_attach,
    ("POST", "/api/bench"): r_bench_start,
    ("GET", "/api/bench"): r_bench,
    ("GET", "/api/history"): r_history,
    ("GET", "/api/report/bundle"): r_report_bundle,
    ("GET", "/api/report/servers"): r_report_servers,
    ("POST", "/api/report/send"): r_report_send,
    ("GET", "/api/score/check"): r_score_check,
    ("GET", "/api/thinking"): r_thinking,
    ("POST", "/api/thinking/override"): r_thinking_override,
    ("GET", "/api/thinking/table"): r_thinking_table,
    ("POST", "/api/score/send"): r_score_send,
    ("GET", "/api/score/detail"): r_score_detail,
    ("GET", "/api/user"): r_user,
    ("POST", "/api/user"): r_user_set,
    ("GET", "/api/encode/state"): r_enc_state,
    ("POST", "/api/encode/rescan"): r_enc_rescan,
    ("POST", "/api/encode/use"): r_enc_use,
    ("POST", "/api/encode/add-encoder"): r_enc_add,
    ("POST", "/api/encode/key"): r_enc_key,
    ("POST", "/api/encode/get"): r_enc_get,
    ("POST", "/api/encode/get/cancel"): r_enc_get_cancel,
    ("GET", "/api/encode/runtime"): r_enc_runtime,
    ("POST", "/api/encode/runtime/get"): r_enc_runtime_get,
    ("POST", "/api/encode/runtime/cancel"): r_enc_runtime_cancel,
    ("POST", "/api/encode/update"): r_enc_update,
    ("POST", "/api/encode/licence"): r_enc_licence,
    ("POST", "/api/encode/inspect"): r_enc_inspect,
    ("POST", "/api/encode/plan"): r_enc_plan,
    ("POST", "/api/encode/checks"): r_enc_checks,
    ("POST", "/api/encode/start"): r_enc_start,
    ("GET", "/api/encode/job"): r_enc_job,
    ("GET", "/api/encode/log"): r_enc_log,
    ("POST", "/api/encode/pause"): r_enc_pause,
    ("POST", "/api/encode/resume"): r_enc_resume,
    ("POST", "/api/encode/cancel"): r_enc_cancel,
    ("POST", "/api/encode/discard"): r_enc_discard,
    ("POST", "/api/encode/test"): r_enc_test,
    ("GET", "/api/encode/browse"): r_enc_browse,
    ("GET", "/api/update"): r_pxa_update,
    ("GET", "/api/lib/update"): r_lib_update,
    ("POST", "/api/lib/update/apply"): r_lib_update_apply,
    ("POST", "/api/lib/update/rollback"): r_lib_update_rollback,
    ("POST", "/api/lib/update/settings"): r_lib_update_settings,
    ("GET", "/api/expert-map"): r_expert_map,
    ("POST", "/api/expert-map/rebuild"): r_expert_map_rebuild,
    ("POST", "/api/expert-map/reset"): r_expert_map_reset,
    ("POST", "/api/update/apply"): r_pxa_update_apply,
    ("POST", "/api/update/channel"): r_pxa_update_channel,
    ("GET", "/api/update/beta"): r_update_beta,
    ("POST", "/api/update/beta/redeem"): r_update_beta_redeem,
    ("POST", "/api/update/beta/docker"): r_update_beta_docker,
    ("GET", "/api/update/notes"): r_pxa_update_notes,
    ("GET", "/api/health"): r_health,
    ("GET", "/api/gpu/state"): r_gpu_state,
    ("GET", "/api/gpu/profiles"): r_gpu_profiles,
    ("POST", "/api/gpu/profiles"): r_gpu_profile_save,
    ("DELETE", "/api/gpu/profiles"): r_gpu_profile_delete,
    ("POST", "/api/gpu/plan"): r_gpu_plan,
    ("POST", "/api/gpu/apply"): r_gpu_apply,
    ("POST", "/api/gpu/reset"): r_gpu_reset,
    ("POST", "/api/gpu/reserve"): r_gpu_reserve,
    ("POST", "/api/gpu/maintenance"): r_gpu_maintenance,
    ("POST", "/api/gpu/quiet"): r_gpu_quiet,
    ("GET", "/api/gpu/audit"): r_gpu_audit,
    ("GET", "/api/gpu/export"): r_gpu_export,
    ("POST", "/api/gpu/import"): r_gpu_import,
    ("POST", "/api/gpu/schedules"): r_gpu_schedules,
    ("POST", "/api/gpu/supervisor"): r_gpu_supervisor,
    ("GET", "/api/gpu/clocks"): r_gpu_clocks,
    ("POST", "/api/gpu/power"): r_gpu_power,
    ("POST", "/api/gpu/undo"): r_gpu_undo,
    ("POST", "/api/gpu/ui"): r_gpu_ui,
}

def chat_host_policy(app):
    """Host access for the chat agent (v3.1). A Control bound to this machine may offer it; a Control on
    the LAN (--lan) refuses it unless the owner has set host_access_lan in the config, because on the LAN
    the person approving a command is not necessarily the person at the keyboard. The allowlist file and
    the audit log live in the config dir; the audit record is redacted like every other Control record."""
    def gate():
        c = load_config()
        if getattr(app, "lan", False) and not c.get("host_access_lan"):
            return False, ("host access is off over the network. Control is reachable from other machines, "
                           "so this stays off until you set host_access_lan in the Control config.")
        return True, ""
    return CHAT.HostPolicy(gate=gate, allow_path=os.path.join(config_dir(), "host-allow.json"),
                           audit=lambda rec: append_jsonl("host-audit.jsonl", redact_obj(rec)))


if CHAT is not None:                     # the chat agent's routes + agent.js/agent.css: one ROUTES.update()/STATIC.update() block
    try:
        CHAT.register(ROUTES, STATIC, Reply, config_dir, load_config, host=chat_host_policy, who=_core.req_who)
        BODY_LIMITS.update(getattr(CHAT, "BODY_LIMITS", None) or {})
    except Exception as _chat_err:       # noqa: BLE001
        CHAT, CHAT_IMPORT_ERROR = None, f"{_chat_err.__class__.__name__}: {_chat_err}"
        _core.CHAT, _core.CHAT_IMPORT_ERROR = CHAT, CHAT_IMPORT_ERROR     # App reads them from core

# PXA Store, local side (PLUGINS-STORE-DESIGN.md): GET /api/store/catalog (fixture) and GET /api/store/entitlements
from . import store as _store  # noqa: E402
_store.register(ROUTES, config_dir)
from .plugins import host as _plughost  # noqa: E402
_plughost.register(ROUTES)                  # /api/plugins/host*: the plugin host (S2); /api/plugins/<id>/* is server.py's forward

# the FREE shareable bench card (todo control-share-report): its own page /share.html + GET /api/share/*; reads the bench history only
from . import share as _share  # noqa: E402
_share.register(ROUTES, STATIC)


def _store_urls():
    return _store.announced()


__all__ = [_k for _k in list(globals()) if not _k.startswith("__")]   # star-import carries the _private names too
