"""The ONE unlock gate for every supporter plugin and supporter feature in PXA Control (PLUGINS-PROTOTYPE-PLAN S1, S0).

Run authority is a grant licd signed for THIS install and nothing else: store.EntitlementCache keeps it in
<config>/store/grants.json (Ed25519 over b'pxa-plugin-ent|' + canonical statement, v 1, iss licd, the install key's public
half, issued .. not_after with 5 min skew, a clock more than 48 h behind the newest time seen counts as nothing, the signed
revocation list). It is machine-bound, it expires and it can be revoked. The saved wallet statement and a live wallet view
are account-wide and never expire, so they only inform the panel (panel() below), never a run (security review F2/F3).

One gate, two readers: the closed engine library reads the same grants file with the same rules for kind
"feature" items (ggml/src/pxqn-feature.cpp, Booster P7); tests/test-pxa-plugin-gate.py replays that reader's cases here so
both answer alike. Control code asks state() / run_allowed() / require(); nothing else reads grants for a run.

Costs are never in plugin or Control code: cost() reads the item's row of the signed catalogue (licd's, in coins since
_coins_v1; the fixture otherwise). unlock() is the coins panel's own path: the signed wallet redeem, then the saved
statement, then sync() pulls the grant for this computer.
"""
from . import PluginError

RUNS = ("active", "refresh_due")
# what the panel says per state; {name} is the item's name. Every Store item unlocks with PXA coins (owner 10-24);
# the Store wording rules apply (donations, coins, unlock).
MESSAGES = {
    "active": "",
    "refresh_due": "",
    "missing": "{name} unlocks with PXA coins in the Store.",
    "expired": "{name} needs one connection to the Store to renew it on this computer.",
    "revoked": "{name} is no longer unlocked on this computer.",
    "wrong_machine": "{name} is unlocked for another computer. Open Store, PXA Coins, and sync this one.",
    "clock": "This computer's clock is behind the last time PXA Control saw. Set the clock right to use {name}.",
    "invalid": "{name}'s unlock on this computer could not be checked. Open Store, PXA Coins, and sync.",
    "statement": "{name} is unlocked on your wallet. Open Store, PXA Coins, and sync once to finish on this computer.",
}
# when no grant runs, the most useful reason to show (first match wins)
_REASON_ORDER = ("revoked", "expired", "wrong_machine", "invalid")


class Locked(PluginError):
    """a supporter plugin or feature asked to run without a grant for this computer."""

    def __init__(self, msg, state):
        super().__init__(msg, "locked")
        self.status = 403
        self.state = state


def _cache(cfg_dir, cache):
    from .. import store as S
    return cache if cache is not None else S.EntitlementCache(cfg_dir)


def state(cfg_dir, item_id, kind="plugin", cache=None):
    """-> {"state": active|refresh_due|missing|expired|revoked|wrong_machine|clock|invalid, "item_id", "kind",
    "ent_id"?, "not_after"?, "refresh_after"?}. A run is allowed only in RUNS. The runnable answer IS
    EntitlementCache.active() (same verifier, same clock record), filtered to this item AND this kind (a feature grant never
    runs a plugin of the same name, and the other way round); the other states only say why not."""
    from .. import store as S
    c = _cache(cfg_dir, cache)
    out = {"item_id": item_id, "kind": kind}
    for st, status in c.active():
        if st["item_id"] == item_id and st["kind"] == kind:
            return dict(out, state="active" if status == "ok" else "refresh_due", ent_id=st["ent_id"][:12],
                        not_after=int(st["not_after"]), refresh_after=int(st["refresh_after"]))
    d = c.record()                          # under the store lock, merged with the lock journal (RC3.6 item 4)
    if c.clock() < float(d.get("seen") or 0) - S.ROLLBACK_S:
        return dict(out, state="clock")
    _seed, pub = S.install_key(c.cfg_dir)
    why = set()
    for ent_id, env in (d.get("grants") or {}).items():
        st = env.get("statement") if isinstance(env, dict) else None
        if not isinstance(st, dict) or st.get("item_id") != item_id or st.get("kind") != kind:
            continue
        if ent_id in (d.get("revoked") or []) or st.get("ent_id") in (d.get("revoked") or []):
            why.add("revoked")
            continue
        try:
            S.verify_grant(env, c.pubkeys, c.clock(), pub)
        except S.GrantError as e:
            why.add(e.code if e.code in ("expired", "wrong_machine") else "invalid")
    hl = d.get("hard_locks") or {}
    if S.EntitlementCache._hard_locked(d, item_id) and _lock_says_why(c, d, item_id):
        # a 403 at a sync (S6) or a damaged store file: locked until licd grants this item afresh
        code = str(hl.get(item_id) or hl.get("*") or d.get("hard_lock") or "")[:64]
        return dict(out, state="invalid" if code == "corrupt" else "revoked", hard_lock=code)
    if item_id in (d.get("locks") or {}):              # licd's account answer says the item is no longer active
        return dict(out, state="revoked", lock=str(d["locks"][item_id])[:64])
    for r in _REASON_ORDER:
        if r in why:
            return dict(out, state=r)
    return dict(out, state="missing")


def _lock_says_why(c, d, item_id):
    """review 10-26: whether a hard-locked item's reason is the lock itself. Wording only: a hard-locked item
    never runs either way (active() decides runs). L1: once licd has signed this computer a fresh grant after the lock
    (EntitlementCache.account_lock_lifted, v5's own evidence), an item still locked is one licd did not grant again:
    its reason is what licd's answer says (locks: revoked / ...), or missing, never the old 403. L2: under a damaged
    file's "*" an item outside the held set the lock journal kept (star_held) was never on this computer: it reads
    missing (unlocks with PXA coins), as on the main line; with the held set unknown (star_held None) every item keeps the
    corruption reason until licd answers (fail closed)."""
    if c.account_lock_lifted(d):
        return False
    hl = d.get("hard_locks") or {}
    if item_id in hl:
        return True
    sh = d.get("star_held")
    return not (isinstance(sh, list) and item_id not in sh)


def run_allowed(cfg_dir, item_id, kind="plugin", cache=None):
    """THE check before any supporter plugin or feature does its work."""
    return state(cfg_dir, item_id, kind, cache)["state"] in RUNS


def require(cfg_dir, item_id, kind="plugin", cache=None, name=None):
    """-> the state dict when it may run; raises Locked (403, code locked) with the panel's words otherwise."""
    s = state(cfg_dir, item_id, kind, cache)
    if s["state"] not in RUNS:
        raise Locked(MESSAGES[s["state"]].format(name=name or item_id), s)
    return s


def cost(item):
    """coins for an item from ITS catalogue row (the signed licd catalogue, or the fixture), or None. The amount lives only
    there; a row says it as `coins` or as the API field store.CATALOG_COST_KEY (never shown to a person as a word)."""
    from .. import store as S
    if isinstance(item, dict):
        for k in ("coins", S.CATALOG_COST_KEY):
            v = item.get(k)
            if isinstance(v, bool):
                continue
            if isinstance(v, int) or (isinstance(v, str) and v.isdigit()):
                return int(v)
    return None


def catalog_row(catalog, item_id):
    """the row for item_id in a catalogue ({items: [...]} as catalog_view / catalog_from_licd return it, or a list)."""
    items = catalog.get("items") if isinstance(catalog, dict) else catalog
    for it in items or []:
        if isinstance(it, dict) and it.get("id") == item_id:
            return it
    return None


def _owned_in(owned, item_id):
    for o in owned or []:
        if o == item_id or (isinstance(o, dict) and o.get("item_id") == item_id and o.get("status") not in ("revoked", "removed")):
            return True
    return False


def panel(cfg_dir, item_id, kind="plugin", cache=None, view=None, name=None):
    """the panel's view -> {active, how: grant|statement|wallet|None, state, item_id, have, message}. Only how == "grant" runs.
    view is the wallet object licd JUST returned to the coins panel (redeem / view answer), for the moment before sync stores
    the grant. It is not signed: never pass a client's or a request's dict here, and never use it to run anything."""
    from .. import store as S
    nm = name or item_id
    s = state(cfg_dir, item_id, kind, cache)
    base = {"item_id": item_id, "state": s["state"]}
    if s["state"] in RUNS:
        return dict(base, active=True, how="grant", have=None, message="")
    st = S.load_statement(cfg_dir)
    have = int(st.get("coins") or 0) if st and st.get("coins") is not None else None
    if st and _owned_in([i for i in (st.get("owned") or []) if isinstance(i, str)], item_id):
        return dict(base, active=True, how="statement", have=have, message=MESSAGES["statement"].format(name=nm))
    if isinstance(view, dict) and _owned_in(view.get("owned"), item_id):
        h = view.get("coins")
        return dict(base, active=True, how="wallet", have=None if h is None else int(h), message=MESSAGES["statement"].format(name=nm))
    if have is None and isinstance(view, dict) and view.get("coins") is not None:
        have = int(view.get("coins") or 0)
    msg = MESSAGES[s["state"]].format(name=nm)
    if s["state"] == "missing" and have is not None:
        msg += " This wallet has %d coins." % have
    return dict(base, active=False, how=None, have=have, message=msg)


def unlock(client, cfg_dir, item_id, cache=None):
    """spend the coins for item_id through the signed wallet call (client = store.LicdClient), keep the statement, then
    pull this computer's grant. A 402 not_enough_coins (or any licd refusal) is raised as licd sent it: nothing unlocks.
    A statement or sync miss after the coins moved does not hide the wallet answer (the grant follows on the next sync)."""
    from .. import store as S
    out = client.wallet("redeem", {"item_id": item_id})
    try:
        S.keep_statement(client, cfg_dir, force=True)
    except Exception:  # noqa: BLE001 - the wallet answer stands if the statement read fails
        pass
    try:
        synced = client.sync(_cache(cfg_dir, cache))
    except Exception:  # noqa: BLE001 - same as the coins panel: coins already moved
        synced = None
    if isinstance(out, dict) and synced is not None:
        out = dict(out, synced=synced)
    return out
