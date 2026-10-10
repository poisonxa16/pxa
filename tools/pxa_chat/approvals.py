"""Server-side pending-call registry for the chat agent.

Only calls this server issued can be answered (a random id per pending call, bound to its run), each answer is
"once", "always" (this chat, this tool; never offered for shell) or "deny", and no answer within the timeout
is a deny. Same shape as the Mythos park-and-resume approvals, in memory (a pending call dies with its run).

Host access grants are "always" scopes with a `host:` prefix. They are deliberately the same in-memory dict:
nothing about host access is ever written to disk, so a Control restart, a new chat, or the host switch going
off all drop the grants by construction. A host grant names ONE resolved argv or ONE resolved path -- a grant
for `nvidia-smi -L` does not carry a later `nvidia-smi -L --query`, it just means the next card is one click.
"""
import secrets
import threading
import time

DECISIONS = ("once", "always", "deny")
NEVER_ALWAYS = {"run_command"}
HOST_PREFIX = "host:"


def is_host_scope(scope):
    return isinstance(scope, str) and scope.startswith(HOST_PREFIX)


class Approvals(object):
    def __init__(self, timeout_s=300.0):
        self.timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        self._pending = {}          # id -> {run_id, chat_id, tool, scope, ev, decision, ts}
        self._always = {}           # chat_id -> {scope}

    def allowed(self, chat_id, scope):
        with self._lock:
            return scope in self._always.get(chat_id, set())

    def issue(self, run_id, chat_id, tool, scope=None):
        aid = "ap_" + secrets.token_urlsafe(12)
        with self._lock:
            self._pending[aid] = {"run_id": run_id, "chat_id": chat_id, "tool": tool, "scope": scope or tool,
                                  "ev": threading.Event(), "decision": None, "ts": time.time()}
        return aid

    def answer(self, aid, run_id, decision):
        """-> (ok, code). code: unknown_approval | wrong_run | bad_decision | not_allowed."""
        if decision not in DECISIONS:
            return False, "bad_decision"
        with self._lock:
            p = self._pending.get(aid)
            if p is None:
                return False, "unknown_approval"
            if p["run_id"] != run_id:
                return False, "wrong_run"
            if decision == "always" and p["tool"] in NEVER_ALWAYS:
                return False, "not_allowed"
            if p["decision"] is not None:
                return False, "unknown_approval"
            p["decision"] = decision
            if decision == "always":
                self._always.setdefault(p["chat_id"], set()).add(p["scope"])
            p["ev"].set()
        return True, "ok"

    def wait(self, aid, cancel_ev=None, timeout_s=None):
        """-> 'once' | 'always' | 'deny' | 'timeout' | 'cancelled'. The id is gone afterwards."""
        with self._lock:
            p = self._pending.get(aid)
        if p is None:
            return "deny"
        deadline = time.time() + (self.timeout_s if timeout_s is None else timeout_s)
        try:
            while True:
                if p["ev"].wait(0.2):
                    return p["decision"]
                if cancel_ev is not None and cancel_ev.is_set():
                    return "cancelled"
                if time.time() >= deadline:
                    return "timeout"
        finally:
            with self._lock:
                self._pending.pop(aid, None)

    def pending_for(self, run_id):
        with self._lock:
            return [k for k, v in self._pending.items() if v["run_id"] == run_id]

    def forget_chat(self, chat_id):
        with self._lock:
            self._always.pop(chat_id, None)

    # ---- host-access grants (session-only, never persisted) ------------------------------------
    def host_scopes(self, chat_id):
        """-> the host scopes this chat may run without a card. Test/UI helper."""
        with self._lock:
            return sorted(s for s in self._always.get(chat_id, set()) if is_host_scope(s))

    def forget_host(self, chat_id):
        """drop this chat's host grants only; its other 'always' grants keep working."""
        with self._lock:
            if chat_id in self._always:
                keep = {s for s in self._always[chat_id] if not is_host_scope(s)}
                if keep:
                    self._always[chat_id] = keep
                else:
                    self._always.pop(chat_id, None)

    def forget_all_host(self):
        """the host switch moved (either way): every room's host grants go, nothing else does."""
        with self._lock:
            for cid in list(self._always):
                keep = {s for s in self._always[cid] if not is_host_scope(s)}
                if keep:
                    self._always[cid] = keep
                else:
                    self._always.pop(cid, None)
