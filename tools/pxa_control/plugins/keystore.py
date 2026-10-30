"""Control's key store for plugins (PLUGINS-PROTOTYPE-PLAN S4): a plugin keeps a sign-in key (a YouTube refresh token, an API key)
HERE, by name, never in its own files or settings. Settings hold a reference ("secret:<name>"); the value goes to the plugin only
through the host call secrets.get, over its authenticated channel.

* Scope: one file per plugin, <config>/plugins/secrets/<plugin_id>.json (0600, folder 0700); a plugin reaches only its own names
  (the host passes the caller's id, never the plugin's word for it). Uninstall leaves it for the user to clear (Forget keys).
* Never printed: no value in an audit row (name and size only), an answer to the tab, a log line, an exception text or a repr.
* Limits: 16 names per plugin, a name is 1-48 of a-z 0-9 . _ -, a value is a string up to 8 KiB.
Honest limit (v1): the file is protected by its mode, as Control's own HF token is; an OS keychain (DPAPI, Keychain, libsecret) is a
later backend behind the same calls.
"""
import json
import os
import re
import threading

MAX_NAMES = 16
MAX_VALUE = 8 << 10
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,47}$")
REF_PREFIX = "secret:"


class KeyStoreError(ValueError):
    def __init__(self, msg, code="invalid", status=400):
        ValueError.__init__(self, msg)
        self.code, self.status = code, status


def ref(name):
    """the reference a plugin keeps in its settings instead of the value."""
    return REF_PREFIX + name


def is_ref(value):
    return isinstance(value, str) and value.startswith(REF_PREFIX) and bool(NAME_RE.match(value[len(REF_PREFIX):]))


class KeyStore(object):
    def __init__(self, cfg_dir):
        self.dir = os.path.join(cfg_dir, "plugins", "secrets")
        self.lock = threading.Lock()

    def __repr__(self):
        return "<KeyStore>"

    def _path(self, pid):
        return os.path.join(self.dir, pid + ".json")

    def _load(self, pid):
        try:
            with open(self._path(pid)) as f:
                d = json.load(f)
            return {k: v for k, v in d.items() if isinstance(k, str) and isinstance(v, str)} if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, pid, d):
        os.makedirs(self.dir, mode=0o700, exist_ok=True)
        os.chmod(self.dir, 0o700)
        p = self._path(pid)
        tmp = p + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(d, f, sort_keys=True)
        os.replace(tmp, p)

    @staticmethod
    def _name(name):
        n = str(name or "")
        if n.startswith(REF_PREFIX):
            n = n[len(REF_PREFIX):]
        if not NAME_RE.match(n):
            raise KeyStoreError("name: 1-48 of a-z 0-9 . _ - (or secret:<name>)")
        return n

    def put(self, pid, name, value):
        n = self._name(name)
        if not isinstance(value, str) or not value or len(value.encode("utf-8")) > MAX_VALUE:
            raise KeyStoreError("value: a string of 1-%d bytes" % MAX_VALUE)
        with self.lock:
            d = self._load(pid)
            if n not in d and len(d) >= MAX_NAMES:
                raise KeyStoreError("at most %d keys per plugin" % MAX_NAMES, "too_many", 409)
            d[n] = value
            self._save(pid, d)
        return {"ref": ref(n), "bytes": len(value.encode("utf-8"))}

    def get(self, pid, name):
        n = self._name(name)
        with self.lock:
            v = self._load(pid).get(n)
        if v is None:
            raise KeyStoreError("no key named %s" % n, "not_found", 404)
        return v

    def delete(self, pid, name):
        n = self._name(name)
        with self.lock:
            d = self._load(pid)
            had = d.pop(n, None) is not None
            if had:
                self._save(pid, d)
        return {"deleted": had}

    def names(self, pid):
        with self.lock:
            return sorted(self._load(pid))

    def forget(self, pid):
        """the user's Forget keys: every key of this plugin."""
        with self.lock:
            try:
                os.unlink(self._path(pid))
                return True
            except OSError:
                return False

    # ---- host calls ---------------------------------------------------------------------------------------------------
    def call(self, pid, name, args):
        """secrets.put {name, value} -> {ref, bytes}; secrets.get {name} -> {value}; secrets.delete {name}; secrets.names."""
        if name == "secrets.put":
            return self.put(pid, args.get("name"), args.get("value"))
        if name == "secrets.get":
            return {"value": self.get(pid, args.get("name"))}
        if name == "secrets.delete":
            return self.delete(pid, args.get("name"))
        if name == "secrets.names":
            return {"names": self.names(pid)}
        raise KeyStoreError("no such key-store call", "unknown_call", 404)
