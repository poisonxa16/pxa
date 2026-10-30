"""PXA Control - the launcher's browser front end (`pxa`, `pxa-launch --gui`), as a package.

The modules, and what each owns (the stable internal API is what this file re-exports; everything that imported
the old single-file tools/pxa_control.py keeps working through these names):

    core      config, levers, launch validation, the seats (Seat), the Live sampler, App (the whole model)
    auth      the security policy: fetch-metadata/Origin checks, CSP + headers, the token throttle, scrubbing
    bridge    the launcher bridge: auto-start rule, run records, find_running / ensure_control
    routes    the HTTP API: r_<name>(app, body, query), ROUTES, the STATIC page map, the chat agent's routes
    server    the request handler, the bounded threading server, serve()
    plugins   first-party plugins: manifest loader + validator + consent records (stubs, PLUGINS-STORE-DESIGN.md)
    store     PXA Store, local side: signed-grant verifier, entitlement cache, catalog/entitlement routes (stubs)
    desktop   the double-click entry: start, open the browser signed in, tray icon when one is available

`python -m pxa_control` (and the compiled pxa-control binary) runs desktop.main().

Setting an attribute on this package (a test's `C.ENC = fake`, mock.patch.object(C, ...)) sets it in every
submodule that has that name, so a patch reaches the code that reads it, as it did when Control was one file.
"""
import sys as _sys

# No __pycache__ next to Control's own sources (review N3): the launchers turn bytecode writing off before this file
# loads (python3 -B / PYTHONDONTWRITEBYTECODE=1 / the entry modules); this keeps it off for every submodule whatever the entry.
_sys.dont_write_bytecode = True
import types as _types

from . import core, auth, bridge, routes, server          # noqa: E402  (order: each imports the ones before it)
from .core import *  # noqa: F401,F403
from .auth import *  # noqa: F401,F403
from .bridge import *  # noqa: F401,F403
from .routes import *  # noqa: F401,F403
from .server import *  # noqa: F401,F403
from . import plugins, store, desktop                     # noqa: E402,F401

_SUBMODULES = (core, auth, bridge, routes, server)


class _Package(_types.ModuleType):
    """forward attribute writes to the submodules that hold the name (see the docstring)."""

    def __setattr__(self, name, value):
        if not name.startswith("__"):
            for m in _SUBMODULES:
                if name in m.__dict__:
                    m.__dict__[name] = value
        super().__setattr__(name, value)

    def __delattr__(self, name):
        for m in _SUBMODULES:
            m.__dict__.pop(name, None)
        super().__delattr__(name)


_sys.modules[__name__].__class__ = _Package
