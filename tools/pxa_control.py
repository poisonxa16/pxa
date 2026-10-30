"""Compatibility shim. PXA Control is now the package tools/pxa_control/ (2026-10-14).

`import pxa_control` with tools/ on sys.path gets the package (a package directory wins over a module of the same
name), so nothing that imports it changes. This file stays for callers that load tools/pxa_control.py by PATH
(importlib spec_from_file_location, runpy, release file lists, `python3 tools/pxa_control.py`): it loads the package
and re-exports its names. Running it starts Control like the compiled pxa-control binary does.
"""
import importlib.util as _ilu
import os as _os
import sys as _sys

# No __pycache__ next to Control's own sources (review N3): bytecode writing off before the package loads, and for the
# python children Control starts.
_sys.dont_write_bytecode = True
_os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

_PKG_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "pxa_control")
_pkg = _sys.modules.get("pxa_control")
if _pkg is None or not hasattr(_pkg, "__path__"):          # not loaded yet, or the name is held by this shim
    _spec = _ilu.spec_from_file_location("pxa_control", _os.path.join(_PKG_DIR, "__init__.py"),
                                         submodule_search_locations=[_PKG_DIR])
    _pkg = _ilu.module_from_spec(_spec)
    _sys.modules["pxa_control"] = _pkg
    _spec.loader.exec_module(_pkg)
globals().update({_k: _v for _k, _v in vars(_pkg).items() if not _k.startswith("__")})

if __name__ == "__main__":
    from pxa_control import desktop as _desktop
    _sys.exit(_desktop.main())
