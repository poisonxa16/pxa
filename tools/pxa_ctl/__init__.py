"""pxa_ctl: PXA Control's GPU profile backend (v3.1, 2026-10-08).

Layers, each usable and testable on its own (stdlib only):
    driver      GPU adapters: nvidia-smi, NVML (pynvml), a mock for tests, a null one when there is no driver
    store       profiles, presets, schedules, reserved cards, quiet-mode state (JSON, schema-versioned, validated)
    guard       maintenance mode, reserved cards and lock files: who may NOT be touched right now
    audit       append-only record of every change (who, when, before, after, result)
    engine      dry-run diff, apply, readback verify, rollback, reset
    supervisor  auto-start of servers: state machine, health wait, backoff, crash-loop detection
    schedule    time-of-day profile switches
    service     GpuControl: the facade PXA Control's routes call

Everything that changes a GPU or starts a server is OFF unless the owner's config switches it on
(allow_gpu_control / gpu_autostart / gpu_schedules); the page alone can never switch it on.
"""
from .errors import CtlError, Refused, Locked, Invalid  # noqa: F401

VERSION = "1"
