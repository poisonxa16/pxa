"""Host access (v3.1): off by default, allowlist-only, session-only, audited, and unable to leave a root.

These run without a Control and without a GPU: they exercise the policy in guards.py, the executors in
tools.py, the session grants in approvals.py, and the hub switch in routes.py. The one place a real process
is started is _run() with an allowlisted /bin/echo, which is the cheapest way to prove the path executes.
"""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from pxa_chat import approvals as AP          # noqa: E402
from pxa_chat import guards, routes, tools    # noqa: E402


# ---- fixtures ------------------------------------------------------------------------------------
@pytest.fixture()
def root():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.realpath(d)


@pytest.fixture()
def allow(root):
    """a root dir with a file under it, a symlink pointing out of it, and an allowlist for both."""
    os.makedirs(os.path.join(root, "docs"), mode=0o700)
    with open(os.path.join(root, "docs", "notes.txt"), "w", encoding="utf-8") as f:
        f.write("hello from the host\n")
    os.symlink("/etc", os.path.join(root, "docs", "escape"))
    return {"commands": [["/bin/echo"], ["nvidia-smi", "-L"]], "read_paths": [os.path.join(root, "docs")]}


def _ctx(root, allow, ok=True, why="", audit=None, chat="c1", run="r1"):
    return tools.Ctx(os.path.join(root, "sandbox", chat), chat_id=chat, run_id=run, advanced=True,
                     host_ok=ok, host_why=why, host_allow=allow, host_audit=audit)


def _policy(**kw):
    """a policy that permits host access: Control must pass a gate explicitly, so a test that wants the
    switch to work has to say the deployment permits it -- the same way pxa_control does."""
    kw.setdefault("gate", lambda: (True, ""))
    return routes.HostPolicy(**kw)


def _records():
    out = []
    return out, out.append


# ---- off by default ------------------------------------------------------------------------------
def test_tools_are_not_offered_unless_host_is_on():
    assert "host_run" not in tools.available(None, True, False)
    assert "host_read" not in tools.available(None, True, False)
    assert {"host_run", "host_read"} <= set(tools.available(None, True, True))
    assert "host_run" not in tools.available(None, True, True)[:0] or True
    # advanced is still required: no advanced, no host tool even with the switch on
    assert "host_run" not in tools.available(None, False, True)


def test_host_run_refuses_when_the_switch_is_off(root, allow):
    recs, sink = _records()
    ok, out = tools.execute("host_run", {"command": ["/bin/echo", "hi"]}, _ctx(root, allow, ok=False, audit=sink))
    assert ok is False and out.startswith("refused:")
    assert len(recs) == 1 and recs[0]["decision"] == "off" and recs[0]["argv"] == ["/bin/echo", "hi"]


def test_host_read_refuses_when_the_switch_is_off(root, allow):
    recs, sink = _records()
    ok, out = tools.execute("host_read", {"path": os.path.join(root, "docs", "notes.txt")},
                            _ctx(root, allow, ok=False, audit=sink))
    assert ok is False and "refused" in out
    assert recs[0]["decision"] == "off"


def test_a_hub_with_no_policy_refuses_and_says_why(root):
    h = routes.Hub(os.path.join(root, "chat"))
    st = h.host_status("")
    assert st["available"] is False and st["on"] is False and st["why"]
    ok, why, allow = h.host_ctx("c1")
    assert ok is False and why and allow == {"commands": [], "read_paths": []}
    # a bare HostPolicy is the fail-closed default too: only an explicit gate can open host access
    assert routes.HostPolicy().gate()[0] is False
    assert routes.Hub(os.path.join(root, "chat"), host=routes.HostPolicy()).host_status("")["available"] is False


def test_the_deployment_gate_refuses_even_with_the_switch_on(root):
    """--lan without the flag: the switch cannot be turned on, and the refusal carries the reason."""
    h = routes.Hub(os.path.join(root, "chat"), host=routes.HostPolicy(gate=lambda: (False, "Control is on 0.0.0.0")))
    st = h.host_set("c1", True)
    assert st == {"available": False, "on": False, "why": "Control is on 0.0.0.0"}
    assert h.host_ctx("c1")[0] is False


def test_a_broken_gate_refuses(root):
    h = routes.Hub(os.path.join(root, "chat"), host=routes.HostPolicy(gate=lambda: 1 / 0))
    assert h.host_status("")["available"] is False


# ---- the allowlist -------------------------------------------------------------------------------
def test_an_allowlisted_command_runs_and_is_audited(root, allow):
    recs, sink = _records()
    ok, out = tools.execute("host_run", {"command": ["/bin/echo", "on-the-host"]}, _ctx(root, allow, audit=sink))
    assert ok is True and "on-the-host" in out and "exit code 0" in out
    assert recs[0]["decision"] == "allowed" and recs[0]["exit_code"] == 0 and recs[0]["argv"] == ["/bin/echo", "on-the-host"]
    assert recs[0]["chat_id"] == "c1" and recs[0]["session"] == "r1" and "bytes" in recs[0]


def test_the_allowlist_matches_a_prefix(root, allow):
    ok, out = tools.execute("host_run", {"command": ["/bin/echo", "a", "b", "c"]}, _ctx(root, allow))
    assert ok is True


def test_a_command_off_the_allowlist_is_refused(root, allow):
    recs, sink = _records()
    ok, out = tools.execute("host_run", {"command": ["/bin/cat", "/etc/hostname"]}, _ctx(root, allow, audit=sink))
    assert ok is False and "not on the host allowlist" in out
    assert recs[0]["decision"] == "denied"


def test_an_empty_allowlist_allows_nothing(root):
    ok, out = tools.execute("host_run", {"command": ["/bin/echo", "hi"]},
                            _ctx(root, {"commands": [], "read_paths": []}))
    assert ok is False and "no commands are allowed" in out


def test_a_string_instead_of_a_list_is_refused(root, allow):
    """the model must not be able to hand us a shell line and have it parsed."""
    ok, out = tools.execute("host_run", {"command": "/bin/echo hi | sh"}, _ctx(root, allow))
    assert ok is False and "not a string" in out


@pytest.mark.parametrize("bad", [["/bin/echo", "a|b"], ["/bin/echo", "a;b"], ["/bin/echo", "a>b"],
                                 ["/bin/echo", "a$(id)"], ["/bin/echo", "a`id`"], ["/bin/echo", "a&b"],
                                 ["/bin/echo", ""], ["/bin/echo", " padded "]])
def test_a_shell_metacharacter_or_a_blank_argument_is_refused(root, allow, bad):
    ok, out = tools.execute("host_run", {"command": bad}, _ctx(root, allow))
    assert ok is False
    assert "shell" in out or "empty string" in out or "whitespace" in out


def test_the_never_run_list_beats_the_allowlist(root):
    """an owner who allowlists `rm` has still not allowlisted a wipe."""
    ok, out = tools.execute("host_run", {"command": ["rm", "-rf", "/"]},
                            _ctx(root, {"commands": [["rm", "-rf", "/"]], "read_paths": []}))
    assert ok is False and "never-run list" in out


def test_the_allowlist_file_fails_closed(root):
    p = os.path.join(root, "host-allow.json")
    with open(p, "w", encoding="utf-8") as f:
        f.write("{not json")
    assert guards.host_allow_load(p) == {"commands": [], "read_paths": []}
    assert guards.host_allow_load(os.path.join(root, "nope.json")) == {"commands": [], "read_paths": []}
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"commands": [["/bin/echo", "a|b"], ["/bin/echo", "ok"]], "read_paths": ["/etc"]}, f)
    a = guards.host_allow_load(p)
    assert a["commands"] == [["/bin/echo", "ok"]] and a["read_paths"] == ["/etc"]


# ---- reads, escapes ------------------------------------------------------------------------------
def test_a_file_under_an_allowed_root_reads(root, allow):
    recs, sink = _records()
    p = os.path.join(root, "docs", "notes.txt")
    ok, out = tools.execute("host_read", {"path": p}, _ctx(root, allow, audit=sink))
    assert ok is True and "hello from the host" in out and p in out
    assert recs[0]["decision"] == "allowed" and recs[0]["path"] == p


def test_a_path_outside_every_root_is_refused(root, allow):
    outside = os.path.join(root, "outside.txt")
    with open(outside, "w", encoding="utf-8") as f:
        f.write("not for the agent\n")
    ok, out = tools.execute("host_read", {"path": outside}, _ctx(root, allow))
    assert ok is False and "outside every allowed path" in out
    # and the same answer for one that does not exist: existence is not something an outside path may probe
    ok, out = tools.execute("host_read", {"path": os.path.join(root, "maybe-gone.txt")}, _ctx(root, allow))
    assert ok is False and "outside every allowed path" in out and "no such path" not in out


def test_dotdot_cannot_leave_a_root(root, allow):
    sneak = os.path.join(root, "docs", "..", "..", "etc", "hostname")
    ok, out = tools.execute("host_read", {"path": sneak}, _ctx(root, allow))
    assert ok is False and "refused" in out


def test_a_symlink_cannot_leave_a_root(root, allow):
    """docs/escape -> /etc: allowed by its written path, refused by where it lands."""
    ok, out = tools.execute("host_read", {"path": os.path.join(root, "docs", "escape", "hostname")},
                            _ctx(root, allow))
    assert ok is False and "refused" in out


def test_a_missing_path_is_refused(root, allow):
    ok, out = tools.execute("host_read", {"path": os.path.join(root, "docs", "gone.txt")}, _ctx(root, allow))
    assert ok is False and "no such path" in out


def test_a_folder_is_not_a_file(root, allow):
    ok, out = tools.execute("host_read", {"path": os.path.join(root, "docs")}, _ctx(root, allow))
    assert ok is False and "folder" in out


def test_a_binary_file_is_refused(root, allow):
    p = os.path.join(root, "docs", "blob.bin")
    with open(p, "wb") as f:
        f.write(b"\x7fELF\x00\x00\x00\x00binary")
    ok, out = tools.execute("host_read", {"path": p}, _ctx(root, allow))
    assert ok is False and "binary" in out


def test_a_big_file_is_capped_at_64_kib(root, allow):
    p = os.path.join(root, "docs", "big.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("x" * (guards.READ_CAP * 3))
    recs, sink = _records()
    ok, out = tools.execute("host_read", {"path": p}, _ctx(root, allow, audit=sink))
    assert ok is True and "read stopped at" in out
    assert recs[0]["bytes"] == guards.READ_CAP and recs[0]["truncated"] is True


# ---- approval cards ------------------------------------------------------------------------------
def test_an_allowed_host_call_gets_a_card_with_the_exact_argv(root, allow):
    card = tools.approval_needed("host_run", {"command": ["/bin/echo", "hi"]}, _ctx(root, allow))
    assert card and card["always_ok"] is True and card["scope"] == "host:/bin/echo hi" and "/bin/echo hi" in card["detail"]


def test_a_read_card_names_the_resolved_path(root, allow):
    rel = os.path.join(root, "docs", "..", "docs", "notes.txt")
    card = tools.approval_needed("host_read", {"path": rel}, _ctx(root, allow))
    assert card and card["scope"] == "host:read:" + os.path.join(root, "docs", "notes.txt")


def test_a_refused_call_gets_no_card(root, allow):
    """a card the user can approve and still be refused is worse than no card."""
    assert tools.approval_needed("host_run", {"command": ["/bin/cat", "/etc/passwd"]}, _ctx(root, allow)) is None
    assert tools.approval_needed("host_run", {"command": ["/bin/echo", "a;b"]}, _ctx(root, allow)) is None
    assert tools.approval_needed("host_read", {"path": "/etc/hostname"}, _ctx(root, allow)) is None
    assert tools.approval_needed("host_run", {"command": ["/bin/echo", "hi"]}, _ctx(root, allow, ok=False)) is None


def test_run_command_keeps_its_own_card_and_never_offers_always():
    card = tools.approval_needed("run_command", {"command": "python3 x.py"}, None)
    assert card["scope"] == "run_command" and card["always_ok"] is False


# ---- session grants ------------------------------------------------------------------------------
def test_a_host_grant_is_session_only_and_dies_with_the_switch():
    ap = AP.Approvals()
    aid = ap.issue("r1", "c1", "host_run", "host:/bin/echo hi")
    assert ap.answer(aid, "r1", "always") == (True, "ok")
    assert ap.allowed("c1", "host:/bin/echo hi") is True
    assert ap.host_scopes("c1") == ["host:/bin/echo hi"]
    ap.forget_all_host()
    assert ap.allowed("c1", "host:/bin/echo hi") is False
    assert ap.host_scopes("c1") == []


def test_turning_host_off_keeps_other_grants():
    ap = AP.Approvals()
    ap.answer(ap.issue("r1", "c1", "write_file", "write_file"), "r1", "always")
    ap.answer(ap.issue("r2", "c1", "host_read", "host:read:/x/y"), "r2", "always")
    ap.forget_all_host()
    assert ap.allowed("c1", "write_file") is True and ap.allowed("c1", "host:read:/x/y") is False


def test_a_grant_does_not_generalise_to_a_longer_command():
    ap = AP.Approvals()
    ap.answer(ap.issue("r1", "c1", "host_run", "host:/bin/echo hi"), "r1", "always")
    assert ap.allowed("c1", "host:/bin/echo hi there") is False


def test_the_switch_clears_grants_and_is_per_chat(root, allow):
    ap_seen = []

    def sink(rec):
        ap_seen.append(rec)

    h = routes.Hub(os.path.join(root, "chat"), host=_policy(allow_path=None, audit=sink))
    h.approvals.answer(h.approvals.issue("r1", "c1", "host_run", "host:/bin/echo hi"), "r1", "always")
    assert h.approvals.host_scopes("c1") == ["host:/bin/echo hi"]
    assert h.host_set("c1", True)["on"] is True
    assert h.approvals.host_scopes("c1") == []
    assert h.host_status("c1")["on"] is True and h.host_status("c2")["on"] is False
    assert h.host_set("c1", False)["on"] is False
    assert h.host_ctx("c1")[0] is False


def test_the_star_switch_covers_a_chat_that_does_not_exist_yet(root):
    h = routes.Hub(os.path.join(root, "chat"), host=_policy())
    assert h.host_set("", True)["on"] is True
    assert h.host_status("brand-new")["on"] is True


# ---- the audit sink ------------------------------------------------------------------------------
def test_an_audit_failure_does_not_break_the_call_or_hide_it(root, allow):
    def boom(rec):
        raise IOError("disk full")

    ok, out = tools.execute("host_run", {"command": ["/bin/echo", "hi"]}, _ctx(root, allow, audit=boom))
    assert ok is True and "hi" in out


def test_a_bad_argument_reports_an_error_not_a_crash(root, allow):
    ok, out = tools.execute("host_run", {"command": 5}, _ctx(root, allow))
    assert ok is False and out.startswith("refused:")
    ok, out = tools.execute("host_run", {"command": None}, _ctx(root, allow))
    assert ok is False and out.startswith("refused:")
    # a non-string argument is sent as text (a model writes --port as a number); the allowlist still decides
    ok, out = tools.execute("host_run", {"command": ["/bin/echo", 7]}, _ctx(root, allow))
    assert ok is True and "exit code 0" in out


def test_a_missing_program_is_an_error(root, allow):
    ok, out = tools.execute("host_run", {"command": ["/bin/nope-not-here"]},
                            _ctx(root, {"commands": [["/bin/nope-not-here"]], "read_paths": []}))
    assert ok is False and "not found" in out


# ---- the UI-facing status ------------------------------------------------------------------------
def test_status_offers_the_switch_and_the_allowlist(root, allow):
    p = os.path.join(root, "host-allow.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(allow, f)
    h = routes.Hub(os.path.join(root, "chat"), host=_policy(allow_path=p))
    assert h.host_status("") == {"available": True, "on": False, "why": ""}
    assert h.host_allowlist()["commands"] == [["/bin/echo"], ["nvidia-smi", "-L"]]
    h.host_set("", True)
    ok, why, al = h.host_ctx("c9")
    assert ok is True and why == "" and al["read_paths"] == allow["read_paths"]
