
## RC3.1 hardening (QA RC3 findings)

- **F78 Reception FAQ**: callers hear only the owner-curated FAQ (`reception.faq` in vigil.json plus
  `GET/POST /v1/vigil/ops/reception/faq`, `POST .../faq/<id>/delete`, UI token). General memory is never read.
- **F80**: a VIP caller ID counts only with STIR/SHAKEN attestation `A` (Twilio `StirVerstat`, SIP `attestation`).
- **F79**: earned autonomy never answers approvals in a tainted run; approvals in tainted runs do not count toward graduation.
- **F81**: VIP match is exact on the normalized address; DMARC pass or aligned DKIM/SPF (topmost
  Authentication-Results) is required for severity above warning and for the emergency override.
- **F82**: `alerts.caps` (`per_rule_hour` default 10, `global_hour` default 30); excess alerts fold into one hourly digest alert.
- **F83**: guardrail argv matching strips VAR=, env (incl. -S), nice, ionice, timeout, nohup, stdbuf, sudo, xargs, setsid ... and recurses into sh/bash -c scripts (grouping included).
- **F84/F85**: sealed held turns bind chat, approval and turn in the AAD; approval cards and grant scopes are
  sealed in the db; persisted events and audit lines carry only `[sealed]` forms (the live card is served from
  memory); `secure_delete=ON`, and legacy plaintext rows are re-sealed then checkpointed + VACUUMed. A blob that
  fails authentication is dropped at recovery (never run).
- **F86**: Landlock-only sandbox: Landlock TCP rules (ABI >= 4) plus a seccomp filter denying AF_INET/AF_INET6/AF_PACKET sockets and io_uring; if either cannot be applied the command is refused.
- **F87**: bwrap uses `--unshare-pid`; Landlock ABI >= 6 scopes signals and abstract unix sockets. Limit: on a
  kernel with Landlock ABI < 6 and no usable bwrap, a sandboxed command can still signal same-user processes.
- **Tasks**: a failing critique with no listed issues replans (bounded) instead of marking the task done.
