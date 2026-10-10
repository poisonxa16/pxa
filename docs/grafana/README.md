# Prometheus and Grafana

PXA Control exposes its own sampler on `GET /metrics` when `PXA_CONTROL_METRICS=1` (or the same switch in Settings). That text is the card gauges (`pxa_card_temperature_celsius`, power, clocks, memory), the per-server gauges (`pxa_server_decode_tokens_per_second`, `pxa_server_draft_acceptance_ratio`, and the rest), and the counters since Control started (`pxa_server_requests_total`).

Control also appends each server's own `/metrics`, with a `server` label. The engine lines are the usual `llamacpp:` counters plus `llamacpp:draft_tokens_total` and `llamacpp:draft_tokens_accepted_total`, labelled `type` for the drafter that proposed the tokens (mtp, ngram-mod, draft, none, and the other speculative types).

Point Prometheus at Control, not at every engine port:

```yaml
scrape_configs:
  - job_name: pxa-control
    static_configs:
      - targets: ["127.0.0.1:8088"]
```

The port is whatever Control is listening on. Import [pxa-control.json](pxa-control.json) and pick that Prometheus datasource. The dashboard is the gauges above; it does not add a series Control does not already export.
