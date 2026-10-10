# Home Assistant

Two ways to get your PXA server's numbers into Home Assistant:

| | what you need | what you get |
|---|---|---|
| **REST** (this half) | nothing but HA | the server's own endpoints: status, slots, model, speed, KV cache, slot forks |
| **MQTT** (below) | PXA Control + an MQTT broker | the same set *plus* per-GPU temperature/VRAM/utilisation and the expert map, auto-discovered |

**Everything here is local network only.** The server and Home Assistant talk to each other
directly on your LAN. No cloud service, no account, no telemetry to anyone: the values never
leave your house.

---

## 1. The endpoints

All four are the server's own HTTP endpoints. They answer on the same host and port as the API.

| endpoint | returns | use it for | poll |
|---|---|---|---|
| `/health` | JSON | is the server up, how many slots are busy | 15 s |
| `/props` | JSON | model name, context size, slot count | 300 s (it never changes while a model is loaded) |
| `/slots` | JSON array | per-slot state and prompt progress | 15 s |
| `/metrics` | Prometheus text | speed, KV-cache usage, request counters, slot-fork savings | 15 s |

```
$ curl -s http://your-host:8080/health
{"status":"ok","slots_idle":2,"slots_processing":0}

$ curl -s http://your-host:8080/props | head -c 120
{"system_prompt":"","model_alias":"qwen3-0.6b-q8","model_path":"/models/qwen3-0.6b-q8.gguf",...

$ curl -s http://your-host:8080/slots | head -c 80
[{"n_ctx":1024,"n_predict":-1,"model":"qwen3-0.6b-q8",...

$ curl -s http://your-host:8080/metrics | grep predicted_tokens_seconds
llamacpp:predicted_tokens_seconds 279.07
```

Two things to know before you start:

- **`/metrics` is off by default.** Start the server with `--metrics` (or the equivalent
  `LLAMA_ARG_ENDPOINT_METRICS=1`) or that endpoint returns `404`. The other three are always on.
- **`/health`, `/v1/health`, `/models`, `/v1/models` and `/api/tags` answer without the
  API key** when you run with `--api-key`. Everything else answers `401` until you send
  the key (see §4). That is deliberate: whatever is watching the service has to be able
  to reach the health and model-list endpoints.

---

## 2. REST sensors

Add this to `configuration.yaml`. **Edit one line** — the anchor marked `# <-- your server` is
used by every sensor below, so the address appears exactly once.

```yaml
rest:
  # ---- status and slots: the cheap endpoint, poll it often ----
  - resource: &pxa_health http://192.168.1.50:8080/health   # <-- your server
    scan_interval: 15
    headers: &pxa_headers
      # delete these two lines if you do NOT run the server with --api-key
      Authorization: !secret pxa_api_header
    sensor:
      - name: PXA server status
        unique_id: pxa_server_status
        value_template: "{{ value_json.status }}"
      - name: PXA slots idle
        unique_id: pxa_slots_idle
        value_template: "{{ value_json.slots_idle }}"
        state_class: measurement
      - name: PXA slots processing
        unique_id: pxa_slots_processing
        value_template: "{{ value_json.slots_processing }}"
        state_class: measurement
    binary_sensor:
      - name: PXA generating
        unique_id: pxa_generating
        device_class: running
        value_template: "{{ 'on' if value_json.slots_processing > 0 else 'off' }}"

  # ---- model and geometry: these do not change while a model is loaded ----
  - resource: &pxa_props http://192.168.1.50:8080/props
    scan_interval: 300
    headers: *pxa_headers
    sensor:
      - name: PXA model
        unique_id: pxa_model
        # model_alias is whatever you passed to --alias; with no alias it is the model's PATH, so
        # this takes the last segment of either field -- a name, never a path
        value_template: "{{ (value_json.model_alias or value_json.model_name).split('/')[-1] }}"
      - name: PXA context size
        unique_id: pxa_context_size
        value_template: "{{ value_json.default_generation_settings.n_ctx }}"
        unit_of_measurement: tokens
        state_class: measurement
      - name: PXA total slots
        unique_id: pxa_total_slots
        value_template: "{{ value_json.total_slots }}"
        state_class: measurement

  # ---- speed and cache: Prometheus text, read with a regex ----
  - resource: &pxa_metrics http://192.168.1.50:8080/metrics
    scan_interval: 15
    headers: *pxa_headers
    sensor:
      - name: PXA generation speed
        unique_id: pxa_generation_speed
        value_template: "{{ value | regex_findall_index('llamacpp:predicted_tokens_seconds ([0-9.]+)') | float(0) | round(1) }}"
        unit_of_measurement: tok/s
        state_class: measurement
      - name: PXA prompt speed
        unique_id: pxa_prompt_speed
        value_template: "{{ value | regex_findall_index('llamacpp:prompt_tokens_seconds ([0-9.]+)') | float(0) | round(1) }}"
        unit_of_measurement: tok/s
        state_class: measurement
      - name: PXA requests processing
        unique_id: pxa_requests_processing
        value_template: "{{ value | regex_findall_index('llamacpp:requests_processing ([0-9.]+)') | int(0) }}"
        state_class: measurement
      - name: PXA requests deferred
        unique_id: pxa_requests_deferred
        value_template: "{{ value | regex_findall_index('llamacpp:requests_deferred ([0-9.]+)') | int(0) }}"
        state_class: measurement
      - name: PXA KV cache used
        unique_id: pxa_kv_cache_used
        value_template: "{{ (value | regex_findall_index('llamacpp:kv_cache_usage_ratio ([0-9.]+)') | float(0) * 100) | round(1) }}"
        unit_of_measurement: "%"
        state_class: measurement
      - name: PXA KV cache tokens
        unique_id: pxa_kv_cache_tokens
        value_template: "{{ value | regex_findall_index('llamacpp:kv_cache_tokens ([0-9]+)') | int(0) }}"
        unit_of_measurement: tokens
        state_class: measurement
      - name: PXA tokens generated
        unique_id: pxa_tokens_generated
        value_template: "{{ value | regex_findall_index('llamacpp:tokens_predicted_total ([0-9]+)') | int(0) }}"
        unit_of_measurement: tokens
        state_class: total_increasing
      - name: PXA tokens prompted
        unique_id: pxa_tokens_prompted
        value_template: "{{ value | regex_findall_index('llamacpp:prompt_tokens_total ([0-9]+)') | int(0) }}"
        unit_of_measurement: tokens
        state_class: total_increasing

  # ---- per-slot state: 0 = sleeping, 1 = generating ----
  - resource: &pxa_slots http://192.168.1.50:8080/slots
    scan_interval: 15
    headers: *pxa_headers
    sensor:
      - name: PXA slot states
        unique_id: pxa_slot_states
        # one digit per slot, in order -- "0,1" is a two-slot server with the second one busy
        value_template: "{{ value_json | map(attribute='state') | join(',') }}"
      - name: PXA slot 0 state
        unique_id: pxa_slot_0_state
        value_template: "{{ value_json[0].state }}"
      # add one block per slot, index = the slot number:
      # - name: PXA slot 1 state
      #   value_template: "{{ value_json[1].state }}"
```

> The `&`/`*` marks are YAML anchors: `&pxa_health` names a value, `*pxa_headers` reuses it. If you
> prefer plain repetition, copy the `headers:` block into each entry and drop the anchors — the
> sensors themselves do not change.

**Which slot is busy is the server's business, not slot 0's.** A request can be served by any
free slot, so `PXA slot 0 state` is only ever about slot 0 — read `PXA slot states` (all of them at
once) or `PXA slots processing` (the count) when you want to know whether the server is working.
`PXA generating` is the one to trigger automations on: it is `on` for exactly as long as at least
one slot is producing tokens.

### Slot-fork savings (v3.1)

The server saves prompt work when a request can start from another slot's live KV prefix. Those
numbers are counters, so they are useful as *history*, not as a live dial:

```yaml
  - resource: http://192.168.1.50:8080/metrics
    scan_interval: 60
    headers: *pxa_headers
    sensor:
      - name: PXA slot forks
        unique_id: pxa_slot_forks
        value_template: "{{ value | regex_findall_index('llamacpp:slot_forks_total ([0-9]+)') | int(0) }}"
        state_class: total_increasing
      - name: PXA fork tokens saved
        unique_id: pxa_fork_tokens_saved
        value_template: "{{ value | regex_findall_index('llamacpp:slot_fork_prompt_tokens_saved_total ([0-9]+)') | int(0) }}"
        unit_of_measurement: tokens
        state_class: total_increasing
```

### One word of state

The four raw numbers above are enough for a dashboard but not for an automation. This turns them
into a single value you can trigger on:

```yaml
template:
  - sensor:
      - name: PXA activity
        unique_id: pxa_activity
        state: >
          {% if not has_value('sensor.pxa_server_status') or is_state('sensor.pxa_server_status', 'unknown') %}offline
          {% elif states('sensor.pxa_slots_processing') | int(0) > 0 %}generating
          {% else %}idle{% endif %}
```

---

## 3. A dashboard card

```yaml
type: entities
title: PXA server
entities:
  - entity: sensor.pxa_activity
    name: Activity
  - entity: binary_sensor.pxa_generating
  - entity: sensor.pxa_model
  - entity: sensor.pxa_generation_speed
  - entity: sensor.pxa_prompt_speed
  - entity: sensor.pxa_slots_processing
  - entity: sensor.pxa_context_size
  - entity: sensor.pxa_kv_cache_used
```

...or, if you prefer gauges:

```yaml
type: grid
columns: 2
square: false
cards:
  - type: gauge
    entity: sensor.pxa_generation_speed
    name: Generation
    min: 0
    max: 100
  - type: gauge
    entity: sensor.pxa_kv_cache_used
    name: KV cache
    min: 0
    max: 100
    severity:
      green: 0
      yellow: 75
      red: 90
  - type: gauge
    entity: sensor.pxa_prompt_speed
    name: Prompt
    min: 0
    max: 1000
```

---

## 4. If your server uses `--api-key`

Only `/health` answers without the key. `/props`, `/slots` and `/metrics` return `401`, so every
sensor above that points at one of those three needs the key in a header — that is what the
`headers:` block does.

Put the key in `secrets.yaml`, never in `configuration.yaml`:

```yaml
# secrets.yaml
pxa_api_header: "Bearer your-key-here"
```

The `Bearer ` prefix is part of the value; the header the server expects is
`Authorization: Bearer <key>`. A missing header and a wrong key both give the same `401`, so if a
sensor reads `unknown`, check the key before you check the template.

---

## 5. Checks

```bash
# is the server answering on the LAN at all?
curl -s http://192.168.1.50:8080/health

# 404 here means you did not start the server with --metrics
curl -s http://192.168.1.50:8080/metrics | head

# 401 here is the API key, not a broken endpoint
curl -s -o /dev/null -w '%{http_code}\n' -H 'Authorization: Bearer your-key' \
     http://192.168.1.50:8080/metrics
```

In Home Assistant, **Developer tools → Template** renders any `value_template` from this page
against the live server, so you can paste one in and see the value before you restart HA.

---

## 6. MQTT, with Home Assistant discovery

The REST half above asks the **server** for its numbers. This half publishes the numbers **PXA
Control** is already collecting — the same ones the Live tab shows — to an MQTT broker, and lets
Home Assistant build the entities by itself. It is the only way to get per-GPU figures and the
expert map into HA, because those are Control's readings, not the server's.

**It is off until you turn it on.** Nothing is published and no broker is contacted otherwise.

### Turn it on

In PXA Control: **Advanced settings → Home Assistant**. Fill in the broker, press **Test
connection** (this only checks the link — it publishes nothing), then **Save**.

| Setting | Default | Notes |
|---|---|---|
| Publish to Home Assistant | off | the master switch |
| Broker host | — | the address of your MQTT broker, e.g. `homeassistant.local` or an IP |
| Broker port | `1883` | `8883` is the usual port when TLS is on |
| User, Password | — | empty if your broker allows anonymous connections |
| Encrypted (TLS) | off | turn on for a broker that requires it (`8883`) |
| Base topic | `pxa` | where the values are published |
| Interval | `15` seconds | how often a reading is sent |
| Home Assistant discovery | on | off = publish the values only, and you write the entities yourself |

The same settings can be given as environment variables when PXA Control starts, and an
environment variable wins over the page: `PXA_CONTROL_MQTT=1`, `PXA_CONTROL_MQTT_HOST`,
`PXA_CONTROL_MQTT_PORT`, `PXA_CONTROL_MQTT_USER`, `PXA_CONTROL_MQTT_PASSWORD`,
`PXA_CONTROL_MQTT_TLS`, `PXA_CONTROL_MQTT_BASE`, `PXA_CONTROL_MQTT_PREFIX`,
`PXA_CONTROL_MQTT_INTERVAL`, `PXA_CONTROL_MQTT_DISCOVERY`. The broker password is stored with your
other PXA Control settings (that file is readable only by you) and is never written to a log or
shown back to the page.

### What appears in Home Assistant

With discovery on, each reading is a real entity, grouped under a device named **`PXA <hostname>`**,
with one sub-device per GPU:

| Device | Entities |
|---|---|
| `PXA <host>` | generation speed, prompt speed, slots busy, slots, KV cache used, expert-cache hit, draft acceptance, model, state |
| `PXA <host> GPU 0`, `GPU 1`, … | temperature, VRAM used, VRAM total, utilisation, power |
| `PXA <host>` (diagnostic) | RAM used, CPU used, swap used |
| `PXA <host>` (expert map) | learned sessions, the map's state (learning / idle) |

They show up without any YAML: Home Assistant reads a retained *discovery* message per entity and
adds it on its own. If your broker requires the `homeassistant` prefix to be something else, change
it with `PXA_CONTROL_MQTT_PREFIX`; the entities are then published under that root instead.

One metric that this build does not report simply produces no entity — a card with no power reading
gets no power sensor, rather than a sensor frozen at zero.

### Availability

The publisher sends `online` to `<base>/status` when it is connected, and registers that same topic
as its **last will**. When PXA Control shuts down it says `offline` itself; if it instead crashes or
loses the broker, the broker publishes that same `offline` for it. Either way your entities go
unavailable rather than holding a stale number, so an automation that waits for a fresh reading can
watch the entity's availability and be right.

### Checking it by hand

Any MQTT client will do; `mosquitto_sub` is the usual one:

```bash
# what Control is publishing (topics, then values)
mosquitto_sub -h your-broker -t 'pxa/#' -v

# the discovery messages Home Assistant acts on
mosquitto_sub -h your-broker -t 'homeassistant/#' -v

# is the publisher online? press Ctrl-C and watch it go "offline" via the last will
mosquitto_sub -h your-broker -t 'pxa/status' -v
```

A value that arrives as `pxa/server_8080/tokens_per_second` is the same generation speed the Live
tab shows; the part after the last `/` is the metric, the part before it names the thing it belongs
to (a server by its port, `gpu_0`, `host`, `expert_map`).

### Settings files, if you want them

If you would rather not use discovery, turn it off and write the sensors yourself — every value is
a plain number on `<base>/<thing>/<metric>`, so an MQTT sensor is one line each:

```yaml
mqtt:
  sensor:
    - name: PXA generation speed
      unique_id: pxa_mqtt_generation_speed
      state_topic: pxa/server_8080/tokens_per_second
      availability_topic: pxa/status
      payload_available: online
      payload_not_available: offline
      unit_of_measurement: tok/s
      state_class: measurement
```

Take the exact topic from `mosquitto_sub -t 'pxa/#' -v` — the middle part is the server's port,
which is whatever your server is on.
