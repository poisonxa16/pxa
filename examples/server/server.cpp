#pragma warning(disable : 4996)
#include "server-context.h"
#include "server-common.h"
#include "server-chat.h"
#include "server-cors-proxy.h"
#include "chat.h"

#include "common.h"
#include "speculative.h"
#include "mtmd.h"
#include "sampling.h"
#include "llama.h"
#include "llama-vocab.h"
#include <fstream>
       

// mime type for sending response
#define MIMETYPE_JSON "application/json; charset=utf-8"


#ifndef NDEBUG
// crash the server in debug mode, otherwise send an http 500 error
#define CPPHTTPLIB_NO_EXCEPTIONS 1
#endif

#include <nlohmann/json.hpp>
#include "index.html.gz.hpp"
#include "index_llamacpp.html.gz.hpp"
#include "loading.html.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <set>
#include <mutex>
#include <thread>
#include <signal.h>
#include <memory>
#include <random>
#include <algorithm>
#include <climits>
#include <src/llama-impl.h>
#ifdef SQLITE3_MODERN_CPP_SUPPORT
#include <sqlite_modern_cpp.h>

struct DatabaseHandle {
    sqlite::database db;

    DatabaseHandle(const std::string& path) : db(path) {
        db << "CREATE TABLE IF NOT EXISTS sessions (key TEXT PRIMARY KEY, data TEXT)";
        db << "CREATE TABLE IF NOT EXISTS templates (key TEXT PRIMARY KEY, data TEXT)";
        db << "CREATE TABLE IF NOT EXISTS names (key TEXT PRIMARY KEY, data TEXT)";
    }
};
#endif

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;
constexpr int HTTP_POLLING_SECONDS = 1;

bool server_verbose = false;
bool server_log_json = true;


enum server_state {
    SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,          // Server is ready and model is loaded
    SERVER_STATE_ERROR           // An error occurred, load_model failed
};


static inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
    case STOP_TYPE_EOS:   return "eos";
    case STOP_TYPE_WORD:  return "word";
    case STOP_TYPE_LIMIT: return "limit";
    default:              return "none";
    }
}


inline std::string get_model_name(std::string path)
{
    std::string filename = path.substr(path.find_last_of("/\\") + 1);
    return filename;
};


static json format_final_response_oaicompat(const json& request, json result, const std::string& completion_id, bool streaming = false) {
    bool stopped_word = result.count("stopped_word") != 0;
    bool stopped_eos = json_value(result, "stopped_eos", false);
    int num_tokens_predicted = json_value(result, "tokens_predicted", 0);
    int num_prompt_tokens = json_value(result, "tokens_evaluated", 0);
    std::string content = json_value(result, "content", std::string(""));

    std::string finish_reason = "length";
    if (stopped_word || stopped_eos) {
        finish_reason = "stop";
    }

    json choices =
        streaming ? json::array({ json{{"finish_reason", finish_reason},
                                        {"index", 0},
                                        {"delta", json::object()}} })
        : json::array({ json{{"finish_reason", finish_reason},
                              {"index", 0},
                              {"message", json{{"content", content},
                                               {"role", "assistant"}}}} });

    std::time_t t = std::time(0);

    json res = json{
        {"choices", choices},
        {"created", t},
        {"model",
            json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
        {"object", streaming ? "chat.completion.chunk" : "chat.completion"},
        {"usage", json {
            {"completion_tokens", num_tokens_predicted},
            {"prompt_tokens",     num_prompt_tokens},
            {"total_tokens",      num_tokens_predicted + num_prompt_tokens}
        }},
        {"id", completion_id}
    };

    if (server_verbose) {
        res["__verbose"] = result;
    }

    if (result.contains("completion_probabilities")) {
        res["completion_probabilities"] = json_value(result, "completion_probabilities", json::array());
    }

    return res;
}

// return value is vector as there is one case where we might need to generate two responses
static std::vector<json> format_partial_response_oaicompat(server_task_result task_result, const std::string& completion_id) {
    json result = task_result.data;
    std::cout << result.dump(4) << std::endl;
    if (!result.contains("model") || !result.contains("oaicompat_token_ctr")) {
        return std::vector<json>({ result });
    }

    bool first = json_value(result, "oaicompat_token_ctr", 0) == 0;
    std::string modelname = json_value(result, "model", std::string(DEFAULT_OAICOMPAT_MODEL));

    bool stopped_word = json_value(result, "stopped_word", false);
    bool stopped_eos = json_value(result, "stopped_eos", false);
    bool stopped_limit = json_value(result, "stopped_limit", false);
    std::string content = json_value(result, "content", std::string(""));

    std::string finish_reason;
    if (stopped_word || stopped_eos) {
        finish_reason = "stop";
    }
    if (stopped_limit) {
        finish_reason = "length";
    }

    std::time_t t = std::time(0);

    json choices;

    if (!finish_reason.empty()) {
        choices = json::array({ json{{"finish_reason", finish_reason},
                                    {"index", 0},
                                    {"delta", json::object()}} });
    }
    else {
        if (first) {
            if (content.empty()) {
                choices = json::array({ json{{"finish_reason", nullptr},
                                            {"index", 0},
                                            {"delta", json{{"role", "assistant"}}}} });
            }
            else {
                // We have to send this as two updates to conform to openai behavior
                json initial_ret = json{ {"choices", json::array({json{
                                        {"finish_reason", nullptr},
                                        {"index", 0},
                                        {"delta", json{
                                            {"role", "assistant"}
                                        }}}})},
                            {"created", t},
                            {"id", completion_id},
                            {"model", modelname},
                            {"object", "chat.completion.chunk"} };

                json second_ret = json{
                            {"choices", json::array({json{{"finish_reason", nullptr},
                                                            {"index", 0},
                                                            {"delta", json{
                                                            {"content", content}}}
                                                            }})},
                            {"created", t},
                            {"id", completion_id},
                            {"model", modelname},
                            {"object", "chat.completion.chunk"} };

                return std::vector<json>({ initial_ret, second_ret });
            }
        }
        else {
            // Some idiosyncrasy in task processing logic makes several trailing calls
            // with empty content, we ignore these at the calee site.
            if (content.empty()) {
                return std::vector<json>({ json::object() });
            }

            choices = json::array({ json{
                {"finish_reason", nullptr},
                {"index", 0},
                {"delta",
                json{
                    {"content", content},
                }},
            } });
        }
    }

    json ret = json{
        {"choices", choices},
        {"created", t},
        {"id",      completion_id},
        {"model",   modelname},
        {"object",  "chat.completion.chunk"}
    };

    if (task_result.timings.prompt_n != -1) {
        ret.push_back({ "timings", task_result.timings.to_json() });
    }

    //
    if (!finish_reason.empty()) {
        int num_tokens_predicted = json_value(result, "tokens_predicted", 0);
        int num_prompt_tokens = json_value(result, "tokens_evaluated", 0);
        ret.push_back({ "usage", json {
            {"completion_tokens", num_tokens_predicted},
            {"prompt_tokens",     num_prompt_tokens},
            {"total_tokens",      num_tokens_predicted + num_prompt_tokens}
        } });
    }

    return std::vector<json>({ ret });
}


static json format_embeddings_response_oaicompat(const json& request, const json& embeddings, bool use_base64 = false) {
    json data = json::array();
    int32_t n_tokens = 0;
    int i = 0;
    for (const auto& elem : embeddings) {
        json embedding_obj;

        if (use_base64) {
            const auto& vec = json_value(elem, "embedding", json::array()).get<std::vector<float>>();
            const char* data_ptr = reinterpret_cast<const char*>(vec.data());
            size_t data_size = vec.size() * sizeof(float);
            embedding_obj = {
                {"embedding", base64::encode(data_ptr, data_size)},
                {"index", i++},
                {"object", "embedding"},
                {"encoding_format", "base64"}
            };
        }
        else {
            embedding_obj = {
                {"embedding", json_value(elem, "embedding", json::array())},
                {"index", i++},
                {"object", "embedding"}
            };
        }
        data.push_back(embedding_obj);
        n_tokens += json_value(elem, "tokens_evaluated", 0);
    }
    json res = json{
        {"model", json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
        {"object", "list"},
        {"usage", json {
            {"prompt_tokens", n_tokens},
            {"total_tokens", n_tokens}
        }},
        {"data", data}
    };

    return res;
}

static void log_server_request(const httplib::Request & req, const httplib::Response & res) {
    // skip GH copilot requests when using default port
    if (req.path == "/v1/health" || req.path == "/v1/completions") {
        return;
    }

    LOG_INFO("request", {
        {"remote_addr", req.remote_addr},
        {"remote_port", req.remote_port},
        {"status",      res.status},
        {"method",      req.method},
        {"path",        req.path},
        {"params",      req.params},
    });

    LOG_VERBOSE("request", {
        {"request",  req.body},
        {"response", res.body},
    });
}

// generator-like API for server responses, support pooling connection state and aggregating results
struct server_response_reader {
    std::unordered_set<int> id_tasks;
    server_context& ctx_server;
    size_t received_count = 0;
    bool cancelled = false;

    server_response_reader(server_context& ctx_server) : ctx_server(ctx_server) {}
    ~server_response_reader() {
        stop();
    }

    void post_tasks(std::vector<server_task>&& tasks) {       
        id_tasks = server_task::get_list_id(tasks);
        ctx_server.queue_results.add_waiting_tasks(tasks);
        ctx_server.queue_tasks.post(std::move(tasks));
    }
    
    bool has_next() {
        return !cancelled && received_count < id_tasks.size();
    }

    // cancel-cascade fix (PR #1941 / b1eb8bb): true only if one of THIS reader's
    // tasks is on a slot (the active decode). Used to gate llama_decode_stop() so a
    // queued/deferred task's disconnect cannot abort another task's active decode via
    // the process-global stop_internal_decode flag. Best-effort cross-thread read
    // (slots are not resized at runtime; same race class as the global).
    bool any_task_on_slot() const {
        for (const auto & slot : ctx_server.slots) {
            if (slot.is_processing() && id_tasks.count(slot.id_task)) return true;
        }
        return false;
    }

    // PXA_SCOPED_CANCEL_v1 (sole-occupant gating): abort the in-flight GPU decode ONLY when every
    // processing slot belongs to THIS (disconnected) reader. llama_decode_stop() is process-global
    // and the ret=-3 unwind touches every slot that was in the batch — so if another client's gen
    // shares the batch, aborting nukes it too (observed: the innocent request is silently released
    // and its client hangs to its own timeout). When other slots are active we skip the hard abort
    // entirely: stop() has already posted SERVER_TASK_TYPE_CANCEL for our tasks, which releases our
    // slot cleanly at the next token boundary (costs at most one decode step for a dead client, and
    // the innocent gens never notice). Same best-effort cross-thread read class as any_task_on_slot.
    void cancel_active_decode() const {
        bool ours = false, others = false;
        for (const auto & slot : ctx_server.slots) {
            if (!slot.is_processing()) continue;
            if (id_tasks.count(slot.id_task)) ours = true; else others = true;
        }
        if (ours && !others) llama_decode_stop();
    }

    // return nullptr if should_stop() is true before receiving a result
    // note: if one error is received, it will stop further processing and return error result
    server_task_result_ptr next(const std::function<bool()>& should_stop) {
        while (true) {
            server_task_result_ptr result = ctx_server.queue_results.recv_with_timeout(id_tasks, HTTP_POLLING_SECONDS);
            if (result == nullptr) {
                // timeout, check stop condition
                if (should_stop()) {
                    SRV_DBG("%s", "stopping wait for next result due to should_stop condition\n");
                    return nullptr;
                }
            }
            else {
                if (result->is_error()) {
                    stop(); // cancel remaining tasks
                    SRV_DBG("%s", "received error result, stopping further processing\n");
                    return result;
                }
                if (result->is_stop()) {
                    received_count++;
                }
                return result;
            }
        }

        // should not reach here
    }

    struct batch_response {
        bool is_terminated = false; // if true, indicates that processing was stopped before all results were received
        std::vector<server_task_result_ptr> results;
        server_task_result_ptr error; // nullptr if no error
    };

    batch_response wait_for_all(const std::function<bool()>& should_stop) {
        batch_response batch_res;
        batch_res.results.resize(id_tasks.size());
        while (has_next()) {
            auto res = next(should_stop);
            if (res == nullptr) {
                batch_res.is_terminated = true;
                return batch_res;
            }
            if (res->error) {
                batch_res.error = std::move(res);
                return batch_res;
            }
            const size_t idx = res->get_index();
            GGML_ASSERT(idx < batch_res.results.size() && "index out of range");
            GGML_ASSERT(batch_res.results[idx] == nullptr && "duplicate result received");
            batch_res.results[idx] = std::move(res);
        }
        return batch_res;
    }

    void stop() {
        ctx_server.queue_results.remove_waiting_task_ids(id_tasks);
        if (has_next() && !cancelled) {
            // if tasks is not finished yet, cancel them
            cancelled = true;
            std::vector<server_task> cancel_tasks;
            cancel_tasks.reserve(id_tasks.size());
            for (const auto& id_task : id_tasks) {
                SRV_WRN("cancel task, id_task = %d\n", id_task);
                server_task task(SERVER_TASK_TYPE_CANCEL);
                task.id_target = id_task;
                ctx_server.queue_results.remove_waiting_task_id(id_task);
                cancel_tasks.push_back(std::move(task));
            }
            // push to beginning of the queue, so it has highest priority
            ctx_server.queue_tasks.post(std::move(cancel_tasks), true);
        }
        else {
            SRV_DBG("%s", "all tasks already finished, no need to cancel\n");
        }
    }
};

auto res_err = [](httplib::Response& res, json error_data) {
    json final_response{ {"error", error_data} };
    res.set_content(safe_json_to_str(final_response), MIMETYPE_JSON);
    res.status = json_value(error_data, "code", 500);
};

auto res_ok = [](httplib::Response& res, const json& data) {
    res.set_content(safe_json_to_str(data), "application/json; charset=utf-8");
    res.status = 200;
};

std::function<void(int)> shutdown_handler;
std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

static void log_prompt(const gpt_params & params_base, const json & body) {
    if (params_base.minilog) {
        LOG_TEE("Prompt:\n%s\n", body.dump(4).c_str());
    }
}

// ---------------------------------------------------------------------------
// PXA POSTURE LAYER (2026-07-22): PXA_MODE=balance|max + ADAPTIVE-UB.
// The two owner-facing postures (the PRODUCT; the kernel levers are the means):
//   BALANCE (default, the daily): -fa on, ub 2048-class. Best decode AND best-possible
//     prefill IN the fa-on regime — carried by PXA_FA_PREFILL_SPLIT + PXA_FA_MASK_SKIP_TILE
//     (see ggml/src/ggml-cuda/pxa/pxa-enhance.cuh); fa-on decode is byte-untouched by construction.
//   MAX (bulk ingest): -fa off, largest-fitting ub. Absolute max prefill, decode secondary.
// This layer only FILLS FLAGS THE CLI LEFT UNSET — explicit -fa/-ub (or their
// LLAMA_ARG_* env forms) always win. PXA_REFERENCE=1 stands the posture layer down
// entirely (pure reference path, stock defaults). Kernel-lever selection lives in the
// pxa_mode() resolver in pxa-enhance.cuh — keep the PXA_MODE parsing here in lockstep.
//
// ADAPTIVE-UB: probe free/total VRAM per assigned CUDA device and pick the largest ub in
// {2048,1024,768,512} that plausibly fits next to the model's per-device share (heuristic
// ~0.5 MiB compute buffer per ub token — deliberately optimistic so full-VRAM cards keep
// their measured optimum; the card-type default is the primary selector). Safe fallback =
// card-type default: >=15 GiB card -> 2048, >=10 GiB (11 GB 1080Ti class) -> 768, else 512.
#include "ggml.h"          // ggml_pxa_config_level -- the PXA master config level
#if defined(GGML_USE_CUDA)
#   include "ggml-cuda.h"
#endif
#include <cstring>
#include <initializer_list>

// read general.architecture from the gguf header (metadata only, no tensor data) — the
// posture layer runs BEFORE model load but needs the arch for MLA-class exceptions.
static std::string pxa_gguf_arch(const std::string & fname) {
    struct gguf_init_params ip = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
    struct gguf_context * g = gguf_init_from_file(fname.c_str(), ip);
    if (!g) return "";
    std::string arch;
    const int i = gguf_find_key(g, "general.architecture");
    if (i >= 0) arch = gguf_get_val_str(g, i);
    gguf_free(g);
    return arch;
}

// 0 = balance, 1 = max, -1 = reference (posture layer stands down)
static int pxa_posture_mode() {
    if (ggml_pxa_config_level() == 0) return -1;
    const char * m = getenv("PXA_MODE");
    return (m && (m[0] == 'm' || m[0] == 'M')) ? 1 : 0;
}

static bool pxa_argv_has(int argc, char ** argv, std::initializer_list<const char *> names) {
    for (int i = 1; i < argc; ++i) {
        for (const char * n : names) {
            if (strcmp(argv[i], n) == 0) return true;
        }
    }
    return false;
}

// ── PXA_CONTAINER_AWARE_v1 (2026-07-30) ─────────────────────────────────────────────────
// Detect whether this process runs under a container supervisor (Docker/Podman/containerd/
// k8s/LXC). The wedge monitor's exit-and-let-the-orchestrator-restart contract is only
// valid when an orchestrator exists; bare metal gets an in-process recovery attempt and a
// distinct exit code instead (see the PXA_WEDGE_EXIT_v1 block in main). Multi-signal, and
// the verdict is logged once at startup so it is auditable.
// Precedence: PXA_IN_CONTAINER=0|1 (operator override) > filesystem/cgroup/env evidence.
#ifndef _WIN32
#include <sys/stat.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <unistd.h>
#include <sstream>

static bool pxa_file_exists(const char * p) {
    struct stat st;
    return ::stat(p, &st) == 0;
}

static bool pxa_file_contains(const char * path, std::initializer_list<const char *> needles, std::string & hit) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        for (const char * n : needles) {
            if (line.find(n) != std::string::npos) { hit = n; return true; }
        }
    }
    return false;
}

// Container-internal mountinfo signature: /etc/hostname, /etc/hosts or /etc/resolv.conf
// bind-mounted from a container-manager path. Checking the MOUNT POINT (field 5) is what
// keeps a docker HOST honest — a host's mountinfo also lists /var/lib/docker/containers/...
// mounts of its running containers, so a bare substring test would misclassify every
// docker-running bare-metal box as "in a container".
static bool pxa_mountinfo_container_root(std::string & hit) {
    std::ifstream f("/proc/self/mountinfo");
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream is(line);
        std::string f1, f2, f3, f4, mp;
        if (!(is >> f1 >> f2 >> f3 >> f4 >> mp)) continue;
        if (mp != "/etc/hostname" && mp != "/etc/hosts" && mp != "/etc/resolv.conf") continue;
        for (const char * n : {"/docker/containers/", "/containers/storage/", "/kubelet/pods/", "/lxc/"}) {
            if (line.find(n) != std::string::npos) {
                hit = std::string(n) + " -> " + mp;
                return true;
            }
        }
    }
    return false;
}

static bool pxa_pid1_env_has_container(std::string & kv_out) {
    std::ifstream f("/proc/1/environ", std::ios::binary);
    if (!f) return false; // unreadable (not root in a shared pid ns) — other signals decide
    std::string blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t pos = 0;
    while (pos < blob.size()) {
        size_t end = blob.find('\0', pos);
        if (end == std::string::npos) end = blob.size();
        if (blob.compare(pos, 10, "container=") == 0) {
            kv_out = blob.substr(pos, end - pos);
            return true;
        }
        pos = end + 1;
    }
    return false;
}
#endif // !_WIN32

static bool pxa_detect_container(std::string & why) {
    const char * ov = getenv("PXA_IN_CONTAINER");
    if (ov && *ov) {
        why = std::string("PXA_IN_CONTAINER=") + ov + " (operator override)";
        return atoi(ov) != 0;
    }
#ifdef _WIN32
    why = "windows: no container detection implemented, assuming bare metal";
    return false;
#else
    std::string hit;
    if (pxa_file_exists("/.dockerenv"))        { why = "/.dockerenv present (docker)";           return true; }
    if (pxa_file_exists("/run/.containerenv")) { why = "/run/.containerenv present (podman)";    return true; }
    std::string cenv;
    if (pxa_pid1_env_has_container(cenv))      { why = cenv + " in /proc/1/environ (lxc/nspawn/podman)"; return true; }
    if (pxa_file_contains("/proc/1/cgroup", {"docker", "containerd", "kubepods", "libpod", "lxc", "buildkit"}, hit)) {
        why = std::string("'") + hit + "' in /proc/1/cgroup"; return true;
    }
    if (pxa_mountinfo_container_root(hit)) {
        why = std::string("'") + hit + "' in /proc/self/mountinfo"; return true;
    }
    why = "no container markers (checked /.dockerenv, /run/.containerenv, /proc/1/environ, /proc/1/cgroup, /proc/self/mountinfo)";
    return false;
#endif
}

// ── PXA_PORT_GUARD_v1 (2026-07-30) ──────────────────────────────────────────────────────
// Refuse to start when another LIVE process already answers on the target port. bind()
// alone is not enough of a story: when a fork-orphan (see PXA_BT_NOFORK_v1 in ggml.c) or a
// forgotten sibling holds the socket, the stock failure is a one-line bind error that says
// nothing about WHO holds the port — and with SO_REUSEPORT-style setups two servers can
// silently share one. This probe makes the refusal loud and actionable. PXA_PORT_GUARD=0
// disables it.
#ifndef _WIN32
static bool pxa_port_has_live_listener(const std::string & host, int port) {
    const std::string probe_host = (host.empty() || host == "0.0.0.0") ? "127.0.0.1" : host;
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    char portstr[16];
    snprintf(portstr, sizeof(portstr), "%d", port);
    struct addrinfo * res = nullptr;
    if (getaddrinfo(probe_host.c_str(), portstr, &hints, &res) != 0 || !res) {
        return false; // cannot probe — let bind_to_port decide
    }
    bool live = false;
    for (struct addrinfo * ai = res; ai; ai = ai->ai_next) {
        int fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        struct timeval tv = {1, 0};
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        if (::connect(fd, ai->ai_addr, (socklen_t) ai->ai_addrlen) == 0) {
            live = true;
            ::close(fd);
            break;
        }
        ::close(fd);
    }
    freeaddrinfo(res);
    return live;
}
#endif // !_WIN32

// true only if EVERY visible CUDA device is cc 610 (sm_61: P40/GTX 10-series class). That
// arch is fp16-starved (1:64 rate vs sm_60/P100's full-rate fp16, sm_70+'s tensor cores) and
// the MLA flash-attention path leans on fp16 — community P40 measurement: fa-on decode is
// 75-326% SLOWER there than fa-off, and the gap widens with context. A mixed fleet (any
// sm_60/70+ device present) keeps the fa-on posture below; only an all-sm_61 fleet flips it.
static bool pxa_all_devices_sm61() {
#if defined(GGML_USE_CUDA)
    const int ndev = ggml_backend_cuda_get_device_count();
    if (ndev <= 0) return false;
    for (int i = 0; i < ndev; ++i) {
        if (ggml_backend_cuda_get_device_cc(i) != 610) return false;
    }
    return true;
#else
    return false;
#endif
}

// returns the chosen ubatch (>0) or 0 = leave params.n_ubatch untouched; fills *why
// Smallest per-device VRAM headroom, in bytes, that will be left after the weights land: the
// minimum free VRAM across the visible CUDA devices minus this model's uniform-split share.
// Same estimate pxa_adaptive_ubatch uses (uniform split, tensor-split not modelled) and the same
// caveat: it runs BEFORE the model loads, so it is an estimate, deliberately used only to DECLINE
// things -- never to claim something fits. Returns -1 when it cannot tell (no CUDA, unreadable
// model file), which every caller must treat as "do not arm".
// Per-device share of the model file honouring an explicit -ts (2026-09-06, AUTO-DEFAULTS-AUDIT gap b,
// order #1400a). Before this the estimators divided the file evenly across devices; on the Flash-Next
// seat (-ts 5079,12612,12612,11897, card 0 shared with another server) that reported a negative
// headroom for a model that had been serving for days, floored -ub to 512 (377.67 t/s @3121 vs the
// seat's 478.27) and declined speculation for a reason that was arithmetic, not memory. With -ts the
// share is model_bytes * ts[i] / sum(ts). Returns false when no -ts was given (uniform split).
static bool pxa_device_shares(const gpt_params & params, int ndev, size_t model_bytes, std::vector<size_t> & share) {
    share.assign(ndev, model_bytes / (size_t) std::max(1, ndev));
    double sum = 0.0;
    for (int i = 0; i < ndev; ++i) sum += params.tensor_split[i] > 0 ? params.tensor_split[i] : 0.0;
    if (sum <= 0.0) return false;
    for (int i = 0; i < ndev; ++i) {
        const double f = params.tensor_split[i] > 0 ? params.tensor_split[i] / sum : 0.0;
        share[i] = (size_t) ((double) model_bytes * f);
    }
    return true;
}
// -ot overrides that pin tensors to the CPU (e.g. Flash-Next's 54 GB per-layer embedding table) make the
// file size an overestimate of what lands on the devices. The estimators cannot know by how much, so
// they only say so; a negative headroom under an -ot=CPU override is an estimator artefact, not a fact.
static bool pxa_has_cpu_override(const gpt_params & params) {
    for (const auto & o : params.tensor_buft_overrides) {
        if (o.buft && ggml_backend_buft_name(o.buft) && strstr(ggml_backend_buft_name(o.buft), "CPU")) return true;
    }
    return false;
}

static long long pxa_min_vram_headroom(const gpt_params & params, std::string & why) {
#if defined(GGML_USE_CUDA)
    char buf[320];
    const int ndev = ggml_backend_cuda_get_device_count();
    if (ndev <= 0) { why = "no CUDA devices"; return -1; }
    size_t model_bytes = 0;
    {
        std::ifstream f(params.model, std::ios::binary | std::ios::ate);
        if (f.good()) model_bytes = (size_t) f.tellg();
    }
    if (model_bytes == 0) { why = "model size unknown"; return -1; }
    std::vector<size_t> share;
    const bool ts = pxa_device_shares(params, ndev, model_bytes, share);
    long long head = LLONG_MAX; int worst = 0; size_t worst_free = 0;
    for (int i = 0; i < ndev; ++i) {
        size_t free = 0, total = 0;
        ggml_backend_cuda_get_device_memory(i, &free, &total);
        const long long h = (long long) free - (long long) share[i];
        if (h < head) { head = h; worst = i; worst_free = free; }
    }
    snprintf(buf, sizeof(buf), "dev%d free %zu MiB, model share %zu MiB (%s) -> headroom %lld MiB%s",
             worst, worst_free>>20, share[worst]>>20, ts ? "per -ts" : "uniform split", head>>20,
             pxa_has_cpu_override(params) ? "; -ot pins tensors to CPU, so the file overstates device bytes" : "");
    why = buf;
    return head;
#else
    (void) params;
    why = "no CUDA build";
    return -1;
#endif
}

// True when every visible CUDA device is sm_61 and there is exactly one of them -- the 11 GB
// 1080 Ti seat, where the weights already fill the card and any extra context is fatal.
static bool pxa_single_sm61(void) {
#if defined(GGML_USE_CUDA)
    return ggml_backend_cuda_get_device_count() == 1 && ggml_backend_cuda_get_device_cc(0) == 610;
#else
    return false;
#endif
}

static int pxa_adaptive_ubatch(const gpt_params & params, std::string & why) {
#if defined(GGML_USE_CUDA)
    const int ndev = ggml_backend_cuda_get_device_count();
    if (ndev <= 0) { why = "no CUDA devices - ub untouched"; return 0; }
    size_t min_free = SIZE_MAX, min_total = SIZE_MAX;
    for (int i = 0; i < ndev; ++i) {
        size_t free = 0, total = 0;
        ggml_backend_cuda_get_device_memory(i, &free, &total);
        if (free  < min_free)  min_free  = free;
        if (total < min_total) min_total = total;
    }
    const size_t GiB = 1024ull*1024ull*1024ull;
    // card-type default (the safe fallback + the cap): 16 GB class -> 2048, 11 GB class -> 768
    const int card_default = min_total >= 15*GiB ? 2048 : min_total >= 10*GiB ? 768 : 512;
    // model per-device share (uniform-split estimate; tensor-split refinement not modeled)
    size_t model_bytes = 0;
    {
        std::ifstream f(params.model, std::ios::binary | std::ios::ate);
        if (f.good()) model_bytes = (size_t) f.tellg();
    }
    char buf[256];
    if (model_bytes == 0) {
        snprintf(buf, sizeof(buf), "card-type default (model size unknown; min total %zu MiB)", min_total>>20);
        why = buf;
        return card_default;
    }
    // per-device share honouring -ts (2026-09-06); the binding device is the one with the least headroom
    std::vector<size_t> shares;
    const bool ts = pxa_device_shares(params, ndev, model_bytes, shares);
    long long headroom_ll = LLONG_MAX; size_t share = 0; int worst = 0;
    for (int i = 0; i < ndev; ++i) {
        size_t free = 0, total = 0;
        ggml_backend_cuda_get_device_memory(i, &free, &total);
        const long long h = (long long) free - (long long) shares[i];
        if (h < headroom_ll) { headroom_ll = h; share = shares[i]; worst = i; }
    }
    if (headroom_ll < 0) {
        // A negative headroom says the model would not load at all -- which is not what happens on
        // the seats this runs on (an -ot=CPU override, or a shared card, made the file-size share
        // wrong). An estimator that is provably wrong must not pick the slowest rung: fall back to
        // the card-type default and say why (2026-09-06; the old floor cost the 4x P100 seat 21 %
        // of its prefill, AUTO-DEFAULTS-AUDIT gap b).
        snprintf(buf, sizeof(buf), "card-type default (estimate unreliable: dev%d share %zu MiB (%s) exceeds its free VRAM%s)",
                 worst, share>>20, ts ? "per -ts" : "uniform split",
                 pxa_has_cpu_override(params) ? "; -ot pins tensors to CPU" : "");
        why = buf;
        return card_default;
    }
    const size_t headroom = (size_t) headroom_ll;
    static const int ladder[] = { 2048, 1024, 768, 512 };
    for (int ub : ladder) {
        if (ub > card_default) continue;
        const size_t need = (size_t) ub * 512ull * 1024ull; // ~0.5 MiB per ub token (optimistic)
        if (need <= headroom) {
            snprintf(buf, sizeof(buf), "adaptive: dev%d free-min %zu MiB, min total %zu MiB, model share %zu MiB (%s) -> headroom %zu MiB",
                     worst, min_free>>20, min_total>>20, share>>20, ts ? "per -ts" : "uniform split", headroom>>20);
            why = buf;
            return ub;
        }
    }
    snprintf(buf, sizeof(buf), "adaptive floor (dev%d headroom %zu MiB too tight, %s)", worst, headroom>>20, ts ? "per -ts" : "uniform split");
    why = buf;
    return 512;
#else
    (void) params;
    why = "no CUDA build - ub untouched";
    return 0;
#endif
}

int main(int argc, char ** argv) {
#if SERVER_VERBOSE != 1
    log_disable();
#endif
    // own arguments required by this example
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }

    // parse arguments from environment variables
    gpt_params_parse_from_env(params);

    // TODO: not great to use extern vars
    server_log_json = params.log_json;
    server_verbose = params.verbosity > 0;

    // PXA_CONTAINER_AWARE_v1: detect the runtime once, loudly — the wedge monitor's restart
    // contract depends on it (see the PXA_WEDGE_EXIT_v1 block below). Always printed, even
    // with logging disabled, so post-mortems can tell which contract was in force.
    std::string pxa_container_why;
    const bool pxa_in_container = pxa_detect_container(pxa_container_why);
    fprintf(stderr, "PXA_CONTAINER_AWARE_v1: runtime=%s (%s)\n",
            pxa_in_container ? "container" : "bare-metal", pxa_container_why.c_str());
    fflush(stderr);


    // struct that contains llama context and inference
    server_context ctx_server;

    if (!params.system_prompt.empty()) {
        ctx_server.system_prompt_set(params.system_prompt);
    }

    if (params.model_alias == "unknown") {
        params.model_alias = params.model;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // PXA POSTURE (2026-07-22): PXA_MODE=balance|max fills -fa/-ub the CLI left unset;
    // explicit flags always win; PXA_REFERENCE=1 stands this down. See the block above main.
    {
        const int  pmode       = pxa_posture_mode();
        const bool fa_explicit = pxa_argv_has(argc, argv, {"-fa", "--flash-attn", "-no-fa", "--no-flash-attn"})
                                 || getenv("LLAMA_ARG_FLASH_ATTN") != nullptr;
        const bool ub_explicit = pxa_argv_has(argc, argv, {"-ub", "--ubatch-size"})
                                 || getenv("LLAMA_ARG_UBATCH") != nullptr;
        // deepseek2 (GLM-4.7-Flash / DeepSeek class, MLA attention): fa-off degrades
        // CATASTROPHICALLY with context (community-measured on a P40: 37 -> 3.3 t/s by
        // 36k ctx; -fa on fixed it). The posture layer must never leave an MLA model on
        // fa-off implicitly — explicit user flags still win (and draw the warning below).
        const bool mla_explicit = pxa_argv_has(argc, argv, {"-mla", "--mla-use"});
        const std::string gguf_arch = pxa_gguf_arch(params.model);
        const bool is_mla_arch = gguf_arch == "deepseek2";
        // cc-aware fa default: an all-sm_61 fleet is fp16-starved for the MLA FA path (see
        // pxa_all_devices_sm61 above) — it gets fa default OFF instead of the ON default that
        // holds for every other fleet (sm_60/P100, sm_70+/tensor-core, or mixed).
        const bool mla_fa_off_arch = is_mla_arch && pxa_all_devices_sm61();
        if (pmode < 0) {
            fprintf(stderr, "PXA posture: reference (PXA_REFERENCE=1) - fa/ub untouched (fa=%s ub=%d)\n",
                    params.flash_attn ? "on" : "off", params.n_ubatch);
        } else {
            if (!fa_explicit) {
                params.flash_attn = (pmode == 0);   // BALANCE -> fa on (serving); MAX -> fa off (ingest)
                if (pmode == 1 && is_mla_arch && !mla_fa_off_arch) {
                    // ARCH EXCEPTION: max mode is fa-off ingest, but MLA requires fa — except on
                    // an all-sm_61 fleet, where fa-off is the measured-correct posture (below).
                    params.flash_attn = true;
                    fprintf(stderr, "PXA posture: mode=max but arch=deepseek2 — fa kept ON (MLA requires it)\n");
                }
                if (is_mla_arch) {
                    if (mla_fa_off_arch) {
                        params.flash_attn = false;
                        fprintf(stderr, "PXA posture: arch=deepseek2 on sm_61 — fa default OFF (fp16-starved FA "
                                        "path; measured 75-326%% slower decode with fa on), mla=%d%s\n",
                                params.mla_attn, mla_explicit ? " (explicit)" : "");
                    } else if (params.flash_attn) {
                        fprintf(stderr, "PXA posture: arch=deepseek2 (MLA) — fa auto-wired ON, mla=%d%s\n",
                                params.mla_attn, mla_explicit ? " (explicit)" : "");
                    }
                }
            }
            if (is_mla_arch && !mla_explicit && params.mla_attn != 3) {
                params.mla_attn = 3;   // deepseek2 default: -mla 3 (user-set wins), unchanged by device cc
            }
            // PXA PER-CARD BATCH DEFAULTS (2026-09-03): on a topology the campaign actually
            // measured, -b/-ub the user left unset are filled from the measured cell instead of
            // from the generic VRAM ladder below. The ladder is a heuristic ("what plausibly
            // fits"); this is a measurement ("what was fastest on these exact cards"), so it
            // takes precedence -- but only where a measurement exists, and only for a flag that
            // was NOT given. See ggml_backend_cuda_pxa_suggest_batch() for the table.
            const bool b_explicit = pxa_argv_has(argc, argv, {"-b", "--batch-size"})
                                    || getenv("LLAMA_ARG_BATCH") != nullptr;
            bool card_b = false, card_ub = false;
#if defined(GGML_USE_CUDA)
            if (!ub_explicit || !b_explicit) {
                int         sb = 0, sub = 0;
                const char * cell = nullptr;
                if (ggml_backend_cuda_pxa_suggest_batch(&sb, &sub, &cell)) {
                    if (!b_explicit)  { params.n_batch  = sb;  card_b  = true; }
                    if (!ub_explicit) { params.n_ubatch = sub; card_ub = true; }
                    fprintf(stderr,
                            "PXA_AUTO: batch defaults for %s -> -b %d%s -ub %d%s "
                            "(measured cell; pass -b/-ub to override, PXA_ENHANCE=0 for the "
                            "stock defaults)\n",
                            cell, params.n_batch, card_b ? "" : " (explicit, kept)",
                            params.n_ubatch, card_ub ? "" : " (explicit, kept)");
                    // 1x sm_61 also gets --ctx-checkpoints 0. The checkpoint pool is per slot and
                    // sized off n_ctx; on the 11 GB card the weights leave no room for it, and the
                    // measured 1080 Ti protocol has always run with it off. Only fills the flag
                    // when the user did not pass it.
                    if (pxa_single_sm61()
                        && !pxa_argv_has(argc, argv, {"--ctx-checkpoints"})
                        && params.ctx_checkpoints_n != 0) {
                        params.ctx_checkpoints_n = 0;
                        fprintf(stderr, "PXA_AUTO: --ctx-checkpoints 0 on the 1x 1080 Ti cell "
                                        "(11 GB card, weights leave no room for the checkpoint "
                                        "pool; pass --ctx-checkpoints N to override)\n");
                    }
                }
            }
#else
            (void) b_explicit;
#endif
            std::string ub_why = card_ub ? "PXA per-card measured default" : "explicit CLI/env - ub untouched";
            if (!ub_explicit && !card_ub) {
                const int ub = pxa_adaptive_ubatch(params, ub_why);
                if (ub > 0) {
                    params.n_ubatch = ub;
                    if (params.n_batch < ub && !b_explicit) {
                        params.n_batch = ub;   // keep -b >= -ub so the ub choice is not clamped away
                    }
                }
            }
            fprintf(stderr, "PXA posture: mode=%s [%s] fa=%s%s ub=%d (%s)\n",
                    pmode == 1 ? "max" : "balance",
                    pmode == 1 ? "fa-off ingest, absolute max prefill" : "fa-on serving, decode-first",
                    params.flash_attn ? "on" : "off", fa_explicit ? " (explicit)" : "",
                    params.n_ubatch, ub_why.c_str());
        }
        // the fa-off warning is suppressed on an all-sm_61 fleet — fa-off is the RECOMMENDED
        // config there (see mla_fa_off_arch); it still fires for sm_60/70+ where fa-off is a
        // real regression, whether the user set it explicitly or it's a mixed-fleet default.
        if (is_mla_arch && !params.flash_attn && !mla_fa_off_arch) {
            fprintf(stderr, "WARNING: deepseek2/MLA with -fa off degrades severely with context; use -fa on -mla 3\n");
        }
    }

    // PXA AUTO-TS (2026-07-29): fill -ts on the ONE measured mixed cell. T4 (ENHANCE
    // topology tune, 2026-07-23) measured a V100-heavy -ts 1.4,0.6 as the DOMINANT mixed-rig
    // lever: +9.78% decode (76.27 -> 83.73 t/s, PXQ4-35B split V100+P100, -sm layer, 588-tok
    // prompt), where every kernel lever was neutral; 1.45 loses decode and 1.5+ OOMs the V100.
    // Scope is deliberately NARROW: exactly 2 CUDA devices, one sm_70 + one sm_60, -ts unset,
    // ENHANCE level — every other topology keeps the stock proportional-by-VRAM default (the
    // measured basis does not generalize; a wrong -ts can OOM). CLI -ts always wins;
    // PXA_AUTO_TS=0 forces off, =1 forces on at any level (never under PXA_REFERENCE=1).
    {
#if defined(GGML_USE_CUDA)
        const char * at = getenv("PXA_AUTO_TS");
        const bool  ref2 = ggml_pxa_config_level() == 0;
        const bool  enh2 = ggml_pxa_config_level() >= 2;
        const bool  active = at ? (atoi(at) != 0 && !ref2) : enh2;
        const bool  ts_explicit = pxa_argv_has(argc, argv, {"-ts", "--tensor-split"});
        bool ts_unset = true;
        for (size_t i = 0; i < llama_max_devices(); ++i) {
            if (params.tensor_split[i] != 0.0f) { ts_unset = false; break; }
        }
        if (active && !ts_explicit && ts_unset && ggml_backend_cuda_get_device_count() == 2) {
            const int cc0 = ggml_backend_cuda_get_device_cc(0);
            const int cc1 = ggml_backend_cuda_get_device_cc(1);
            const bool mixed_70_60 = (cc0 == 700 && cc1 == 600) || (cc0 == 600 && cc1 == 700);
            if (mixed_70_60) {
                const int hi = (cc0 == 700) ? 0 : 1;
                params.tensor_split[hi]     = 1.4f;
                params.tensor_split[1 - hi] = 0.6f;
                fprintf(stderr, "PXA_AUTO: tensor_split dev%d(sm_70)=1.4 dev%d(sm_60)=0.6 "
                                "(measured T4 cell: +9.78%% decode vs balanced, PXQ4-35B V100+P100 "
                                "-sm layer; override -ts or PXA_AUTO_TS=0)\n", hi, 1 - hi);
            }
        }
#endif
    }

    // PXA AUTO-SAMPLERS (2026-07-29): model-family sampler DEFAULTS, ENHANCE-level only.
    // Fills only fields the CLI left unset (per-field --temp/--top-k/--top-p/--min-p
    // explicitness check); per-request sampler params still override per request as always.
    // Gate: PXA_AUTO_SAMPLERS=0 forces off, =1 forces on at any level; unset -> engages only
    // under PXA_ENHANCE=1 (and never under PXA_REFERENCE=1), so DEFAULT-level behavior is
    // byte-identical to before this layer existed. Every applied default logs value+reason.
    // The one PERFORMANCE-critical rule (measured): top_k<=0 means a full-vocab CPU sort
    // EVERY token — on a ~201k vocab that HALVED decode (31 vs 65 t/s, gpt-oss-120b,
    // 6-card cell, 2026-07-04). The guard clamps an unset-or-0 top_k to 100.
    {
        const char * as = getenv("PXA_AUTO_SAMPLERS");
        const bool  ref = ggml_pxa_config_level() == 0;
        const bool  enh = ggml_pxa_config_level() >= 2;
        const bool  active = as ? (atoi(as) != 0 && !ref) : enh;
        if (active) {
            const std::string arch = pxa_gguf_arch(params.model);
            const bool t_exp = pxa_argv_has(argc, argv, {"--temp"});
            const bool k_exp = pxa_argv_has(argc, argv, {"--top-k"});
            const bool p_exp = pxa_argv_has(argc, argv, {"--top-p"});
            const bool m_exp = pxa_argv_has(argc, argv, {"--min-p"});
            struct pxa_sampler_row { const char * arch; float temp; int top_k; float top_p; float min_p; const char * why; };
            static const pxa_sampler_row tab[] = {
                // gpt-oss "official" is top_k=0 — measured to HALVE decode (full-vocab sort);
                // top_k=100 is the measured-negligible-quality fix (2026-07-04).
                { "gpt-oss",   1.0f, 100, 1.00f, 0.00f, "gpt-oss family defaults; top_k=100 not 0 (top_k=0 halves decode: full-201k-vocab CPU sort/token)" },
                { "qwen35moe", 0.7f,  20, 0.80f, 0.00f, "qwen no-think agent defaults (official Qwen no-think sampler set)" },
                { "qwen3next", 0.7f,  20, 0.80f, 0.00f, "qwen no-think agent defaults (official Qwen no-think sampler set)" },
                { "qwen35",    0.7f,  20, 0.80f, 0.00f, "qwen no-think agent defaults (official Qwen no-think sampler set)" },
                { "laguna",    0.7f,  20, 0.80f, 0.00f, "qwen-family hybrid — qwen no-think defaults (HEURISTIC: no per-model sampler sweep yet)" },
                // qwen4exp = Flash-Next (Qwen3.8 hybrid MoE), the flagship seat's arch; it had no row and
                // booted on stock llama.cpp samplers ("no family row", Phase P alex-ref boot.txt:5).
                // Same official Qwen no-think set as the other qwen arches (AUTO-DEFAULTS-AUDIT gap e,
                // order #1400b); HEURISTIC until a per-model sweep exists.
                { "qwen4exp",  0.7f,  20, 0.80f, 0.00f, "qwen-family hybrid MoE (Flash-Next) — qwen no-think defaults (HEURISTIC: no per-model sampler sweep yet)" },
            };
            const pxa_sampler_row * row = nullptr;
            for (const auto & r : tab) {
                if (arch == r.arch) { row = &r; break; }
            }
            if (row) {
                if (!t_exp) params.sparams.temp  = row->temp;
                if (!k_exp) params.sparams.top_k = row->top_k;
                if (!p_exp) params.sparams.top_p = row->top_p;
                if (!m_exp) params.sparams.min_p = row->min_p;
                fprintf(stderr, "PXA_AUTO: samplers arch=%s -> temp=%.2f%s top_k=%d%s top_p=%.2f%s min_p=%.2f%s (%s; "
                                "override: the CLI flag or PXA_AUTO_SAMPLERS=0)\n",
                        arch.c_str(),
                        params.sparams.temp,  t_exp ? "(cli)" : "",
                        params.sparams.top_k, k_exp ? "(cli)" : "",
                        params.sparams.top_p, p_exp ? "(cli)" : "",
                        params.sparams.min_p, m_exp ? "(cli)" : "",
                        row->why);
            } else {
                fprintf(stderr, "PXA_AUTO: samplers arch=%s -> no family row, stock defaults kept "
                                "(temp=%.2f top_k=%d top_p=%.2f min_p=%.2f)\n",
                        arch.empty() ? "?" : arch.c_str(),
                        params.sparams.temp, params.sparams.top_k, params.sparams.top_p, params.sparams.min_p);
            }
            // The generic performance guard, independent of the family table.
            if (!k_exp && params.sparams.top_k <= 0) {
                params.sparams.top_k = 100;
                fprintf(stderr, "PXA_AUTO: top_k<=0 clamped to 100 (top_k=0 forces a full-vocab CPU sort "
                                "every token — measured decode HALVED on a 201k vocab; override --top-k)\n");
            }
        }
    }

    // PXA AUTO-SPEC (2026-08-03): arm the measured-best SELF-speculation drafter per model
    // family, so ENHANCE users get it without knowing the flag exists.
    //
    // MEASURED BASIS -- 4x P100, Qwen3.5-122B-A10B (community "heretic" abliteration) PXQU48, -c 32768 -b 512 -ub 512 -fa on,
    // ~14-15k fill, control taken in the same session. FIRST-REQUEST (cold n-gram cache):
    //   ngram-mod:n_max=4,n_min=2   code  prompt  24.44 -> 30.05 t/s  (+23.0%)
    //                               prose prompt  23.69 -> 24.78 t/s  (+4.6%)
    // Prefill is untouched in both (262.63 -> 259.16, ~1%): this is a decode-side lever only.
    //
    // ⚠ DO NOT QUOTE THE MEDIAN-OF-3 FIGURES (+38.8% / +35.7%). The bench replays the SAME
    // prompt three times against one server, so reps 2-3 regenerate a continuation already in
    // the n-gram cache -- prose acceptance hit 1.000 (100/100), which is the tell. Only rep1 is
    // a first-request number. Warm figures are not fiction (multi-turn chat resends history,
    // agentic loops reread the same files) but they are an upper bound, not the default case.
    //
    // ⚠ WHY ONLY qwen35moe, AND WHY NOT mtp:
    // On the SAME model, MTP is a TAX -- 27.10 -> 24.78 (-8.6%) at n_max=1 and -29.8% at
    // n_max=2, despite 0.800 acceptance. A drafter that costs a transformer forward per cycle
    // cannot pay for itself on bandwidth-bound sparse-MoE decode, where a k-token verify batch
    // routes to up to k*8 distinct experts and so reads ~k* the expert bytes. ngram-mod's
    // drafter is a table lookup and costs nothing, which is why the same verify economics leave
    // it profitable and MTP not. Do NOT generalise "speculation helps" from this row.
    //
    // ⚠ THE GAIN IS WORKLOAD-DEPENDENT. Acceptance tracks how repetitive the generated text is:
    // code repeats n-grams hard (identifiers, indentation already in context) and scores ~0.96;
    // prose is far lower. The same effect on the dense 27B with MTP measured +23.4% on code but
    // only +3.4% on prose (acceptance 0.841 vs 0.556). This arms the drafter, not a promise.
    //
    // Gate: PXA_AUTO_SPEC=0 forces off, =1 forces on at any level; unset -> engages only under
    // PXA_ENHANCE=1, and never under PXA_REFERENCE=1, so DEFAULT-level behaviour is unchanged.
    // An explicit --spec-type or a draft model (-md) always wins -- user intent is never
    // overridden, only absence is filled.
    {
        const char * sp  = getenv("PXA_AUTO_SPEC");
        const bool   ref3 = ggml_pxa_config_level() == 0;
        const bool   enh3 = ggml_pxa_config_level() >= 2;
        const bool   active = sp ? (atoi(sp) != 0 && !ref3) : enh3;

        const bool spec_explicit  = pxa_argv_has(argc, argv, {"--spec-type", "--spec-stage"})
                                    || !params.speculative.stages.empty();
        const bool draft_explicit = pxa_argv_has(argc, argv, {"-md", "--model-draft"});

        // VRAM GATE (2026-09-03). Arming a speculation stage costs device memory: the draft
        // context, its KV, and the per-slot checkpoint buffers the verify path keeps. On a card
        // whose weights already fill it -- the 11 GB 1080 Ti seat is the measured case, PXQ2
        // weights are 9.9 GiB of 11.0 -- that turns a working server into one that OOMs mid
        // prefill. It never did so before because AUTO_SPEC only armed for people who exported
        // PXA_ENHANCE=1; with ENHANCE as the default it would become a default OOM for every
        // 11-12 GB card. So the auto-arm now declines when the estimated post-weights headroom
        // is under 2 GiB, and declines outright on a single-card sm_61 fleet. This is an
        // AUTO-ARM gate only: an explicit --spec-type, -md, or PXA_AUTO_SPEC=1 still gets what
        // it asked for, because a user who names the flag has accepted the cost.
        bool spec_vram_ok = true;
        std::string spec_vram_why;
        if (active && sp == nullptr) {
            const long long head = pxa_min_vram_headroom(params, spec_vram_why);
            const long long need = 2LL*1024*1024*1024;   // 2 GiB
            if (pxa_single_sm61()) {
                spec_vram_ok  = false;
                spec_vram_why = "single-card sm_61 (11 GB class): the weights fill the card and a "
                                "draft context OOMs mid-prefill -- measured on the 1080 Ti PXQ2 tier";
            } else if (head < 0) {
                spec_vram_ok  = false;
                spec_vram_why = "cannot estimate free VRAM (" + spec_vram_why + ")";
            } else if (head < need) {
                spec_vram_ok  = false;
                spec_vram_why = "only " + spec_vram_why + ", under the 2048 MiB a draft context needs";
            }
            if (!spec_vram_ok) {
                fprintf(stderr, "PXA_AUTO: spec DECLINED -- %s (override with --spec-type, or "
                                "PXA_AUTO_SPEC=1 to force)\n", spec_vram_why.c_str());
            }
        }

        if (active && spec_vram_ok && !spec_explicit && !draft_explicit) {
            const std::string arch = pxa_gguf_arch(params.model);
            struct pxa_spec_row { const char * arch; const char * stage; const char * why; };
            static const pxa_spec_row spec_tab[] = {
                { "qwen35moe", "ngram-mod:n_max=4,n_min=2",
                  "measured +23.0% first-request decode on code traffic (24.44->30.05 t/s, 4xP100 122B "
                  "PXQU48-core), +4.6% on prose, prefill-neutral; drafts only on an n-gram match so it "
                  "costs ~nothing when it cannot predict, unlike mtp which drafts unconditionally and "
                  "measured -8.6% on this model" },
            };
            const pxa_spec_row * srow = nullptr;
            for (const auto & r : spec_tab) {
                if (arch == r.arch) { srow = &r; break; }
            }
            if (srow) {
                params.speculative.stages.push_back(common_speculative_stage_from_arg(srow->stage));
                const auto resolved = params.speculative.get_resolved_stages();
                params.speculative.type = resolved.empty()
                                        ? COMMON_SPECULATIVE_TYPE_NONE : resolved.front().type;
                params.has_mtp = params.speculative.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP);
                fprintf(stderr, "PXA_AUTO: spec arch=%s -> --spec-type %s (%s; gain scales with how "
                                "repetitive the OUTPUT is -- high on code/tool traffic, much lower on "
                                "prose; override --spec-type or PXA_AUTO_SPEC=0)\n",
                        arch.c_str(), srow->stage, srow->why);
            } else if (!arch.empty()) {
                fprintf(stderr, "PXA_AUTO: spec arch=%s -> no measured row, speculation left off "
                                "(only families with a measured win are armed; --spec-type to force)\n",
                        arch.c_str());
            }
        }
    }

    // PXA_SPEC_RELAXED truth line (2026-09-06, plan X4 / Board B9): when a speculative stage is
    // armed AND relaxed acceptance is live, say so with the floor. It changes the OUTPUT at temp>0
    // (a draft token inside the target's post-filter candidate set is kept instead of the target's
    // own sample); at temperature 0 the exact-match path runs regardless, so greedy gates and the
    // "with speculation" rows measured at temp 0 are exact-verify either way.
    if (!params.speculative.get_resolved_stages().empty()) {
        if (common_sampler_spec_relaxed_active()) {
            fprintf(stderr, "PXA_SPEC_RELAXED: ON (pmin=%.3f) -- relaxed draft acceptance is live for temp>0 "
                            "sampling (level default at ENHANCE); exact verification at temp 0, or set "
                            "PXA_SPEC_RELAXED=0\n", common_sampler_spec_relaxed_pmin());
        } else {
            fprintf(stderr, "PXA_SPEC_RELAXED: OFF -- exact-match draft acceptance at every temperature\n");
        }
    }

    LOG_INFO("build info", {
        {"build",  LLAMA_BUILD_NUMBER},
        {"commit", LLAMA_COMMIT}
    });

    LOG_INFO("system info", {
        {"n_threads",       params.n_threads},
        {"n_threads_batch", params.n_threads_batch},
        {"total_threads",   std::thread::hardware_concurrency()},
        {"system_info",     llama_print_system_info()},
    });

    std::unique_ptr<httplib::Server> svr;
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if (params.ssl_file_key != "" && params.ssl_file_cert != "") {
        LOG_INFO("Running with SSL", {{"key", params.ssl_file_key}, {"cert", params.ssl_file_cert}});
        svr.reset(
            new httplib::SSLServer(params.ssl_file_cert.c_str(), params.ssl_file_key.c_str())
        );
    } else {
        LOG_INFO("Running without SSL", {});
        svr.reset(new httplib::Server());
    }
#else
    svr.reset(new httplib::Server());
#endif

    std::atomic<server_state> state{SERVER_STATE_LOADING_MODEL};

    svr->set_default_headers({{"Server", "PXA"}});

    svr->set_logger(log_server_request);



    svr->set_exception_handler([](const httplib::Request &, httplib::Response & res, std::exception_ptr ep) {
        std::string message;
        try {
            std::rethrow_exception(std::move(ep));
        } catch (std::exception & e) {
            message = e.what();
        } catch (...) {
            message = "Unknown Exception";
        }

        json formatted_error = format_error_response(message, ERROR_TYPE_SERVER);
        LOG_VERBOSE("Got exception", formatted_error);
        res_err(res, formatted_error);
    });

    svr->set_error_handler([](const httplib::Request &, httplib::Response & res) {
        if (res.status == 404) {
            res_err(res, format_error_response("File Not Found", ERROR_TYPE_NOT_FOUND));
        }
        // for other error codes, we skip processing here because it's already done by res_err()
    });

    // set timeouts and change hostname and port
    svr->set_read_timeout (params.timeout_read);
    svr->set_write_timeout(params.timeout_write);

#ifndef _WIN32
    // PXA_PORT_GUARD_v1: never let a second server silently coexist with a live sibling on
    // the same port (the failure mode behind the 2026-07-30 bare-metal duplicate-server
    // incident). Probe BEFORE bind so the refusal is fast and names the actual cause.
    {
        const char * pg = getenv("PXA_PORT_GUARD");
        if (!(pg && atoi(pg) == 0) && pxa_port_has_live_listener(params.hostname, params.port)) {
            fprintf(stderr,
                "\nPXA_PORT_GUARD_v1: a LIVE listener already answers on %s:%d — refusing to start a second server on the same port.\n"
                "Find and stop the sibling first: ss -ltnp 'sport = :%d'\n"
                "(If it is a deadlocked fork-orphan, kill the CHILD before the parent — killing the parent first orphans the child and it keeps the port.)\n"
                "Set PXA_PORT_GUARD=0 to bypass this check.\n\n",
                params.hostname.c_str(), params.port, params.port);
            return 1;
        }
    }
#endif

    if (!svr->bind_to_port(params.hostname, params.port)) {
        fprintf(stderr, "\ncouldn't bind to server socket: hostname=%s port=%d\n\n", params.hostname.c_str(), params.port);
        return 1;
    }

    std::unordered_map<std::string, std::string> log_data;

    log_data["hostname"] = params.hostname;
    log_data["port"]     = std::to_string(params.port);

    if (params.api_keys.size() == 1) {
        auto key = params.api_keys[0];
        log_data["api_key"] = "api_key: ****" + key.substr(std::max((int)(key.length() - 4), 0));
    } else if (params.api_keys.size() > 1) {
        log_data["api_key"] = "api_key: " + std::to_string(params.api_keys.size()) + " keys loaded";
    }

    // Necessary similarity of prompt for slot selection
    ctx_server.slot_prompt_similarity = params.slot_prompt_similarity;
    ctx_server.cache_ram_n_min = params.cache_ram_n_min;
    ctx_server.cache_ram_similarity = params.cache_ram_similarity;
#ifdef SQLITE3_MODERN_CPP_SUPPORT
    auto db_handle = std::make_shared<DatabaseHandle>(params.sql_save_file);
    bool sqlite_extension_loaded = false;
    if (!params.sqlite_zstd_ext_file.empty()) {
        auto* conn = db_handle->db.connection().get();
        sqlite3_enable_load_extension(conn, 1);
        char* errmsg = nullptr;
        const int rc = sqlite3_load_extension(
            conn,
            params.sqlite_zstd_ext_file.c_str(),
            nullptr,
            &errmsg
        );
        if(rc != SQLITE_OK) {
            const std::string err = errmsg ? errmsg : "Unknown extension error";
            sqlite3_free(errmsg);
            LOG_WARNING("Failed to load extension", {{"err", err}});
        }
	else {
            sqlite_extension_loaded = true;
        }
        sqlite3_enable_load_extension(conn, 0);
    }
#else
    auto db_handle = false;
#endif
    // load the model
    if (!ctx_server.load_model(params)) {
        state.store(SERVER_STATE_ERROR);
        return 1;
    } else {
        try {
            ctx_server.init();
        } catch (const std::exception & e) {
            LOG_ERROR("server init failed", {{"error", e.what()}});
            state.store(SERVER_STATE_ERROR);
            return 1;
        }
        state.store(SERVER_STATE_READY);
    }

    LOG_INFO("model loaded", {});

    // PXA: the startup log says what the loaded FILE is (codec / bpw / pxa.pxq* provenance),
    // so a serving box can be identified from its own log instead of from the filename.
    {
        char pxq_desc[256];
        if (llama_model_pxq_file_desc(ctx_server.model, pxq_desc, sizeof(pxq_desc)) > 0) {
            LOG_INFO(pxq_desc, {});
        }
    }

    const auto model_meta = ctx_server.model_meta();

    // print sample chat example to make it clear which template is used

    //    LOG_INFO("chat template", {
    //    {"chat_template", common_chat_templates_source(ctx_server.chat_templates.get())},
    //});

    //LOG_INFO("chat template", {
    //    {"chat_example", common_chat_format_example(ctx_server.chat_templates.get(), ctx_server.params_base.use_jinja, {}).c_str()
    //    },
    //        {"built_in",     params.chat_template.empty()},
    //    });
    //
    // Middlewares
    //

    auto middleware_validate_api_key = [&params](const httplib::Request & req, httplib::Response & res) {
        static const std::unordered_set<std::string> public_endpoints = {
            "/health",
            "/v1/health",
            "/models",
            "/v1/models",
            "/api/tags"
        };

        // If API key is not set, skip validation
        if (params.api_keys.empty()) {
            return true;
        }

        // If path is public or is static file, skip validation
        if (public_endpoints.find(req.path) != public_endpoints.end() || req.path == "/") {
            return true;
        }

        // Check for API key in the header
        auto auth_header = req.get_header_value("Authorization");

        std::string prefix = "Bearer ";
        if (auth_header.substr(0, prefix.size()) == prefix) {
            std::string received_api_key = auth_header.substr(prefix.size());
            if (std::find(params.api_keys.begin(), params.api_keys.end(), received_api_key) != params.api_keys.end()) {
                return true; // API key is valid
            }
        }

        auth_header = req.get_header_value("X-Api-Key");

        if (std::find(params.api_keys.begin(), params.api_keys.end(), auth_header) != params.api_keys.end()) {
            return true; // API key is valid
        }

        // API key is invalid or not provided
        res.status = 401;
        res.set_content(
            (json {
                {"error", {
                    {"message", "Invalid API Key"},
                    {"type", "authentication_error"},
                    {"code", 401}
                }}
            }).dump(-1, ' ', false, json::error_handler_t::replace),
            "application/json; charset=utf-8"
        );
        LOG_WARNING("Unauthorized: Invalid API Key\n", {});
        return false;
    };

    auto middleware_server_state = [&state](const httplib::Request& req, httplib::Response& res) {
        server_state current_state = state.load();
        if (current_state == SERVER_STATE_LOADING_MODEL) {
            auto tmp = string_split<std::string>(req.path, '.');
            if (req.path == "/" || tmp.back() == "html") {
                res.set_content(reinterpret_cast<const char*>(loading_html), loading_html_len, "text/html; charset=utf-8");
                res.status = 503;
            }
            else if (req.path == "/models" || req.path == "/v1/models" || req.path == "/api/tags") {
                // allow the models endpoint to be accessed during loading
                return true;
            }
            else {
                res_err(res, format_error_response("Loading model", ERROR_TYPE_UNAVAILABLE));
            }
            return false;
        }
        return true;
    };

    // register server middlewares
    svr->set_pre_routing_handler([&middleware_validate_api_key, &middleware_server_state](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        // If this is OPTIONS request, skip validation because browsers don't include Authorization header
        if (req.method == "OPTIONS") {
            res.set_header("Access-Control-Allow-Credentials", "true");
            res.set_header("Access-Control-Allow-Methods", "GET, POST");
            res.set_header("Access-Control-Allow-Headers", "*");
            res.set_content("", "text/html"); // blank response, no data
            return httplib::Server::HandlerResponse::Handled; // skip further processing
        }
        if (!middleware_server_state(req, res)) {
            return httplib::Server::HandlerResponse::Handled;
        }
        if (!middleware_validate_api_key(req, res)) {
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
        });

    //
    // Route handlers (or controllers)
    //

    const auto handle_health = [&](const httplib::Request & req, httplib::Response & res) {
        server_state current_state = state.load();
        switch (current_state) {
            case SERVER_STATE_READY:
                {
                    // PXA_HONEST_HEALTH_v1: if a llama_decode has been in-flight past the stall deadline the
                    // main loop is WEDGED — return 503 immediately, WITHOUT posting a metrics task and
                    // blocking on queue_results.recv (that block permanently parks an HTTP worker and is the
                    // true thread-saturation feeder). Deadline via PXA_HEALTH_STALL_MS (default 60000; 0=off).
                    static const int64_t pxa_stall_ms = []{ const char* e = getenv("PXA_HEALTH_STALL_MS"); return e ? atoll(e) : 60000LL; }();
                    if (ctx_server.pxa_decode_wedged(pxa_stall_ms)) {
                        json health = {{"status", "stalled"}, {"stall_ms", pxa_stall_ms}};
                        res.status = 503;
                        res.set_content(health.dump(), "application/json");
                        break;
                    }

                    // request slots data using task queue
                    server_task task;
                    task.id   = ctx_server.queue_tasks.get_new_id();
                    task.type = SERVER_TASK_TYPE_METRICS;
                    task.id_target = -1;

                    ctx_server.queue_results.add_waiting_task_id(task.id);
                    ctx_server.queue_tasks.post(std::move(task));

                    // get the result (the atomic wedge-check above already returns 503 before reaching this
                    // blocking recv on a real llama_decode wedge, so this never parks a worker during a wedge).
                    server_task_result result = ctx_server.queue_results.recv(task.id);
                    ctx_server.queue_results.remove_waiting_task_id(task.id);

                    const int n_idle_slots       = result.data.at("idle");
                    const int n_processing_slots = result.data.at("processing");

                    json health = {
                        {"status",           "ok"},
                        {"slots_idle",       n_idle_slots},
                        {"slots_processing", n_processing_slots}
                    };

                    res.status = 200; // HTTP OK
                    if (params.endpoint_slots && req.has_param("include_slots")) {
                        health["slots"] = result.data.at("slots");
                    }

                    if (n_idle_slots == 0) {
                        health["status"] = "no slot available";
                        if (req.has_param("fail_on_no_slot")) {
                            res.status = 503; // HTTP Service Unavailable
                        }
                    }

                    res.set_content(health.dump(), "application/json");
                    break;
                }
            case SERVER_STATE_LOADING_MODEL:
                {
                    res_err(res, format_error_response("Loading model", ERROR_TYPE_UNAVAILABLE));
                } break;
            case SERVER_STATE_ERROR:
                {
                    res_err(res, format_error_response("Model failed to load", ERROR_TYPE_SERVER));
                } break;
        }
    };

    const auto handle_slots = [&](const httplib::Request &, httplib::Response & res) {
        if (!params.endpoint_slots) {
            res_err(res, format_error_response("This server does not support slots endpoint.", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        // request slots data using task queue
        server_task task;
        task.id = ctx_server.queue_tasks.get_new_id();
        task.id_multi  = -1;
        task.id_target = -1;
        task.type = SERVER_TASK_TYPE_METRICS;

        ctx_server.queue_results.add_waiting_task_id(task.id);
        ctx_server.queue_tasks.post(std::move(task));

        // get the result
        server_task_result result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        res.set_content(result.data.at("slots").dump(), "application/json");
        res.status = 200; // HTTP OK
    };

    const auto handle_metrics = [&](const httplib::Request &, httplib::Response & res) {
        if (!params.endpoint_metrics) {
            res_err(res, format_error_response("This server does not support metrics endpoint.", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        // request slots data using task queue
        server_task task;
        task.id = ctx_server.queue_tasks.get_new_id();
        task.id_multi  = -1;
        task.id_target = -1;
        task.type = SERVER_TASK_TYPE_METRICS;
        task.data.push_back({{"reset_bucket", true}});

        ctx_server.queue_results.add_waiting_task_id(task.id);
        ctx_server.queue_tasks.post(std::move(task));

        // get the result
        server_task_result result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        json data = result.data;

        const uint64_t n_prompt_tokens_processed = data.at("n_prompt_tokens_processed");
        const uint64_t t_prompt_processing       = data.at("t_prompt_processing");

        const uint64_t n_tokens_predicted  = data.at("n_tokens_predicted");
        const uint64_t t_tokens_generation = data.at("t_tokens_generation");

        const int32_t kv_cache_used_cells = data.at("kv_cache_used_cells");

        // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
        json all_metrics_def = json {
            {"counter", {{
                    {"name",  "prompt_tokens_total"},
                    {"help",  "Number of prompt tokens processed."},
                    {"value",  (uint64_t) data.at("n_prompt_tokens_processed_total")}
            }, {
                    {"name",  "prompt_seconds_total"},
                    {"help",  "Prompt process time"},
                    {"value",  (uint64_t) data.at("t_prompt_processing_total") / 1.e3}
            }, {
                    {"name",  "tokens_predicted_total"},
                    {"help",  "Number of generation tokens processed."},
                    {"value",  (uint64_t) data.at("n_tokens_predicted_total")}
            }, {
                    {"name",  "tokens_predicted_seconds_total"},
                    {"help",  "Predict process time"},
                    {"value",  (uint64_t) data.at("t_tokens_generation_total") / 1.e3}
            }}},
            {"gauge", {{
                    {"name",  "prompt_tokens_seconds"},
                    {"help",  "Average prompt throughput in tokens/s."},
                    {"value",  n_prompt_tokens_processed ? 1.e3 / t_prompt_processing * n_prompt_tokens_processed : 0.}
            },{
                    {"name",  "predicted_tokens_seconds"},
                    {"help",  "Average generation throughput in tokens/s."},
                    {"value",  n_tokens_predicted ? 1.e3 / t_tokens_generation * n_tokens_predicted : 0.}
            },{
                    {"name",  "kv_cache_usage_ratio"},
                    {"help",  "KV-cache usage. 1 means 100 percent usage."},
                    {"value",  1. * kv_cache_used_cells / params.n_ctx}
            },{
                    {"name",  "kv_cache_tokens"},
                    {"help",  "KV-cache tokens."},
                    {"value",  (uint64_t) data.at("kv_cache_tokens_count")}
            },{
                    {"name",  "requests_processing"},
                    {"help",  "Number of request processing."},
                    {"value",  (uint64_t) data.at("processing")}
            },{
                    {"name",  "requests_deferred"},
                    {"help",  "Number of request deferred."},
                    {"value",  (uint64_t) data.at("deferred")}
            }}}
        };

        std::stringstream prometheus;

        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();

            for (const auto & metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");

                auto value = json_value(metric_def, "value", 0.);
                prometheus << "# HELP llamacpp:" << name << " " << help  << "\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\n"
                            << "llamacpp:"        << name << " " << value << "\n";
            }
        }

        const int64_t t_start = data.at("t_start");
        res.set_header("Process-Start-Time-Unix", std::to_string(t_start));

        res.set_content(prometheus.str(), "text/plain; version=0.0.4");
        res.status = 200; // HTTP OK
    };

    const auto handle_slots_save = [&ctx_server, &params](const httplib::Request & req, httplib::Response & res, int id_slot) {
        json request_data = json::parse(req.body);
        std::string filename = request_data.at("filename");
        if (!fs_validate_filename(filename)) {
            res_err(res, format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        std::string filepath = params.slot_save_path + filename;

        server_task task;
        task.type = SERVER_TASK_TYPE_SLOT_SAVE;
        task.data = {
            { "id_slot", id_slot },
            { "filename", filename },
            { "filepath", filepath }
        };

        const int id_task = ctx_server.queue_tasks.post(std::move(task));
        ctx_server.queue_results.add_waiting_task_id(id_task);

        server_task_result result = ctx_server.queue_results.recv(id_task);
        ctx_server.queue_results.remove_waiting_task_id(id_task);

        if (result.error) {
            res_err(res, result.data);
        } else {
            res.set_content(result.data.dump(), "application/json");
        }
    };

    const auto handle_slots_restore = [&ctx_server, &params](const httplib::Request & req, httplib::Response & res, int id_slot) {
        json request_data = json::parse(req.body);
        std::string filename = request_data.at("filename");
        if (!fs_validate_filename(filename)) {
            res_err(res, format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        std::string filepath = params.slot_save_path + filename;

        server_task task;
        task.type = SERVER_TASK_TYPE_SLOT_RESTORE;
        task.data = {
            { "id_slot", id_slot },
            { "filename", filename },
            { "filepath", filepath }
        };

        const int id_task = ctx_server.queue_tasks.post(std::move(task));
        ctx_server.queue_results.add_waiting_task_id(id_task);

        server_task_result result = ctx_server.queue_results.recv(id_task);
        ctx_server.queue_results.remove_waiting_task_id(id_task);

        if (result.error) {
            res_err(res, result.data);
        } else {
            res.set_content(result.data.dump(), "application/json");
        }
    };

    const auto handle_slots_erase = [&ctx_server](const httplib::Request & /* req */, httplib::Response & res, int id_slot) {
        server_task task;
        task.type = SERVER_TASK_TYPE_SLOT_ERASE;
        task.data = {
            { "id_slot", id_slot },
        };

        const int id_task = ctx_server.queue_tasks.post(std::move(task));
        ctx_server.queue_results.add_waiting_task_id(id_task);

        server_task_result result = ctx_server.queue_results.recv(id_task);
        ctx_server.queue_results.remove_waiting_task_id(id_task);

        if (result.error) {
            res_err(res, result.data);
        } else {
            res.set_content(result.data.dump(), "application/json");
        }
    };

    const auto handle_slots_action = [&handle_slots_save, &handle_slots_restore, &handle_slots_erase](const httplib::Request & req, httplib::Response & res) {
        std::string id_slot_str = req.path_params.at("id_slot");
        int id_slot;

        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res_err(res, format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        std::string action = req.get_param_value("action");

        if (action == "save") {
            handle_slots_save(req, res, id_slot);
        } else if (action == "restore") {
            handle_slots_restore(req, res, id_slot);
        } else if (action == "erase") {
            handle_slots_erase(req, res, id_slot);
        } else {
            res_err(res, format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        }
    };

    const auto handle_props = [&ctx_server](const httplib::Request & req, httplib::Response & res) {
        std::string template_key = "tokenizer.chat_template", curr_tmpl;
        int32_t tlen = llama_model_meta_val_str(ctx_server.model, template_key.c_str(), nullptr, 0);
        if (tlen > 0) {
            std::vector<char> curr_tmpl_buf(tlen + 1, 0);
            if (llama_model_meta_val_str(ctx_server.model, template_key.c_str(), curr_tmpl_buf.data(), curr_tmpl_buf.size()) == tlen) {
                curr_tmpl = std::string(curr_tmpl_buf.data(), tlen);
            }
        }
        std::string tmpl_default = common_chat_templates_source(ctx_server.chat_params.tmpls.get(), "");
        std::string tmpl_tools = common_chat_templates_source(ctx_server.chat_params.tmpls.get(), "tool_use");

        json data = {
            { "system_prompt",               ctx_server.system_prompt.c_str() },
            { "model_alias",                 ctx_server.params_base.model_alias },
            { "model_path",                  ctx_server.params_base.model},
            { "default_generation_settings", ctx_server.default_generation_settings_for_props },
            { "total_slots",                 ctx_server.params_base.n_parallel },
            { "model_name",                  get_model_name(ctx_server.params_base.model)},
            { "chat_template",               tmpl_default },
            { "chat_template_caps",      ctx_server.chat_template_caps },
            { "bos_token",                   common_token_to_piece(ctx_server.ctx, llama_token_bos(ctx_server.model), /* special= */ true)},
            { "eos_token",                   common_token_to_piece(ctx_server.ctx, llama_token_eos(ctx_server.model), /* special= */ true)},
            { "model_path",                  ctx_server.params_base.model },
            { "modalities",                  json {
                {"vision", ctx_server.chat_params.allow_image},
                {"audio",  ctx_server.chat_params.allow_audio},
            } },
            { "n_ctx",                       ctx_server.n_ctx },
            // PXA_KV_UNIFIED_v1: n_ctx above is the TOTAL ring. Until now a client had to infer the
            // real per-request ceiling by dividing it by total_slots, which is wrong the moment the
            // ring is unified. Publish the effective ceiling and the mode so it cannot be guessed at.
            { "n_ctx_slot",                  ctx_server.n_ctx_slot },
            { "kv_unified",                  ctx_server.params_base.kv_unified },
            { "kv_cache_used_cells",         llama_get_kv_cache_used_cells(ctx_server.ctx) },
            { "cors_proxy_enabled",          ctx_server.params_base.webui_mcp_proxy},

        };

        if (ctx_server.params_base.use_jinja) {
            if (!tmpl_tools.empty()) {
                data["chat_template_tool_use"] = tmpl_tools;
            }
        }
        res.set_content(data.dump(), "application/json; charset=utf-8");
    };

    const auto handle_props_simple = [&ctx_server](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        int n_past = 0;
        int slot_id = 0;
        for (server_slot& slot : ctx_server.slots) {
            if (slot.n_past > n_past) {
                n_past = slot.n_past;
                slot_id = slot.id;
            }
        }
        json data = {
            { "model_name",                  get_model_name(ctx_server.params_base.model)},
            { "model_path",                  ctx_server.params_base.model },
            { "modalities",                  json {
                {"vision", ctx_server.chat_params.allow_image},
                {"audio",  ctx_server.chat_params.allow_audio},
            } },
             { "n_ctx",                       ctx_server.n_ctx }
        };
        res.set_content(data.dump(), "application/json; charset=utf-8");
    };


    // handle completion-like requests (completion, chat, infill)
    // we can optionally provide a custom format for partial results and final results
    const auto handle_completions_impl = [&ctx_server, &params](
        server_task_type type,
        json& data,
        const std::vector<raw_buffer>& files,
        const std::function<bool()>& is_connection_closed,
        httplib::Response& res,
        oaicompat_type oaicompat) -> void {
            GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

            const auto completion_id = gen_chatcmplid();
            // need to store the reader as a pointer, so that it won't be destroyed when the handle returns
            // use shared_ptr as it's shared between the chunked_content_provider() and on_complete()
            const auto rd = std::make_shared<server_response_reader>(ctx_server);

            try {
                std::vector<server_task> tasks;

                const auto& prompt = data.at("prompt");

                // process prompt
                std::vector<server_tokens> inputs;

                if (oaicompat && ctx_server.mctx != nullptr) {
                    // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
                    inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files));
                }
                else {
                    // Everything else, including multimodal completions.
                    inputs = tokenize_input_prompts(llama_get_vocab(ctx_server.ctx), ctx_server.mctx, prompt, true, true);
                }
                tasks.reserve(inputs.size());
                const std::string requested_model_name = json_value(data, "model", std::string());
                const std::string fallback_model_name = get_model_name(ctx_server.params_base.model);
                const std::string oaicompat_model_name = requested_model_name.empty()
                    ? fallback_model_name
                    : requested_model_name;
                for (size_t i = 0; i < inputs.size(); i++) {
                    server_task task = server_task(type);

                    task.id = ctx_server.queue_tasks.get_new_id();
                    task.index = i;

                    task.tokens = std::move(inputs[i]);
                    task.data = data;
                    //task.params = server_task::params_from_json_cmpl(
                    //    ctx_server.ctx,
                    //    ctx_server.params,
                    //    data);
                    task.id_slot = json_value(data, "id_slot", -1);

                    // OAI-compat
                    task.params.oaicompat = oaicompat;
                    task.params.oaicompat_cmpl_id = completion_id;
                    task.params.oaicompat_model = oaicompat_model_name;
                    tasks.push_back(std::move(task));
                }

                rd->post_tasks(std::move(tasks));
            }
            catch (const std::exception& e) {
                res_err(res, format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
                return;
            }
            bool stream = json_value(data, "stream", false);
            if (!stream) {
                // non-stream, wait for the results
                auto all_results = rd->wait_for_all(is_connection_closed);
                if (all_results.is_terminated) {
                    rd->cancel_active_decode(); // PXA_SCOPED_CANCEL_v1: sole-occupant-gated abort (see cancel_active_decode)
                    return; // connection is closed
                }
                else if (all_results.error) {
                    res_err(res, all_results.error->to_json());
                    return;
                }
                else {
                    json arr = json::array();
                    for (auto& res : all_results.results) {
                        GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                        arr.push_back(res->to_json());
                    }
                    // if single request, return single object instead of array
                    res_ok(res, arr.size() == 1 ? arr[0] : arr);              
                }                       
            }
            else {
                // in streaming mode, the first error must be treated as non-stream response
                // this is to match the OAI API behavior
                // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
                server_task_result_ptr first_result = rd->next(is_connection_closed);
                if (first_result == nullptr) {
                    rd->cancel_active_decode(); // PXA_SCOPED_CANCEL_v1: sole-occupant-gated abort (see cancel_active_decode)
                    return; // connection is closed
                }
                else if (first_result->is_error()) {
                    res_err(res, first_result->to_json());
                    return;
                }
                else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(first_result.get()) != nullptr
                    );
                }
                // next responses are streamed
                json first_result_json = first_result->to_json();
                const auto chunked_content_provider = [first_result_json, rd, oaicompat](size_t, httplib::DataSink& sink) mutable -> bool {
                    const auto sse = [oaicompat, &sink](const json& res) {
                        if (oaicompat == OAICOMPAT_TYPE_ANTHROPIC) {
                            return server_sent_anthropic_event(sink, res);
                        }
                        else if (oaicompat == OAICOMPAT_TYPE_RESP) {
                            return server_sent_oai_resp_event(sink, res);
                        }
                        else {
                            return server_sent_event(sink, res);
                        }
                    };
                    // flush the first result as it's not an error
                    if (!first_result_json.empty()) {
                        if (!sse(first_result_json)) {
                            sink.done();
                            return false; // sending failed, go to on_complete()
                        }
                        first_result_json.clear(); // mark as sent
                    }

                    // receive subsequent results
                    auto result = rd->next([&sink] { return !sink.is_writable(); });
                    if (result == nullptr) {
                        sink.done();
                        return false; // connection is closed, go to on_complete()
                    }

                    // send the results
                    json res_json = result->to_json();
                    bool ok = false;
                    if (result->is_error()) {
                        ok = server_sent_event(sink, json{ { "error", result->to_json() } });
                        sink.done();
                        return false; // go to on_complete()
                    }
                    else {
                        GGML_ASSERT(
                            dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                            || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                        );
                        ok = sse(res_json);
                    }

                    if (!ok) {
                        sink.done();
                        return false; // sending failed, go to on_complete()
                    }

                    // check if there is more data
                    if (!rd->has_next()) {
                        if (oaicompat != OAICOMPAT_TYPE_ANTHROPIC && oaicompat != OAICOMPAT_TYPE_NONE && oaicompat != OAICOMPAT_TYPE_RESP) {
                            static const std::string ev_done = "data: [DONE]\n\n";
                            sink.write(ev_done.data(), ev_done.size());
                        }
                        sink.done();
                        return false; // no more data, go to on_complete()
                    }

                    // has next data, continue
                    return true;
                };

                auto on_complete = [rd](bool) {
                    rd->stop();
                };
                res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
            }
    };

    const auto handle_completions = [&ctx_server, &handle_completions_impl](const httplib::Request & req, httplib::Response & res) {
        log_prompt(ctx_server.params_base, json::parse(req.body));
        auto data = json::parse(req.body);
        std::vector<raw_buffer> files; // dummy
        handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            data,
            files,
            req.is_connection_closed,
            res,
            OAICOMPAT_TYPE_NONE);
    };

    const auto handle_completions_oai = [&ctx_server, &handle_completions_impl](const httplib::Request& req, httplib::Response& res) {
        log_prompt(ctx_server.params_base, json::parse(req.body));
        auto body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(body);
        std::vector<raw_buffer> files; // dummy
        handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            data,
            files,
            req.is_connection_closed,
            res,
            OAICOMPAT_TYPE_COMPLETION);
    };

    const auto handle_models = [&params, &model_meta](const httplib::Request & req, httplib::Response & res) {
        json models = {
            {"object", "list"},
            {"data", {
                 {
                     {"id",       params.model_alias},
                     {"object",   "model"},
                     {"created",  std::time(0)},
                     {"owned_by", "llamacpp"},
                     {"meta",     model_meta},
                     {"max_model_len", params.n_ctx}, //vllm specs
                 },
             }}
        };

        res.set_content(models.dump(), "application/json; charset=utf-8");
    };



    const auto handle_chat_completions = [&ctx_server, &params, &handle_completions_impl](const httplib::Request & req, httplib::Response & res) {
        log_prompt(ctx_server.params_base, json::parse(req.body));
        auto body = json::parse(req.body);
        std::vector<raw_buffer> files;
        json data = oaicompat_chat_params_parse(body, ctx_server.chat_params, files);
        handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            data,
            files,
            req.is_connection_closed,
            res,
            OAICOMPAT_TYPE_CHAT);
    };

    const auto handle_responses = [&ctx_server, &handle_completions_impl](const httplib::Request & req, httplib::Response & res) {
        log_prompt(ctx_server.params_base, json::parse(req.body));
        auto body = json::parse(req.body);
        std::vector<raw_buffer> files;
        json body_parsed = server_chat_convert_responses_to_chatcmpl(body);
        json data = oaicompat_chat_params_parse(body_parsed, ctx_server.chat_params, files);
        handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            data,
            files,
            req.is_connection_closed,
            res,
            OAICOMPAT_TYPE_RESP);
    };

    const auto handle_anthropic_messages = [&ctx_server, &handle_completions_impl](const httplib::Request & req, httplib::Response & res) {
        std::vector<raw_buffer> files;
        log_prompt(ctx_server.params_base, json::parse(req.body));
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            ctx_server.chat_params,
            files);
        return handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            req.is_connection_closed,
            res,
            OAICOMPAT_TYPE_ANTHROPIC);
    };

    const auto handle_anthropic_count_tokens = [&ctx_server, &handle_completions_impl](const httplib::Request & req, httplib::Response & res) {
        std::vector<raw_buffer> files;
        log_prompt(ctx_server.params_base, json::parse(req.body));
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            ctx_server.chat_params,
            files);
        json prompt = body_parsed.at("prompt");
        llama_tokens tokens = tokenize_mixed(llama_get_vocab(ctx_server.ctx), prompt, true, true);
        res_ok(res, { {"input_tokens", static_cast<int>(tokens.size())} });
        return res;
    };

    // same with handle_chat_completions, but without inference part
    const auto handle_apply_template = [&ctx_server, &params](const httplib::Request& req, httplib::Response& res) {
        log_prompt(ctx_server.params_base, json::parse(req.body));
        auto body = json::parse(req.body);
        std::vector<raw_buffer> files; // dummy, unused
        json data = oaicompat_chat_params_parse(body,ctx_server.chat_params, files);
        res_ok(res, { { "prompt", std::move(data.at("prompt")) } });
    };

    const auto handle_infill = [&ctx_server, &handle_completions_impl](const httplib::Request & req, httplib::Response & res) {
        log_prompt(ctx_server.params_base, json::parse(req.body));
        json data = json::parse(req.body);
        //avoid double submits: handle_completions_impl below already queues the infill task
        //const int id_task = ctx_server.queue_tasks.get_new_id();
        //server_tokens token; // dummy tokens
        //ctx_server.queue_results.add_waiting_task_id(id_task);
        //ctx_server.request_completion(id_task, -1, data, true, false, token);
        std::vector<raw_buffer> files; // dummy
        handle_completions_impl(
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            req.is_connection_closed,
            res,
            OAICOMPAT_TYPE_NONE); // infill is not OAI compatible
    };

    const auto handle_tokenize = [&ctx_server](const httplib::Request & req, httplib::Response & res) {
        const json body = json::parse(req.body);

        std::vector<llama_token> tokens;
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            tokens = ctx_server.tokenize(body.at("content"), add_special);
        }
        const json data = format_tokenizer_response(tokens);
        return res.set_content(data.dump(), "application/json; charset=utf-8");
    };

    const auto handle_detokenize = [&ctx_server](const httplib::Request & req, httplib::Response & res) {
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const std::vector<llama_token> tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.ctx, tokens);
        }

        const json data = format_detokenized_response(content);
        return res.set_content(data.dump(), "application/json; charset=utf-8");
    };

    const auto handle_embeddings_impl = [&ctx_server](const httplib::Request& req, httplib::Response& res, oaicompat_type oaicompat) {
        if (!ctx_server.params_base.embedding) {
            res_err(res, format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        if (oaicompat != OAICOMPAT_TYPE_NONE && llama_pooling_type(ctx_server.ctx) == LLAMA_POOLING_TYPE_NONE) {
            res_err(res, format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        const json body = json::parse(req.body);

        // for the shape of input/content, see tokenize_input_prompts()
        json prompt;
        if (body.count("input") != 0) {
            prompt = body.at("input");
        }
        else if (body.contains("content")) {
            oaicompat = OAICOMPAT_TYPE_NONE; // "content" field is not OAI compatible
            prompt = body.at("content");
        }
        else {
            res_err(res, format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        bool use_base64 = false;
        if (body.count("encoding_format") != 0) {
            const std::string& format = body.at("encoding_format");
            if (format == "base64") {
                use_base64 = true;
            }
            else if (format != "float") {
                res_err(res, format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
                return;
            }
        }
        auto vocab = llama_get_vocab(ctx_server.ctx);
        auto tokenized_prompts = tokenize_input_prompts(vocab, ctx_server.mctx, prompt, true, true);
        for (const auto& tokens : tokenized_prompts) {
            // this check is necessary for models that do not add BOS token to the input
            if (tokens.empty()) {
                res_err(res, format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
                return;
            }
        }

        int embd_normalize = 2; // default to Euclidean/L2 norm
        if (body.count("embd_normalize") != 0) {
            embd_normalize = body.at("embd_normalize");
            if (llama_pooling_type(ctx_server.ctx) == LLAMA_POOLING_TYPE_NONE) {
                SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", llama_pooling_type(ctx_server.ctx));
            }
        }

        // create and queue the task
        json responses = json::array();
        server_response_reader rd(ctx_server);
        {
            std::vector<server_task> tasks;
            for (size_t i = 0; i < tokenized_prompts.size(); i++) {
                server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

                task.id = ctx_server.queue_tasks.get_new_id();
                task.index = i;
                task.tokens = std::move(tokenized_prompts[i]);

                // OAI-compat
                task.params.oaicompat = oaicompat;
                task.params.embd_normalize = embd_normalize;
                task.embedding = true; // probably not needed
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.is_connection_closed);

        // collect results
        if (all_results.is_terminated) {
            rd.cancel_active_decode(); // PXA_SCOPED_CANCEL_v1: sole-occupant-gated abort (see cancel_active_decode)
            return; // connection is closed
        }
        else if (all_results.error) {
            res_err(res, all_results.error->to_json());
            return;
        }
        else {
            for (auto& res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = oaicompat == OAICOMPAT_TYPE_EMBEDDING
            ? format_embeddings_response_oaicompat(body, responses, use_base64)
            : json(responses);
        res_ok(res, root);

    };

    const auto handle_embeddings = [&ctx_server, &handle_embeddings_impl](const httplib::Request& req, httplib::Response& res) {
        log_prompt(ctx_server.params_base, json::parse(req.body));
        handle_embeddings_impl(req, res, OAICOMPAT_TYPE_NONE);
    };

    const auto handle_embeddings_oai = [&ctx_server, &handle_embeddings_impl](const httplib::Request& req, httplib::Response& res) {
        log_prompt(ctx_server.params_base, json::parse(req.body));
        handle_embeddings_impl(req, res, OAICOMPAT_TYPE_EMBEDDING);
    };


    const auto handle_lora_adapters_list = [&](const httplib::Request & req, httplib::Response & res) {
        json result = json::array();
        for (size_t i = 0; i < ctx_server.lora_adapters.size(); ++i) {
            auto & la = ctx_server.lora_adapters[i];
            result.push_back({
                {"id", i},
                {"path", la.path},
                {"scale", la.scale},
            });
        }
        res.set_content(result.dump(), "application/json");
        res.status = 200; // HTTP OK
    };


    const auto handle_lora_adapters_apply = [&](const httplib::Request & req, httplib::Response & res) {
        log_prompt(ctx_server.params_base, json::parse(req.body));
        const std::vector<json> body = json::parse(req.body);
        int max_idx = ctx_server.lora_adapters.size();

        // clear existing value
        for (auto & la : ctx_server.lora_adapters) {
            la.scale = 0.0f;
        }

        // set value
        for (auto entry : body) {
            int id      = entry.at("id");
            float scale = entry.at("scale");
            if (0 <= id && id < max_idx) {
                ctx_server.lora_adapters[id].scale = scale;
            } else {
                throw std::runtime_error("invalid adapter id");
            }
        }

        server_task task;
        task.type = SERVER_TASK_TYPE_SET_LORA;
        const int id_task = ctx_server.queue_tasks.post(std::move(task));
        ctx_server.queue_results.add_waiting_task_id(id_task);

        server_task_result result = ctx_server.queue_results.recv(id_task);
        ctx_server.queue_results.remove_waiting_task_id(id_task);

        res.set_content(result.data.dump(), "application/json");
        res.status = 200; // HTTP OK
    };

    // Control vector handlers
    const auto handle_control_vectors_list = [&](const httplib::Request & req, httplib::Response & res) {
        json result = json::array();
        for (size_t i = 0; i < ctx_server.control_vectors.size(); ++i) {
            auto & cv = ctx_server.control_vectors[i];
            result.push_back({
                {"id", i},
                {"path", cv.path},
                {"scale", cv.scale},
                {"layer_start", cv.layer_start},
                {"layer_end", cv.layer_end},
                {"applied", cv.applied},
            });
        }
        res.set_content(result.dump(), "application/json");
        res.status = 200; // HTTP OK
    };

    const auto handle_control_vectors_load = [&](const httplib::Request & req, httplib::Response & res) {
        const json body = json::parse(req.body);

        server_task task;
        task.type = SERVER_TASK_TYPE_LOAD_CONTROL_VECTOR;
        task.data = body;

        const int id_task = ctx_server.queue_tasks.post(std::move(task));
        ctx_server.queue_results.add_waiting_task_id(id_task);

        server_task_result result = ctx_server.queue_results.recv(id_task);
        ctx_server.queue_results.remove_waiting_task_id(id_task);

        res.set_content(result.data.dump(), "application/json");
        res.status = result.error ? 400 : 200;
    };

    const auto handle_control_vectors_unload = [&](const httplib::Request & req, httplib::Response & res) {
        const json body = json::parse(req.body);

        server_task task;
        task.type = SERVER_TASK_TYPE_UNLOAD_CONTROL_VECTOR;
        task.data = body;

        const int id_task = ctx_server.queue_tasks.post(std::move(task));
        ctx_server.queue_results.add_waiting_task_id(id_task);

        server_task_result result = ctx_server.queue_results.recv(id_task);
        ctx_server.queue_results.remove_waiting_task_id(id_task);

        res.set_content(result.data.dump(), "application/json");
        res.status = result.error ? 400 : 200;
    };

    const auto handle_control_vectors_apply = [&](const httplib::Request & req, httplib::Response & res) {
        const std::vector<json> body = json::parse(req.body);
        int max_idx = ctx_server.control_vectors.size();

        // Update scales for existing control vectors
        for (auto & cv : ctx_server.control_vectors) {
            cv.scale = 0.0f;  // Reset all scales first
        }

        // Set new scales
        for (auto entry : body) {
            int id = entry.at("id");
            float scale = entry.at("scale");
            if (0 <= id && id < max_idx) {
                ctx_server.control_vectors[id].scale = scale;

                // Optionally update layer range
                if (entry.contains("layer_start")) {
                    ctx_server.control_vectors[id].layer_start = entry.at("layer_start");
                }
                if (entry.contains("layer_end")) {
                    ctx_server.control_vectors[id].layer_end = entry.at("layer_end");
                }
            } else {
                res.set_content(json{{ "success", false }, { "error", "Invalid control vector id" }}.dump(), "application/json");
                res.status = 400;
                return;
            }
        }

        server_task task;
        task.type = SERVER_TASK_TYPE_SET_CONTROL_VECTOR;

        const int id_task = ctx_server.queue_tasks.post(std::move(task));
        ctx_server.queue_results.add_waiting_task_id(id_task);

        server_task_result result = ctx_server.queue_results.recv(id_task);
        ctx_server.queue_results.remove_waiting_task_id(id_task);

        res.set_content(result.data.dump(), "application/json");
        res.status = result.error ? 400 : 200;
    };

    const auto list_saved_prompts = [&ctx_server, &params](const httplib::Request& req, httplib::Response& res) {
        json response = json::array();

        try {
            for (const auto& entry : fs::directory_iterator(params.slot_save_path)) {
                if (!entry.is_regular_file() || entry.file_size() < 12) {
                    continue;
                }

                std::ifstream file(entry.path(), std::ios::binary);
                if (!file) continue;

                uint32_t magic, version, n_token_count;
                file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
                file.read(reinterpret_cast<char*>(&version), sizeof(version));
                file.read(reinterpret_cast<char*>(&n_token_count), sizeof(n_token_count));

                if (magic != LLAMA_STATE_SEQ_MAGIC ||
                    version != LLAMA_STATE_SEQ_VERSION ||
                    entry.file_size() < (12 + (n_token_count * sizeof(llama_token)))) {
                    continue;
                }

                std::vector<llama_token> tokens(n_token_count);
                file.read(reinterpret_cast<char*>(tokens.data()), tokens.size() * sizeof(llama_token));

                //C++17 is not modern enough to have a nice and portable way to get the mtime of a file
                //so the following seems to be needed
                auto ftime = fs::last_write_time(entry.path());
                auto system_time = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                    ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now()
                );
                std::time_t c_time = std::chrono::system_clock::to_time_t(system_time);
                std::tm tm_struct;
                #if defined(_WIN32)
                localtime_s(&tm_struct, &c_time);
                #else
                localtime_r(&c_time, &tm_struct);
                #endif
                std::ostringstream oss;
                oss << std::put_time(&tm_struct, "%Y-%m-%d %H:%M:%S");
                auto str_time = oss.str();


                response.push_back({
                    {"filename", entry.path().filename().string()},
                    {"filesize", entry.file_size()},
                    {"mtime", str_time},
                    {"token_count", n_token_count},
                    {"prompt", tokens_to_str(ctx_server.ctx, tokens)}
                });
            }
        } catch (const std::exception& e) {
            res.status = 500;
            response = {{"error", e.what()}};
        }
        res.set_content(response.dump(), "application/json; charset=utf-8");
    };

    const auto list_slot_prompts = [&ctx_server, &params](const httplib::Request& req, httplib::Response& res) {
        json response = json::array();
        for (server_slot & slot : ctx_server.slots) {
            response.push_back({
                {"slot_id", slot.id},
                {"token_count", slot.cache_tokens.size()},
                {"prompt", slot.cache_tokens.detokenize(ctx_server.ctx, true) }
            });
        }
        res.set_content(response.dump(), "application/json; charset=utf-8");
    };


    const auto delete_saved_prompt = [&ctx_server, &params](const httplib::Request& req, httplib::Response& res)-> void {
        json response;
        namespace fs = std::filesystem;

        try {
            const json body = json::parse(req.body);
            const std::string filename_str = body.at("filename");

            // prevent directory traversal attacks
            if (filename_str.find("..") != std::string::npos || filename_str.find('/') != std::string::npos || filename_str.find('\\') != std::string::npos) {
                res.status = 400;
                response = {{"error", "Invalid filename format."}};
                res.set_content(response.dump(), "application/json; charset=utf-8");
                return;
            }

            const fs::path file_to_delete = fs::path(params.slot_save_path) / fs::path(filename_str);

            if (!fs::exists(file_to_delete) || !fs::is_regular_file(file_to_delete)) {
                res.status = 404;
                response = {{"error", "File not found."}};
                res.set_content(response.dump(), "application/json; charset=utf-8");
                return;
            }

            if (fs::remove(file_to_delete)) {
                response = {
                    {"status", "deleted"},
                    {"filename", filename_str}
                };
            } else {
                res.status = 500;
                response = {{"error", "Failed to delete the file."}};
            }
        } catch (const json::parse_error& e) {
            res.status = 400;
            response = {{"error", "Invalid JSON request body."}};
        } catch (const json::out_of_range& e) {
            res.status = 400;
            response = {{"error", "Missing 'filename' key in request body."}};
        } catch (const std::exception& e) {
            res.status = 500;
            response = {{"error", e.what()}};
        }
        res.set_content(response.dump(), "application/json; charset=utf-8");
    };

    const auto rename_saved_prompt = [&ctx_server, &params](const httplib::Request& req, httplib::Response& res)-> void {
        json response;
        namespace fs = std::filesystem;

        try {
            const json body = json::parse(req.body);
            const std::string old_filename_str = body.at("old_filename");
            const std::string new_filename_str = body.at("new_filename");

            if (old_filename_str.find("..") != std::string::npos || old_filename_str.find_first_of("/\\") != std::string::npos ||
                new_filename_str.find("..") != std::string::npos || new_filename_str.find_first_of("/\\") != std::string::npos) {
                res.status = 400;
                response = {{"error", "Invalid filename format."}};
                res.set_content(response.dump(), "application/json; charset=utf-8");
                return;
            }

            const fs::path old_path = fs::path(params.slot_save_path) / old_filename_str;
            const fs::path new_path = fs::path(params.slot_save_path) / new_filename_str;

            if (!fs::exists(old_path) || !fs::is_regular_file(old_path)) {
                res.status = 404;
                response = {{"error", "Source file not found."}};
                res.set_content(response.dump(), "application/json; charset=utf-8");
                return;
            }

            if (fs::exists(new_path)) {
                res.status = 409;
                response = {{"error", "Destination filename already exists."}};
                res.set_content(response.dump(), "application/json; charset=utf-8");
                return;
            }

            std::error_code ec;
            fs::rename(old_path, new_path, ec);

            if (ec) {
                res.status = 500;
                response = {{"error", "Failed to rename file: " + ec.message()}};
            } else {
                response = {
                    {"status", "renamed"},
                    {"old_filename", old_filename_str},
                    {"new_filename", new_filename_str}
                };
            }

        } catch (const json::parse_error& e) {
            res.status = 400;
            response = {{"error", "Invalid JSON request body."}};
        } catch (const json::out_of_range& e) {
            res.status = 400;
            response = {{"error", "Missing 'old_filename' or 'new_filename' in request body."}};
        } catch (const std::exception& e) {
            res.status = 500;
            response = {{"error", e.what()}};
        }

        res.set_content(response.dump(), "application/json; charset=utf-8");
    };

    auto handle_static_file = [](unsigned char * content, size_t len, const char * mime_type) {
        return [content, len, mime_type](const httplib::Request &, httplib::Response & res) {
            res.set_content(reinterpret_cast<const char*>(content), len, mime_type);
            return false;
        };
    };
#ifdef SQLITE3_MODERN_CPP_SUPPORT
    const auto handle_version = [&params, sqlite_extension_loaded](const httplib::Request&, httplib::Response& res) {
        res.set_content(
            json{{"version", 4},
            {"features", {{"sql", !params.sql_save_file.empty()}, {"zstd_compression", sqlite_extension_loaded}}}}.dump(),
            "application/json"
        );
    };
#else
    const auto handle_version = [](const httplib::Request&, httplib::Response& res)-> void {
        res.set_content(
             json{{"version", 4},
             {"features", {{"sql", false}, {"zstd_compression", false}}}}.dump(),
             "application/json"
        );
    };
#endif

#ifdef SQLITE3_MODERN_CPP_SUPPORT
    auto db_handler = [db_handle](auto func) {
        return [func, db_handle](const httplib::Request& req, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
	    try {
                const json body = !req.body.empty() ? json::parse(req.body) : json::object();
                func(*db_handle, body, req, res);
            } catch(const std::exception& e) {
                res.status = 500;
                res.set_content(
                    json{{"ok", false}, {"message", e.what()}}.dump(),
                    "application/json"
                );
            }
        };
    };
#else
    auto db_handler = [db_handle](auto func) {
        return [func, db_handle](const httplib::Request& req, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
            res.status = 500;
            res.set_content(
                json{{"ok", false}, {"message", "Sqlite3 support was not enabled. Recompile with '-DLLAMA_SERVER_SQLITE3=ON'"}}.dump(),
                "application/json"
            );
        };
    };
#endif

    const auto normalize_store_name = [](const std::string& storeName) {
        if(storeName.empty()) return std::string("sessions");

        std::string normalized;
        normalized.reserve(storeName.size());

        for(char c : storeName) {
            if(std::isalpha(static_cast<unsigned char>(c))) {
                normalized.push_back(std::tolower(static_cast<unsigned char>(c)));
            }
        }

        return normalized.empty() ? "sessions" : normalized;
    };

    const auto get_key_string = [](const json& j) {
        return j.is_string() ? j.get<std::string>() : j.dump();
    };


    const auto handle_load = db_handler([normalize_store_name, get_key_string](auto& db, const json& body, auto&, auto& res) {
        std::string data;
	const std::string store = normalize_store_name(body["storeName"]);
	db.db << "SELECT data FROM " + store + " WHERE key = ?" << get_key_string(body["key"]) >> data;
	if(data.empty()) {
            res.status = 404;
            res.set_content(json{{"ok", false}, {"message", "Key not found"}}.dump(), "application/json");
        } else {
            json response{{"ok", true}};
	    response["result"] = (store == "names") ? json(data) : json::parse(data);
            res.set_content(response.dump(), "application/json");
        }
    });

    const auto handle_save = db_handler([normalize_store_name, get_key_string](auto& db, const json& body, auto&, auto& res) {
        const std::string store = normalize_store_name(body["storeName"]);
        const std::string data = (store == "names") ? body["data"].get<std::string>() : body["data"].dump();
        db.db << "INSERT OR REPLACE INTO " + store + " (key, data) VALUES (?, ?)" << get_key_string(body["key"]) << data;
        res.set_content(json{{"ok", true}, {"result", "Data saved successfully"}}.dump(), "application/json");
    });

    const auto handle_rename = db_handler([get_key_string](auto& db, const json& body, auto&, auto& res) {
        db.db << "UPDATE names SET data = ? WHERE key = ?"
            << body["newName"].get<std::string>()
            << get_key_string(body["key"]);
        res.set_content(json{{"ok", true}, {"result", "Session renamed successfully"}}.dump(), "application/json");
    });

    const auto handle_all = db_handler([normalize_store_name](auto& db, const json& body, auto&, auto& res) {
        json result = json::object();
        db.db << "SELECT key, data FROM " + normalize_store_name(body["storeName"]) >>
            [&](const std::string& key, const std::string& data) {
                result[key] = json::parse(data);
            };
        res.set_content(json{{"ok", true}, {"result", result}}.dump(), "application/json");
    });

    const auto handle_sessions = db_handler([](auto& db, const json& body, auto&, auto& res) {
        json result = json::object();
        db.db << "SELECT key, data FROM names" >> [&](const std::string& key, const std::string& data) {
            result[key] = data;
        };
        res.set_content(json{{"ok", true}, {"result", result}}.dump(), "application/json");
    });

    const auto handle_delete = db_handler([normalize_store_name, get_key_string](auto& db, const json& body, auto&, auto& res) {
        db.db << "DELETE FROM " + normalize_store_name(body["storeName"]) + " WHERE key = ?"
            << get_key_string(body["key"]);
        res.set_content(json{{"ok", true}, {"result", "Session deleted successfully"}}.dump(), "application/json");
    });

    const auto handle_vacuum = db_handler([](auto& db, const json& body, auto&, auto& res) {
        json result = json::object();
        db.db << "VACUUM";
        res.set_content(json{"ok", true}.dump(), "application/json");
    });

    const auto handle_zstd_get_configs = db_handler([](auto& db, const json& body, auto&, auto& res) {
        json result = json::object();
        db.db << "SELECT id, config FROM _zstd_configs" >> [&](const std::string id, const std::string& config) {
            result[id] = config;
        };
        res.set_content(json{{"ok", true}, {"configs", result}}.dump(), "application/json");
    });

    const auto handle_zstd_maintenance = db_handler([](auto& db, const json& body, auto&, auto& res) {
        std::string data;
        if (body["duration"].is_null()) {
            db.db << "select zstd_incremental_maintenance(?, ?)" <<  nullptr << body["db_load"].get<double>() >> data;
        }
	else {
            db.db << "select zstd_incremental_maintenance(?, ?)" << body["duration"].get<double>() << body["db_load"].get<double>() >> data;
        }
        json response{{"ok", true}};
        response["result"] = json::parse(data);
        res.set_content(response.dump(), "application/json");
    });

    const auto handle_zstd_enable = db_handler([](auto& db, const json& body, auto&, auto& res) {
        db.db << "select zstd_enable_transparent('{\"table\": \"" + body["table"].get<std::string>() + "\",\"column\": \"" + body["column"].get<std::string>() + "\", \"compression_level\": " + std::to_string(body["compression_level"].get<int>()) + ", \"dict_chooser\": \"''a''\", \"train_dict_samples_ratio\": " + std::to_string(body["train_dict_samples_ratio"].get<int>()) + "}')";
        res.set_content(json{"ok", true}.dump(), "application/json");
    });

    const auto handle_zstd_config_update = db_handler([](auto& db, const json& body, auto&, auto& res) {
        std::string patch_json = "{\"compression_level\": " + std::to_string(body["compression_level"].get<int>()) + ", \"train_dict_samples_ratio\": " + std::to_string(body["train_dict_samples_ratio"].get<int>()) + "}";
        db.db << "update _zstd_configs set config = json_patch(config, '" + patch_json + "')";
        res.set_content(json{{"ok", true}}.dump(), "application/json");
    });

    //
    // Router
    //
    if (params.webui == COMMON_WEBUI_NONE) {
        LLAMA_LOG_INFO("Web UI is disabled\n");
    }
    else {
        // register static assets routes
        if (!params.public_path.empty()) {
            // Set the base directory for serving static files
            svr->set_base_dir(params.public_path);
        }

        {
            // register static assets routes
            if (!params.public_path.empty()) {
                // Set the base directory for serving static files
                bool is_found = svr->set_mount_point("/", params.public_path);
                if (!is_found) {
                    GGML_ABORT("%s: static assets path not found: %s\n", __func__, params.public_path.c_str());
                    return 1;
                }
            }
            else {

                // using embedded static index.html
                svr->Get("/", [params](const httplib::Request& req, httplib::Response& res) {
                    if (req.get_header_value("Accept-Encoding").find("gzip") == std::string::npos) {
                        res.set_content("Error: gzip is not supported by this browser", "text/plain");
                    }
                    else {
                        res.set_header("Content-Encoding", "gzip");
                        // COEP and COOP headers, required by pyodide (python interpreter)
                        res.set_header("Cross-Origin-Embedder-Policy", "require-corp");
                        res.set_header("Cross-Origin-Opener-Policy", "same-origin");
                        if (params.webui == COMMON_WEBUI_AUTO) {
                            res.set_content(reinterpret_cast<const char*>(index_html_gz), index_html_gz_len, "text/html; charset=utf-8");
                        }
                        else if (params.webui == COMMON_WEBUI_LLAMACPP) {
                            res.set_content(reinterpret_cast<const char*>(index_llamacpp_html_gz), index_llamacpp_html_gz_len, "text/html; charset=utf-8");
                        }
                        else {
                            res.set_content(reinterpret_cast<const char*>(index_html_gz), index_html_gz_len, "text/html; charset=utf-8");
                        }
                    }
                    return false;
                    });
            }
        }
    }
    // register API routes
    svr->Get ("/health",              handle_health);
    svr->Get ("/metrics",             handle_metrics);
    svr->Get ("/props",               handle_props);
    svr->Get("/v1/props",             handle_props_simple);
    svr->Get ("/v1/models",           handle_models);
    svr->Post("/completion",          handle_completions); // legacy
    svr->Post("/completions", handle_completions); // legacy
    svr->Post("/v1/completions",     handle_completions_oai);
    svr->Post("/chat/completions",    handle_chat_completions);
    svr->Post("/v1/chat/completions", handle_chat_completions);
    svr->Post("/v1/responses",        handle_responses);
    svr->Post("/v1/messages",         handle_anthropic_messages);
    svr->Post("/v1/messages/count_tokens", handle_anthropic_count_tokens);
    svr->Post("/infill",              handle_infill);
    svr->Post("/embedding",           handle_embeddings); // legacy
    svr->Post("/embeddings",          handle_embeddings);
    svr->Post("/v1/embeddings",       handle_embeddings_oai);
    svr->Post("/tokenize",            handle_tokenize);
    svr->Post("/detokenize",          handle_detokenize);
    svr->Post("/apply-template",      handle_apply_template);
    // LoRA adapters hotswap
    svr->Get ("/lora-adapters",       handle_lora_adapters_list);
    svr->Post("/lora-adapters",       handle_lora_adapters_apply);
    // Control vectors
    svr->Get ("/control-vectors",       handle_control_vectors_list);
    svr->Post("/control-vectors/load",   handle_control_vectors_load);
    svr->Post("/control-vectors/unload", handle_control_vectors_unload);
    svr->Post("/control-vectors/apply",  handle_control_vectors_apply);
    // Save & load slots
    svr->Get ("/slots",               handle_slots);
    svr->Get ("/slots/list",          list_slot_prompts);
    if (!params.slot_save_path.empty()) {
        // these endpoints rely on slot_save_path existing
        svr->Post("/slots/:id_slot",  handle_slots_action);
        svr->Get ("/list",            list_saved_prompts);
        svr->Post("/delete_prompt",   delete_saved_prompt);
        svr->Post("/rename_prompt",   rename_saved_prompt);

    }

    svr->Get ("/version", handle_version);
    if (!params.sql_save_file.empty()) {
        // these endpoints rely on sql_save_file existing
        svr->Post("/load", handle_load);
        svr->Post("/save", handle_save);
        svr->Post("/rename", handle_rename);
        svr->Post("/all", handle_all);
        svr->Post("/sessions", handle_sessions);
        svr->Get ("/sessions", handle_sessions);
        svr->Post("/delete", handle_delete);
        //VACUUM is there for the extension but does not require the extension
        svr->Get ("/vacuum", handle_vacuum);
#ifdef SQLITE3_MODERN_CPP_SUPPORT
        if (sqlite_extension_loaded) {
            svr->Get ("/zstd_get_configs", handle_zstd_get_configs);
            svr->Post("/zstd_incremental_maintenance", handle_zstd_maintenance);
            svr->Post("/zstd_enable_transparent", handle_zstd_enable);
            svr->Post("/zstd_update_transparent", handle_zstd_config_update);
	}
#endif
    }

    // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
    if (params.webui_mcp_proxy) {
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "CORS proxy is enabled, do not expose server to untrusted environments\n");
        SRV_WRN("%s", "This feature is EXPERIMENTAL and may be removed or changed in future versions\n");
        SRV_WRN("%s", "-----------------\n");
        svr->Get("/cors-proxy", proxy_handler_get);
        svr->Post("/cors-proxy", proxy_handler_post);
    }
    //
    // Start the server
    //
    if (params.n_threads_http < 1) {
        // +2 threads for monitoring endpoints
        params.n_threads_http = std::max(params.n_parallel + 2, (int32_t) std::thread::hardware_concurrency() - 1);
    }
    log_data["n_threads_http"] =  std::to_string(params.n_threads_http);
    svr->new_task_queue = [&params] { return new httplib::ThreadPool(params.n_threads_http); };

    LOG_INFO("HTTP server listening", log_data);

    // run the HTTP server in a thread - see comment below
    std::thread t([&]() {
        if (!svr->listen_after_bind()) {
            state.store(SERVER_STATE_ERROR);
            return 1;
        }

        return 0;
    });

    ctx_server.queue_tasks.on_new_task([&ctx_server](server_task && task) {
        ctx_server.process_single_task(std::move(task));
        });
    ctx_server.queue_tasks.on_finish_multitask(std::bind(
        &server_context::on_finish_multitask, &ctx_server, std::placeholders::_1));
    ctx_server.queue_tasks.on_update_slots(std::bind(
        &server_context::update_slots, &ctx_server));
    // PXA_DEFER_PUMP_v1: lets the queue loop revive parked tasks the moment a slot frees
    ctx_server.queue_tasks.on_slot_available([&ctx_server]() {
        bool any_available = false;
        for (const auto & slot : ctx_server.slots) {
            if (slot.available()) {
                any_available = true;
                break;
            }
        }
        if (!any_available) {
            return false;
        }
        // PXA_KV_UNIFIED_v1: under the shared ring a free slot is no longer sufficient -- tasks are
        // also parked for lack of KV cells. Re-dispatching into a ring that is still full only makes
        // them bounce straight back to the deferred queue, so require some headroom as well. (The
        // pump already waits up to 500 ms per cycle, so a bounce is a poll and not a spin; this check
        // keeps the churn and the log noise down.) Attention KV cells only -- the per-sequence
        // recurrent GDN state is not in this ring and is not part of this decision.
        if (ctx_server.params_base.kv_unified) {
            const int32_t used    = llama_get_kv_cache_used_cells(ctx_server.ctx);
            const int32_t pending = ctx_server.kv_unified_cells_busy(nullptr);
            if (ctx_server.n_ctx - used - pending <= ctx_server.kv_unified_reserve) {
                return false;
            }
        }
        return true;
        });
    ctx_server.queue_results.on_multitask_update(std::bind(
        &server_queue::update_multitask,
        &ctx_server.queue_tasks,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3
    ));

    shutdown_handler = [&](int) {
        ctx_server.queue_tasks.terminate();
    };

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    // PXA_WEDGE_EXIT_v1 (container-aware since 2026-07-30): honest self-exit on a genuinely
    // hung forward pass (a driver/CUDA wedge no in-process logic can recover — post-Xid the
    // context is poisoned on these cards). A single llama_decode call stuck in flight for
    // > PXA_WEDGE_EXIT_MS (default 180000; 0 = off) across 3 consecutive 5s checks, then:
    //   * container:   _exit(42) — the orchestrator (--restart / the belt) reloads fresh.
    //   * bare metal:  no supervisor exists, so exit-and-hope would just leave the port dead.
    //                  First try in-process recovery: llama_decode_stop() aborts a SOFT wedge
    //                  at the next abort-callback poll (the scoped ret=-3 unwind releases the
    //                  slots honestly and errors the clients — nobody hangs). If the decode is
    //                  STILL in flight 3 checks later it is a hard driver wedge: _exit(41)
    //                  with a LOUD "no supervisor detected, restart me" banner. 41 != 42 so
    //                  logs can tell the two contracts apart.
    // In NEITHER mode does this monitor fork, re-exec, or spawn anything: the process only
    // ever goes away; an EXTERNAL actor brings it back. (The 2026-07-30 duplicate-server
    // incident was the abort-path fork in ggml_print_backtrace — fixed as PXA_BT_NOFORK_v1 —
    // not this monitor; this monitor must never grow such a path.)
    // A legitimate decode call is one ub chunk (seconds even at deep context), so 180s
    // in-flight is unambiguous. Detached: you cannot join a thread that is stuck in the driver.
    {
        const char * pxa_we_env = getenv("PXA_WEDGE_EXIT_MS");
        const int64_t pxa_wedge_exit_ms = pxa_we_env ? atoll(pxa_we_env) : 180000LL;
        if (pxa_wedge_exit_ms > 0) {
            std::thread([&ctx_server, pxa_wedge_exit_ms, pxa_in_container]() {
                const int strikes_max = pxa_in_container ? 3 : 6;
                int  strikes = 0;
                bool recovery_tried = false;
                for (;;) {
                    std::this_thread::sleep_for(std::chrono::seconds(5));
                    if (ctx_server.pxa_decode_wedged(pxa_wedge_exit_ms)) {
                        ++strikes;
                        fprintf(stderr, "PXA_WEDGE_EXIT_v1: llama_decode in flight > %lld ms (strike %d/%d)\n",
                                (long long) pxa_wedge_exit_ms, strikes, strikes_max);
                        fflush(stderr);
                        if (pxa_in_container) {
                            if (strikes >= strikes_max) {
                                fprintf(stderr, "PXA_WEDGE_EXIT_v1: confirmed hung forward pass — self-exit(42) for a fresh restart by the container supervisor\n");
                                fflush(stderr);
                                _exit(42);
                            }
                        } else {
                            if (strikes == 3 && !recovery_tried) {
                                recovery_tried = true;
                                fprintf(stderr, "PXA_WEDGE_EXIT_v1: BARE-METAL, no container supervisor — attempting in-process recovery via llama_decode_stop() (soft-wedge abort)\n");
                                fflush(stderr);
                                llama_decode_stop();
                            } else if (strikes >= strikes_max) {
                                fprintf(stderr,
                                    "PXA_WEDGE_EXIT_v1: =============================================================\n"
                                    "PXA_WEDGE_EXIT_v1: HARD WEDGE — in-process recovery failed, the forward pass is\n"
                                    "PXA_WEDGE_EXIT_v1: stuck in the driver. NO CONTAINER SUPERVISOR WAS DETECTED, so\n"
                                    "PXA_WEDGE_EXIT_v1: NOTHING will restart this server automatically.\n"
                                    "PXA_WEDGE_EXIT_v1: Exiting with code 41 — the OPERATOR must restart it manually.\n"
                                    "PXA_WEDGE_EXIT_v1: (Set PXA_IN_CONTAINER=1 to force the container contract, or\n"
                                    "PXA_WEDGE_EXIT_v1:  PXA_WEDGE_EXIT_MS=0 to disable this monitor entirely.)\n"
                                    "PXA_WEDGE_EXIT_v1: =============================================================\n");
                                fflush(stderr);
                                _exit(41);
                            }
                        }
                    } else {
                        strikes = 0;
                        recovery_tried = false;
                    }
                }
            }).detach();
            LOG_INFO("PXA_WEDGE_EXIT_v1 armed", {{"stall_ms", pxa_wedge_exit_ms}, {"runtime", pxa_in_container ? "container" : "bare-metal"}});
        }
    }

    ctx_server.queue_tasks.start_loop();

    svr->stop();
    t.join();

    llama_backend_free();

    return 0;
}
