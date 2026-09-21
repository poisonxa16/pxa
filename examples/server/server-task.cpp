#include "server-task.h"
#include "server-chat.h"

#include <fstream>
#include <cstdio>

// PXA_CACHE_DISK_v1: fsync of the temporary file and of its directory before/after the rename.
#include <fcntl.h>
#include <unistd.h>

json result_timings::to_json() const {
    json base = {
        {"prompt_n",               prompt_n},
        {"prompt_ms",              prompt_ms},
        {"prompt_per_token_ms",    prompt_per_token_ms},
        {"prompt_per_second",      prompt_per_second},

        {"predicted_n",            predicted_n},
        {"predicted_ms",           predicted_ms},
        {"predicted_per_token_ms", predicted_per_token_ms},
        {"predicted_per_second",   predicted_per_second},

        {"n_ctx",           n_ctx},
        {"n_past",           n_past},
    };

    if (draft_n > 0) {
        base["draft_n"] = draft_n;
        base["draft_n_accepted"] = draft_n_accepted;
    }

    if (forked_n > 0) {
        base["forked_n"] = forked_n; // PXA_SLOT_FORK_v1
    }

    return base;
}


//json server_task_result_cmpl_partial::to_json_non_oaicompat_partial() {
//    // non-OAI-compat JSON
//    json res = json{
//        {"index",            index},
//        {"content",          content},
//        {"tokens",           tokens},
//        {"stop",             false},
//        {"id_slot",          id_multi},
//        {"tokens_predicted", n_decoded},
//        {"tokens_evaluated", n_prompt_tokens},
//    };
//    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
//    if (timings.prompt_n > 0) {
//        res.push_back({ "timings", timings.to_json() });
//    }
//    if (!probs_output.empty()) {
//        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
//    }
//    return res;
//}

//json server_task_result_cmpl_final::to_json_non_oaicompat_final() {
//    json res = json{
//        {"index",               index},
//        {"content",             stream ? "" : content}, // in stream mode, content is already in last partial chunk
//        {"tokens",              stream ? std::vector<llama_token> {} : tokens},
//        {"id_slot",             id_multi},
//        {"stop",                true},
//        {"model",               oaicompat_model},
//        {"tokens_predicted",    n_decoded},
//        {"tokens_evaluated",    n_prompt_tokens},
//        //{"generation_settings", default_generation_settings_for_props.to_json()},
//        {"prompt",              prompt},
//        {"has_new_line",        has_new_line},
//        {"truncated",           truncated},
//        //{"stop_type",           stop_type_to_str(STOP_TYPE_EOS)},
//        {"stopping_word",       stopping_word},
//        {"tokens_cached",       n_tokens_cached},
//        {"timings",             timings.to_json()},
//    };
//    if (!stream && !probs_output.empty()) {
//        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
//    }
//    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
//}

json server_task_result_cmpl_partial::to_json_non_oaicompat_partial() {
    // non-OAI-compat JSON
    return data;
}

json server_task_result_cmpl_final::to_json_non_oaicompat_final() {
    // non-OAI-compat JSON
    return data;
}

json server_task_result_cmpl_partial::to_json_oaicompat_partial() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json res = json{
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"object",             "text_completion"},
        {"usage", json {
            {"completion_tokens", n_decoded},
            {"prompt_tokens",     n_prompt_tokens},
            {"total_tokens",      n_decoded + n_prompt_tokens}
        }},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat_partial();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({ "timings", timings.to_json() });
    }

    return res;
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    return json{
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
}


json server_task_result_cmpl_final::to_json_oaicompat_final() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop_kind == STOP_TYPE_WORD || stop_kind == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json{
        {"choices",            json::array({
            json{
                {"text",          stream ? "" : content}, // in stream mode, content is already in last partial chunk
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat_final();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({ "timings", timings.to_json() });
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat_partial() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json& delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", 0},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"object", "chat.completion.chunk"},
            // PXA_STREAM_SLIM: no per-chunk usage object (OAI semantics: usage only on the final chunk
            // via stream_options.include_usage, which is handled in the final serializer). ~30% smaller chunks.
            });
    };
    // We have to send an initial update to conform to openai behavior
    if (first) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
            });
    }

    for (const auto& diff : oaicompat_msg_diffs) {        
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        GGML_ASSERT(deltas[deltas.size() - 1].at("choices").size() >= 1);

        if (probs_output.size() > 0) {
            deltas[deltas.size() - 1].at("choices").at(0)["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
            };
        }

        if (timings.prompt_n >= 0) {
            deltas[deltas.size() - 1].push_back({ "timings", timings.to_json() });
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp_partial() {
    std::vector<json> events;

    if (n_decoded == 1) {
        events.push_back(json{
            {"event", "response.created"},
            {"data", json{
                {"type", "response.created"},
                {"response", json{
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json{
            {"event", "response.in_progress"},
            {"data", json{
                {"type", "response.in_progress"},
                {"response", json{
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const auto& diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!oai_resp_thinking_block_started) {
                events.push_back(json{
                    {"event", "response.output_item.added"},
                    {"data", json{
                        {"type", "response.output_item.added"},
                        {"item", json{
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                oai_resp_thinking_block_started = true;
            }
            events.push_back(json{
                {"event", "response.reasoning_text.delta"},
                {"data", json{
                    {"type",    "response.reasoning_text.delta"},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!oai_resp_text_block_started) {
                events.push_back(json{
                    {"event", "response.output_item.added"},
                    {"data", json{
                        {"type", "response.output_item.added"},
                        {"item", json{
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json{
                    {"event", "response.content_part.added"},
                    {"data", json{
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"part", json{
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                oai_resp_text_block_started = true;
            }
            events.push_back(json{
                {"event", "response.output_text.delta"},
                {"data", json{
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json{
                {"event", "response.output_item.added"},
                {"data", json{
                    {"type",  "response.output_item.added"},
                    {"item", json{
                        {"arguments", ""},
                        {"call_id",   "fc_" + diff.tool_call_delta.id},
                        {"name",      diff.tool_call_delta.name},
                        {"type",      "function_call"},
                        {"status",    "in_progress"},
                    }},
                }},
            });
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            events.push_back(json{
                {"event", "response.function_call_arguments.delta"},
                {"data", json{
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                }},
            });
        }
    }

    return events;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_final() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    }
    else {
        msg.role = "assistant";
        msg.content = content;
    }
    if (stop_kind == STOP_TYPE_WORD || stop_kind == STOP_TYPE_EOS) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }


    json choice{
        {"finish_reason", finish_reason},
        {"index", 0},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json{
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat_final();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({ "timings", timings.to_json() });
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop_kind == STOP_TYPE_WORD || stop_kind == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto& diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", 0},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"object", "chat.completion.chunk"},
            });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", 0},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"object",             "chat.completion.chunk"},
        });
    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
            });
    }
    if (timings.prompt_n >= 0) {
        deltas.back().push_back({ "timings", timings.to_json() });
    }
    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat_final();
    }

    return deltas;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_final() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    }
    else {
        msg.role = "assistant";
        msg.content = content;
    }

    std::vector<json> output;

    if (!msg.reasoning_content.empty()) {
        output.push_back(json{
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({json{
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (!msg.content.empty()) {
        output.push_back(json{
            {"content", json::array({json{
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     oai_resp_message_id},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const auto& tool_call : oaicompat_msg.tool_calls) {
        output.push_back(json{
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "fc_" + tool_call.id},
            {"name",      tool_call.name},
        });
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> events;
    std::vector<json> output;

    if (!oaicompat_msg.reasoning_content.empty()) {
        const json output_item = json{
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({json{
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        events.push_back(json{
            {"event", "response.output_item.done"},
            {"data", json{
                {"type", "response.output_item.done"},
                {"item", output_item},
            }},
        });
        output.push_back(output_item);
    }

    if (!oaicompat_msg.content.empty()) {
        events.push_back(json{
            {"event", "response.output_text.done"},
            {"data", json{
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"text",    oaicompat_msg.content},
            }},
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content},
        };

        events.push_back(json{
            {"event", "response.content_part.done"},
            {"data", json{
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"part",    content_part},
            }},
        });

        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"},
        };

        events.push_back(json{
            {"event", "response.output_item.done"},
            {"data", json{
                {"type", "response.output_item.done"},
                {"item", output_item},
            }},
        });
        output.push_back(output_item);
    }

    for (const auto& tool_call : oaicompat_msg.tool_calls) {
        const json output_item = {
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "fc_" + tool_call.id},
            {"name",      tool_call.name},
        };
        events.push_back(json{
            {"event", "response.output_item.done"},
            {"data", json{
                {"type", "response.output_item.done"},
                {"item", output_item},
            }},
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    events.push_back(json{
        {"event", "response.completed"},
        {"data", json{
            {"type", "response.completed"},
            {"response", json{
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }},
    });

    return events;
}

json server_task_result_cmpl_final::to_json_anthropic_final() {
    std::string stop_reason = "max_tokens";
    if (stop_kind == STOP_TYPE_WORD || stop_kind == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    }
    else {
        msg.role = "assistant";
        msg.content = content;
    }

    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
            });
    }

    for (const auto& tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        }
        catch (const std::exception&) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop_kind == STOP_TYPE_WORD || stop_kind == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    size_t thinking_block_index = 0;
    size_t text_block_index = anthropic_thinking_block_started ? 1 : 0;

    bool thinking_block_started = anthropic_thinking_block_started;
    bool text_block_started = anthropic_text_block_started;
    std::set<size_t> tool_calls_started;

    for (const auto& diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                    });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
                });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                    });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
                });
        }

        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (thinking_block_started ? 1 : 0) + (text_block_started ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto& full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                    });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                    });
            }
        }
    }

    if (thinking_block_started) {
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
            });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
            });
    }

    if (text_block_started) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
            });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (thinking_block_started ? 1 : 0) + (text_block_started ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
            });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
        });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
        });

    // extra fields for debugging purposes
    if (verbose && !events.empty()) {
        events.front()["data"]["__verbose"] = to_json_non_oaicompat_final();
    }
    // Don't add timings for Anthropic API (breaks spec compliance)
    if (oaicompat != OAICOMPAT_TYPE_ANTHROPIC && timings.prompt_n >= 0 && !events.empty()) {
        events.back()["data"]["timings"] = timings.to_json();
    }

    return events;
}

json server_task_result_cmpl_partial::to_json_anthropic_partial() {
    json events = json::array();
    bool first = n_decoded == 1;

    size_t thinking_block_index = 0;
    size_t text_block_index = anthropic_has_reasoning ? 1 : 0;

    bool thinking_started = anthropic_thinking_block_started;
    bool text_started = anthropic_text_block_started;

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
            });
    }

    for (const auto& diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                    });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
                });
        }

        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                    });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
                });
        }

        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                    });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                    });
            }
        }
    }

    if (verbose && !events.empty() && first) {
        events.front()["data"]["__verbose"] = to_json_non_oaicompat_partial();
    }

    if (timings.prompt_n >= 0 && !events.empty()) {
        events.back()["data"]["timings"] = timings.to_json();
    }

    //if (is_progress && !events.empty()) {
    //    events.back()["data"]["prompt_progress"] = progress.to_json();
    //}

    return events;
}


size_t server_prompt::size() const {
    size_t res = data.size();

    // PXA_CACHE_PARK_SPEC_v1: the speculative payloads are part of what this entry costs, so the
    // RAM limit has to see them. They are ~0 when the lever is off.
    res += data_draft.size();
    res += spec_carry.size();

    for (const auto& checkpoint : checkpoints) {
        res += checkpoint.size();
    }

    return res;
}

size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto& state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto& state : states) {
        res += state.n_tokens();
    }
    return res;

}

bool server_prompt_cache::load(server_prompt& prompt, const server_tokens& tokens_new, llama_context* ctx, int32_t id_slot, float min_reusable_fraction,
                               bool* out_restored) {
    if (out_restored != nullptr) {
        *out_restored = false;
    }
    thinking_tokens think_tokens;
    for (auto it = states.begin(); it != states.end(); ++it) {
        think_tokens = it->think_tokens;
        break;
    }
    server_tokens prompt_tokens;
    server_tokens tokens_new_ex;
    if (think_tokens.exclude) {
        prompt_tokens = prompt.tokens.get_tokens_exclude_think(ctx, think_tokens);
        tokens_new_ex = tokens_new.get_tokens_exclude_think(ctx, think_tokens);
    }
    else {
        // The caller's own prompt is being MEASURED here too, exactly like the candidates below,
        // and the caller reads `server_cached_prompt.tokens` back after this returns. Moving it out
        // emptied it whenever the scan found nothing better -- the normal case, since the scan only
        // promotes on sim_best < sim_cur and sim_best starts as this very conversation -- so the
        // slot came back with no token bookkeeping and re-prefilled the whole history.
        prompt_tokens = prompt.tokens.clone();
        tokens_new_ex = tokens_new.clone();
    }
    const auto lcp_best = prompt_tokens.get_common_prefix(ctx, tokens_new_ex);
    float f_keep_best = float(lcp_best.second) / prompt_tokens.size();
    float sim_best = prompt_tokens.get_tokens_similarity(ctx, tokens_new_ex, prompt.n_kept_prompt, prompt.n_discarded_prompt);
    LLAMA_LOG_INFO(" - looking for better prompt, base f_keep = %.3f, sim = %.3f, n_keep = %d, n_discarded_prompt = %d\n", f_keep_best, sim_best, prompt.n_kept_prompt, prompt.n_discarded_prompt);

    auto it_best = states.end();

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        server_tokens tokens;
        if (think_tokens.exclude) {
            tokens = it->tokens.get_tokens_exclude_think(ctx, think_tokens);
        }
        else {
            // This candidate is only being MEASURED here; it stays in the cache whether or not it
            // wins. Moving its token list out emptied every entry the scan touched, so the winner
            // was restored with no tokens and the caller re-prefilled the whole history anyway.
            tokens = it->tokens.clone();
        }
        const auto lcp_cur = tokens.get_common_prefix(ctx, tokens_new_ex);
        const float f_keep_cur = float(lcp_cur.first) / tokens.size();
        // do not recover a cached prompt whose reusable prefix is below cache-ram-similarity:
        // restoring a 60-130 MiB state for a tiny reusable prefix costs more than reprocessing.
        if (f_keep_cur < min_reusable_fraction) {
            continue;
        }
        const float sim_cur = tokens.get_tokens_similarity(ctx, tokens_new_ex, it->n_kept_prompt, it->n_discarded_prompt);
        if (sim_best < sim_cur) {
            f_keep_best = f_keep_cur;
            sim_best = sim_cur;
            it_best = it;
        }
    }

    if (it_best != states.end()) {
        LLAMA_LOG_INFO(" - found better prompt with f_keep = %.3f, sim = %.3f, n_keep = %d, n_discarded_prompt = %d\n", f_keep_best, sim_best, it_best->n_kept_prompt, it_best->n_discarded_prompt);
        const size_t size = it_best->data.size();
        // PXA_CKPT_APPLY_SYNC (PXA_KERNEL_FIX_20260611): drain in-flight backend work before the
        // bulk state write (see server-context.cpp apply_checkpoint note; sticky CUDA
        // illegal-access class under pipeline scheduling).
        llama_synchronize(ctx);
        const size_t n = llama_state_seq_set_data(ctx, it_best->data.data(), size, id_slot, 0);
        if (n != size) {
            LLAMA_LOG_INFO("failed to restore state with size %zu\n", size);
            return false;
        }

        it_best->data.clear();
        it_best->data.shrink_to_fit();

        prompt = std::move(*it_best);

        states.erase(it_best);

        if (out_restored != nullptr) {
            *out_restored = true;
        }
    }

    return true;
}

server_prompt* server_prompt_cache::alloc(const server_prompt& prompt, size_t state_size) {
    for (auto it = states.begin(); it != states.end();) {
        auto tokens_ctx_shift = prompt.tokens.clone();  // copy cache tokens
        tokens_ctx_shift.discard_n_tokens(prompt.n_kept_prompt, prompt.n_discarded_prompt);
        auto prefix = it->tokens.get_common_prefix(ctx, tokens_ctx_shift);
        const size_t len = prefix.first;
        const size_t len_prompt = prefix.second;
        // first check if the current state is contained fully in the cache
        if (len_prompt == tokens_ctx_shift.size()) {
            LLAMA_LOG_INFO("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
        // next, remove any cached prompts that are fully contained in the current prompt
        else if (len == it->tokens.size()) {
            LLAMA_LOG_INFO(" - removing obsolete cached prompt with length %d\n", (int)len);
            it = states.erase(it);
        }
        else {
            ++it;
        }
    }

    std::vector<uint8_t> state_data;

    // check if we can allocate enough memory for the new state
    try {
        state_data.resize(state_size);
    }
    catch (const std::bad_alloc& e) {
        LLAMA_LOG_INFO("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4 * size());

        LLAMA_LOG_INFO(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    // TODO: for some reason we can't copy server_tokens, so we have to do this workaround
    auto& cur = states.emplace_back();
    cur = {
        /*.tokens          =*/ prompt.tokens.clone(),
        /*.n_keep          =*/ prompt.n_kept_prompt,
        /*.n_discarded_prompt     =*/ prompt.n_discarded_prompt,
        /*.think_tokens                   =*/ prompt.think_tokens,
        /*.data            =*/ std::move(state_data),
        /*.checkpoints     =*/ prompt.checkpoints,
    };

    return &cur;
}


void server_prompt_cache::update() {
    if (limit_size > 0) {
        // always keep at least one state, regardless of the limits
        while (states.size() > 1 && size() > limit_size) {
            if (states.empty()) {
                break;
            }

            LLAMA_LOG_INFO(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", states.front().size() / (1024.0 * 1024.0));

            states.pop_front();
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size / size_per_token) : limit_tokens;

    LLAMA_LOG_INFO(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
        states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto& state : states) {
        LLAMA_LOG_INFO("   - prompt %p: %7d tokens, %7d discarded, checkpoints: %2zu, %9.3f MiB\n",
            (const void*)&state, state.n_tokens(), state.n_discarded_prompt, state.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
}

// ---- PXA_CACHE_DISK_v1 --------------------------------------------------------------------
//
// The RAM prompt cache is where a conversation waits while its slot is given to somebody else.
// Without a file behind it, it waits only until the process ends: every parked conversation loses
// its history on a restart and pays a full re-prefill on its next turn. These two functions write the
// whole cache to one file and read it back.
//
// Format (little-endian, host layout, single file):
//   u32 magic | u32 version | u64 fingerprint | u64 n_states
//   per state: u64 len + JSON header (tokens, n_kept_prompt, n_discarded_prompt, think tokens)
//              u64 n_data   + target sequence state bytes
//              u64 n_draft  + draft (companion) sequence state bytes   (0 when absent)
//              u64 n_carry  + drafter carry bytes                      (0 when absent)
//              u64 n_ckpt, then per checkpoint: u64 len + JSON meta, u64 len + bytes
//
// The fingerprint covers everything that decides whether these bytes can be installed at all
// (model, context geometry, KV types). A file that does not match is refused whole: a per-entry
// decision would have to trust a header written by the same mismatched build.
//
// Entries holding media chunks are skipped on save: their token list references chunk objects that
// do not survive the round trip, and a silently truncated one would restore onto the wrong cells.

static const uint32_t PXA_CACHE_DISK_MAGIC   = 0x43505850u; // "PXPC"
static const uint32_t PXA_CACHE_DISK_VERSION = 1u;

namespace {

void pxa_cache_write_raw(std::ofstream& f, const void* p, size_t n) {
    f.write(reinterpret_cast<const char*>(p), (std::streamsize)n);
}

void pxa_cache_write_u32(std::ofstream& f, uint32_t v) { pxa_cache_write_raw(f, &v, sizeof(v)); }
void pxa_cache_write_u64(std::ofstream& f, uint64_t v) { pxa_cache_write_raw(f, &v, sizeof(v)); }

void pxa_cache_write_blob(std::ofstream& f, const std::vector<uint8_t>& b) {
    pxa_cache_write_u64(f, (uint64_t)b.size());
    if (!b.empty()) {
        pxa_cache_write_raw(f, b.data(), b.size());
    }
}

void pxa_cache_write_str(std::ofstream& f, const std::string& s) {
    pxa_cache_write_u64(f, (uint64_t)s.size());
    if (!s.empty()) {
        pxa_cache_write_raw(f, s.data(), s.size());
    }
}

bool pxa_cache_read_raw(std::ifstream& f, void* p, size_t n) {
    f.read(reinterpret_cast<char*>(p), (std::streamsize)n);
    return (bool)f;
}

bool pxa_cache_read_u32(std::ifstream& f, uint32_t& v) { return pxa_cache_read_raw(f, &v, sizeof(v)); }
bool pxa_cache_read_u64(std::ifstream& f, uint64_t& v) { return pxa_cache_read_raw(f, &v, sizeof(v)); }

// A length read out of a file is attacker-shaped input even when the attacker is a truncated write:
// every one is checked against what is left in the file before anything is resized.
bool pxa_cache_read_blob(std::ifstream& f, uint64_t remaining, std::vector<uint8_t>& b) {
    uint64_t n = 0;
    if (!pxa_cache_read_u64(f, n) || n > remaining) {
        return false;
    }
    b.resize((size_t)n);
    return n == 0 || pxa_cache_read_raw(f, b.data(), (size_t)n);
}

bool pxa_cache_read_str(std::ifstream& f, uint64_t remaining, std::string& s) {
    uint64_t n = 0;
    if (!pxa_cache_read_u64(f, n) || n > remaining) {
        return false;
    }
    s.resize((size_t)n);
    return n == 0 || pxa_cache_read_raw(f, &s[0], (size_t)n);
}

} // namespace

size_t server_prompt_cache::save_file(const std::string& filepath) const {
    const std::string tmppath = filepath + ".tmp";

    std::ofstream f(tmppath, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) {
        LLAMA_LOG_ERROR("%s: cannot open %s for writing\n", __func__, tmppath.c_str());
        return 0;
    }

    size_t n_written = 0;

    std::vector<const server_prompt*> keep;
    for (const auto& state : states) {
        // The predicate is "does this entry hold media chunks", i.e. the media map -- NOT the
        // `has_mtmd` flag, which is only "this process was started with a projector" and is true
        // on every entry such a process makes. Testing the flag skipped the whole cache on any
        // projector build, wrote a header and nothing else, and said "media chunks" while doing it.
        if (state.tokens.has_mtmd_data()) {
            LLAMA_LOG_INFO("%s: skipping a cached prompt with media chunks (%d tokens)\n", __func__, state.n_tokens());
            continue;
        }
        if (state.data.empty()) {
            continue;
        }
        keep.push_back(&state);
    }

    pxa_cache_write_u32(f, PXA_CACHE_DISK_MAGIC);
    pxa_cache_write_u32(f, PXA_CACHE_DISK_VERSION);
    pxa_cache_write_u64(f, fingerprint);
    pxa_cache_write_u64(f, (uint64_t)keep.size());

    for (const auto* state : keep) {
        json hdr;
        hdr["tokens"]             = state->tokens.to_json();
        hdr["n_kept_prompt"]      = state->n_kept_prompt;
        hdr["n_discarded_prompt"] = state->n_discarded_prompt;
        hdr["think_exclude"]      = state->think_tokens.exclude;
        hdr["think_begin"]        = state->think_tokens.begin;
        hdr["think_end"]          = state->think_tokens.end;

        pxa_cache_write_str(f, hdr.dump());
        pxa_cache_write_blob(f, state->data);
        pxa_cache_write_blob(f, state->data_draft);
        pxa_cache_write_blob(f, state->spec_carry);

        pxa_cache_write_u64(f, (uint64_t)state->checkpoints.size());
        for (const auto& ckpt : state->checkpoints) {
            server_prompt_checkpoint copy = ckpt; // to_json() is non-const on the checkpoint
            pxa_cache_write_str(f, copy.to_json().dump());
            pxa_cache_write_blob(f, ckpt.data);
        }

        n_written += state->size();
    }

    f.flush();
    const bool ok = (bool)f;
    const size_t total = ok ? (size_t)f.tellp() : 0;
    f.close();

    if (!ok) {
        LLAMA_LOG_ERROR("%s: failed while writing %s\n", __func__, tmppath.c_str());
        ::remove(tmppath.c_str());
        return 0;
    }

    // Get the bytes onto the medium BEFORE the rename, otherwise a power loss just after it leaves
    // a short file under the real name and the next start reads a prefix of the entries.
    if (int fd = ::open(tmppath.c_str(), O_WRONLY); fd >= 0) {
        if (::fsync(fd) != 0) {
            LLAMA_LOG_ERROR("%s: could not flush %s to disk\n", __func__, tmppath.c_str());
        }
        ::close(fd);
    }

    // rename() over an existing path is itself atomic: a reader sees either the previous file or
    // the new one. Removing the old one first would only open a window in which neither exists.
    if (::rename(tmppath.c_str(), filepath.c_str()) != 0) {
        LLAMA_LOG_ERROR("%s: failed to move %s into place\n", __func__, tmppath.c_str());
        ::remove(tmppath.c_str());
        return 0;
    }

    // And make the rename itself durable.
    {
        const size_t slash = filepath.find_last_of('/');
        const std::string dir = slash == std::string::npos ? std::string(".") : filepath.substr(0, slash == 0 ? 1 : slash);
        if (int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY); dfd >= 0) {
            ::fsync(dfd);
            ::close(dfd);
        }
    }

    LLAMA_LOG_INFO("%s: wrote %zu parked prompt(s), %.3f MiB of state, %.3f MiB total to %s\n",
        __func__, keep.size(), n_written / (1024.0 * 1024.0), total / (1024.0 * 1024.0), filepath.c_str());

    return total;
}

size_t server_prompt_cache::load_file(const std::string& filepath, uint64_t fingerprint_expect, bool has_mtmd_now) {
    std::ifstream f(filepath, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        return 0;
    }

    const std::streampos file_size_pos = f.tellg();
    if (file_size_pos <= 0) {
        return 0;
    }
    const uint64_t file_size = (uint64_t)file_size_pos;
    f.seekg(0, std::ios::beg);

    uint32_t magic = 0, version = 0;
    uint64_t fp = 0, n_states = 0;
    if (!pxa_cache_read_u32(f, magic) || !pxa_cache_read_u32(f, version) ||
        !pxa_cache_read_u64(f, fp) || !pxa_cache_read_u64(f, n_states)) {
        LLAMA_LOG_ERROR("%s: %s is too short to be a prompt cache file\n", __func__, filepath.c_str());
        return 0;
    }
    if (magic != PXA_CACHE_DISK_MAGIC || version != PXA_CACHE_DISK_VERSION) {
        LLAMA_LOG_ERROR("%s: refusing %s: (magic, version) = (%08x, %08x)\n", __func__, filepath.c_str(), magic, version);
        return 0;
    }
    if (fp != fingerprint_expect) {
        LLAMA_LOG_ERROR("%s: refusing %s: written for a different model or context geometry\n", __func__, filepath.c_str());
        return 0;
    }

    size_t n_loaded = 0;
    size_t n_bytes  = 0;

    for (uint64_t i = 0; i < n_states; ++i) {
        const uint64_t remaining = file_size - (uint64_t)f.tellg();

        std::string hdr_str;
        server_prompt state;
        if (!pxa_cache_read_str(f, remaining, hdr_str)) {
            break;
        }

        json hdr;
        try {
            hdr = json::parse(hdr_str);
        } catch (const std::exception& e) {
            LLAMA_LOG_ERROR("%s: unreadable header in %s: %s\n", __func__, filepath.c_str(), e.what());
            break;
        }

        try {
            state.tokens.from_json(hdr.at("tokens"));
        } catch (const std::exception& e) {
            LLAMA_LOG_ERROR("%s: unreadable token list in %s: %s\n", __func__, filepath.c_str(), e.what());
            break;
        }
        // from_json installs the flag the WRITING process had. It is not a property of the bytes:
        // it says whether this process holds a projector, and server_tokens asserts on it both ways
        // (push_back of a media chunk requires it set, insert() requires it clear). Take ours.
        state.tokens.has_mtmd      = has_mtmd_now;
        state.n_kept_prompt        = hdr.value<int>("n_kept_prompt", 0);
        state.n_discarded_prompt   = hdr.value<int>("n_discarded_prompt", 0);
        state.think_tokens.exclude = hdr.value<bool>("think_exclude", true);
        state.think_tokens.begin   = hdr.value<std::string>("think_begin", "<think>");
        state.think_tokens.end     = hdr.value<std::string>("think_end", "</think>");

        if (!pxa_cache_read_blob(f, file_size - (uint64_t)f.tellg(), state.data) ||
            !pxa_cache_read_blob(f, file_size - (uint64_t)f.tellg(), state.data_draft) ||
            !pxa_cache_read_blob(f, file_size - (uint64_t)f.tellg(), state.spec_carry)) {
            break;
        }

        uint64_t n_ckpt = 0;
        if (!pxa_cache_read_u64(f, n_ckpt) || n_ckpt > file_size) {
            break;
        }
        bool ckpt_ok = true;
        for (uint64_t c = 0; c < n_ckpt; ++c) {
            std::string meta_str;
            server_prompt_checkpoint ckpt;
            if (!pxa_cache_read_str(f, file_size - (uint64_t)f.tellg(), meta_str)) {
                ckpt_ok = false;
                break;
            }
            try {
                ckpt.from_json(json::parse(meta_str));
            } catch (const std::exception&) {
                ckpt_ok = false;
                break;
            }
            if (!pxa_cache_read_blob(f, file_size - (uint64_t)f.tellg(), ckpt.data)) {
                ckpt_ok = false;
                break;
            }
            state.checkpoints.push_back(std::move(ckpt));
        }
        if (!ckpt_ok) {
            break;
        }

        n_bytes += state.size();
        states.push_back(std::move(state));
        ++n_loaded;
    }

    if (n_loaded != n_states) {
        LLAMA_LOG_ERROR("%s: %s ended early - kept the %zu parked prompt(s) that read cleanly\n",
            __func__, filepath.c_str(), n_loaded);
    }

    LLAMA_LOG_INFO("%s: restored %zu parked prompt(s), %.3f MiB from %s\n",
        __func__, n_loaded, n_bytes / (1024.0 * 1024.0), filepath.c_str());

    return n_loaded;
}
