#include "server-task.h"
#include "server-queue.h"
#include "server-common.h"

#include "log.h"
#include <chrono>

#define QUE_INF(fmt, ...) LOG_INF("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_WRN(fmt, ...) LOG_WRN("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_ERR(fmt, ...) LOG_ERR("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_DBG(fmt, ...) LOG_DBG("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

#define RES_INF(fmt, ...) LOG_INF("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_WRN(fmt, ...) LOG_WRN("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_ERR(fmt, ...) LOG_ERR("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_DBG(fmt, ...) LOG_DBG("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)


int server_queue::post(server_task task) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    if (task.id == -1) {
        task.id = id++;
        //LOG_VERBOSE("new task id", { {"new_id", task.id} });
        QUE_DBG("new task, id = %d\n", task.id);
    }
    // PXA_CANCEL_DEFERRED_v1 (PXA_KERNEL_FIX_20260611): the vector post() already drops a
    // cancelled task from queue_tasks/queue_tasks_deferred, but the single post() (used by
    // server_context::request_cancel) did not -- a CANCEL for a task still waiting in the
    // deferred queue left a GHOST task that later occupied a slot generating for a dead client
    // (extra head-of-line blocking under pile-up). Mirror the cleanup here.
    if (task.type == SERVER_TASK_TYPE_CANCEL) {
        cleanup_pending_task(task.id_target);
    }
    queue_tasks.push_back(std::move(task));
    condition_tasks.notify_one();
    return task.id;
}

void server_queue::cleanup_pending_task(int id_target) {
    // no need lock because this is called exclusively by post()
    auto rm_func = [id_target](const server_task& task) {
        return task.id == id_target;
    };
    queue_tasks.erase(
        std::remove_if(queue_tasks.begin(), queue_tasks.end(), rm_func),
        queue_tasks.end());
    queue_tasks_deferred.erase(
        std::remove_if(queue_tasks_deferred.begin(), queue_tasks_deferred.end(), rm_func),
        queue_tasks_deferred.end());
}

// multi-task version of post()
int server_queue::post(std::vector<server_task>&& tasks, bool front) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    for (auto& task : tasks) {
        if (task.id == -1) {
            task.id = id++;
        }
        // if this is cancel task make sure to clean up pending tasks
        if (task.type == SERVER_TASK_TYPE_CANCEL) {
            cleanup_pending_task(task.id_target);
        }
        QUE_DBG("new task, id = %d/%d, front = %d\n", task.id, (int)tasks.size(), front);
        if (front) {
            queue_tasks.push_front(std::move(task));
        }
        else {
            queue_tasks.push_back(std::move(task));
        }
    }
    condition_tasks.notify_one();
    return 0;
}

void server_queue::defer(server_task&& task) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    queue_tasks_deferred.push_back(std::move(task));
}

int server_queue::get_new_id() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    int new_id = id++;
    //LOG_VERBOSE("new task id", { {"new_id", new_id} });
    QUE_DBG("new task, id = %d\n", id);
    return new_id;
}

void server_queue::notify_slot_changed() {
    // move deferred tasks back to main loop
    std::unique_lock<std::mutex> lock(mutex_tasks);
    for (auto& task : queue_tasks_deferred) {
        queue_tasks.push_back(std::move(task));
    }
    queue_tasks_deferred.clear();
}

void server_queue::on_new_task(std::function<void(server_task&&)> callback) {
    callback_new_task = std::move(callback);
}


void server_queue::start_loop() {
    running = true;

    while (true) {
        LOG_VERBOSE("new task may arrive", {});

        while (true) {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            if (queue_tasks.empty()) {
                lock.unlock();
                break;
            }
            server_task task = std::move(queue_tasks.front());
            queue_tasks.pop_front();
            lock.unlock();
            //LOG_VERBOSE("callback_new_task", { {"id_task", task.id} });
            {
                static const bool pxa_qdbg = getenv("PXA_DECODE_WALL_DBG") != nullptr;
                if (pxa_qdbg) {
                    const int64_t t0 = ggml_time_us();
                    const int tt = (int) task.type;
                    callback_new_task(std::move(task));
                    const int64_t el = ggml_time_us() - t0;
                    if (el > 2000) fprintf(stderr, "PXA_Q task_type=%d took=%lldus\n", tt, (long long) el);
                } else {
                    callback_new_task(std::move(task));
                }
            }
        }

        LOG_VERBOSE("update_multitasks", {});

        // check if we have any finished multitasks
        auto queue_iterator = queue_multitasks.begin();
        while (queue_iterator != queue_multitasks.end()) {
            if (queue_iterator->subtasks_remaining.empty()) {
                // all subtasks done == multitask is done
                server_task_multi current_multitask = *queue_iterator;
                callback_finish_multitask(current_multitask);
                // remove this multitask
                queue_iterator = queue_multitasks.erase(queue_iterator);
            }
            else {
                ++queue_iterator;
            }
        }

        // all tasks in the current loop is processed, slots data is now ready
        LOG_VERBOSE("callback_update_slots", {});

        {
            static const bool pxa_qdbg2 = getenv("PXA_DECODE_WALL_DBG") != nullptr;
            static int64_t pxa_last_us_done = 0;
            if (pxa_qdbg2) {
                const int64_t t0 = ggml_time_us();
                if (pxa_last_us_done) {
                    const int64_t between = t0 - pxa_last_us_done;
                    if (between > 2000) fprintf(stderr, "PXA_Q between_update_slots=%lldus\n", (long long) between);
                }
                callback_update_slots();
                pxa_last_us_done = ggml_time_us();
            } else {
                callback_update_slots();
            }
        }

        LOG_VERBOSE("wait for new task", {});
        {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            if (queue_tasks.empty()) {
                if (!running) {
                    LOG_VERBOSE("ending start_loop", {});
                    return;
                }
                if (!queue_tasks_deferred.empty()) {
                    // PXA_DEFER_PUMP_v1: while tasks are parked in the deferred queue,
                    // heartbeat instead of sleeping unconditionally, and re-dispatch the
                    // moment a slot can take work. notify_slot_changed() alone cannot revive
                    // tasks that were deferred while slots were already idle (no future
                    // release -> no notify -> permanent starvation: completion tasks parked
                    // forever while /health stayed green, reproduced under a sustained
                    // multi-slot soak — an HTTP-hang signature).
                    condition_tasks.wait_for(lock, std::chrono::milliseconds(500), [&] {
                        return (!queue_tasks.empty() || !running);
                        });
                    if (queue_tasks.empty() && !queue_tasks_deferred.empty() &&
                        callback_slot_available && callback_slot_available()) {
                        LOG_WARNING("deferred-task pump: re-dispatching parked tasks", {
                            {"n_deferred", (int) queue_tasks_deferred.size()},
                            });
                        for (auto & task : queue_tasks_deferred) {
                            queue_tasks.push_back(std::move(task));
                        }
                        queue_tasks_deferred.clear();
                    }
                } else {
                    condition_tasks.wait(lock, [&] {
                        return (!queue_tasks.empty() || !running);
                        });
                }
            }
        }
    }
}


void server_queue::add_multitask(int id_multi, std::vector<int>& sub_ids) {
    std::lock_guard<std::mutex> lock(mutex_tasks);
    server_task_multi multi;
    multi.id = id_multi;
    std::copy(sub_ids.begin(), sub_ids.end(), std::inserter(multi.subtasks_remaining, multi.subtasks_remaining.end()));
    queue_multitasks.push_back(multi);
}


void server_queue::update_multitask(int id_multi, int id_sub, server_task_result& result) {
    std::lock_guard<std::mutex> lock(mutex_tasks);
    for (auto& multitask : queue_multitasks) {
        if (multitask.id == id_multi) {
            multitask.subtasks_remaining.erase(id_sub);
            multitask.results.push_back(result);
        }
    }
}


void server_response::add_waiting_task_id(int id_task) {
    SRV_DBG("add task %d to waiting list. current waiting = %d (before add)\n", id_task, (int)waiting_task_ids.size());

    std::unique_lock<std::mutex> lock(mutex_results);
    waiting_task_ids.insert(id_task);
}

void server_response::add_waiting_tasks(const std::vector<server_task>& tasks) {
    std::unique_lock<std::mutex> lock(mutex_results);

    for (const auto& task : tasks) {
        SRV_DBG("add task %d to waiting list. current waiting = %d (before add)\n", task.id, (int)waiting_task_ids.size());
        waiting_task_ids.insert(task.id);
    }
}

void server_response::remove_waiting_task_id(int id_task) {
    //LOG_VERBOSE("remove waiting for task id", { {"id_task", id_task} });
    QUE_DBG("remove waiting for task id, id = %d\n", id_task);
    std::unique_lock<std::mutex> lock(mutex_results);
    waiting_task_ids.erase(id_task);
}


server_task_result server_response::recv(int id_task) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_results);
        condition_results.wait(lock, [&] {
            return !queue_results_legacy.empty();
            });

        for (int i = 0; i < (int)queue_results_legacy.size(); i++) {
            if (queue_results_legacy[i].id == id_task) {
                assert(queue_results_legacy[i].id_multi == -1);
                server_task_result res = queue_results_legacy[i];
                queue_results_legacy.erase(queue_results_legacy.begin() + i);
                return res;
            }
        }
    }

    // should never reach here
}

// same as recv(), but have timeout in seconds
// if timeout is reached, nullptr is returned
server_task_result_ptr server_response::recv_with_timeout(const std::unordered_set<int>& id_tasks, int timeout) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_results);

        for (int i = 0; i < (int)queue_results.size(); i++) {
            if (id_tasks.find(queue_results[i]->id) != id_tasks.end()) {
                server_task_result_ptr res = std::move(queue_results[i]);
                queue_results.erase(queue_results.begin() + i);
                return res;
            }
        }

        std::cv_status cr_res = condition_results.wait_for(lock, std::chrono::seconds(timeout));
        if (!running) {
            SRV_DBG("%s : queue result stop\n", __func__);
            std::terminate(); // we cannot return here since the caller is HTTP code
        }
        if (cr_res == std::cv_status::timeout) {
            return nullptr;
        }
    }

    // should never reach here
}
void server_response::remove_waiting_task_ids(const std::unordered_set<int>& id_tasks) {
    std::unique_lock<std::mutex> lock(mutex_results);

    for (const auto& id_task : id_tasks) {
        SRV_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int)waiting_task_ids.size());
        waiting_task_ids.erase(id_task);
    }
}

void server_response::send(server_task_result result) {
    //LOG_VERBOSE("send new result", { {"id_task", result.id} });
    QUE_DBG("send new result, id = %d\n", result.id);
    std::unique_lock<std::mutex> lock(mutex_results);
    for (const auto& id_task : waiting_task_ids) {
        // LOG_TEE("waiting task id %i \n", id_task);
        // for now, tasks that have associated parent multitasks just get erased once multitask picks up the result
        if (result.id_multi == id_task) {
            //LOG_VERBOSE("callback_update_multitask", { {"id_task", id_task} });
            QUE_DBG("callback_update_multitask, id = %d\n", id_task);
            callback_update_multitask(id_task, result.id, result);
            continue;
        }

        if (result.id == id_task) {
            //LOG_VERBOSE("queue_results_legacy.push_back", { {"id_task", id_task} });
            QUE_DBG("queue_results.push_back, id = %d\n", id_task);
            queue_results_legacy.push_back(std::move(result));
            condition_results.notify_all();
            return;
        }
    }
}

// Send a new result to a waiting id_task
void server_response::send(server_task_result_ptr&& result) {
    SRV_DBG("sending result for task id = %d\n", result->id);

    std::unique_lock<std::mutex> lock(mutex_results);
    for (const auto& id_task : waiting_task_ids) {
        if (result->id == id_task) {
            SRV_DBG("task id = %d pushed to result queue\n", result->id);

            queue_results.emplace_back(std::move(result));
            condition_results.notify_all();
            return;
        }
    }
}
