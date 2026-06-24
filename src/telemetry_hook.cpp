#include "telemetry_hook.hpp"
#include <iostream>
#include <cmath>
#include <thread>

bool ggml_debug_callback(struct ggml_tensor * t, bool is_input, void * user_data) {
    auto* ring_buf = static_cast<TelemetryRingBuffer*>(user_data);

    if (!is_input && t->name != nullptr) {
        LayerMetric metric;
        metric.name = t->name;
        metric.shape = "[" + std::to_string(t->ne[0]) + "," + std::to_string(t->ne[1]) + "]";
        metric.latency_ms = 0.45f; 

        if (t->type == GGML_TYPE_F32 && t->data != nullptr) {
            float* data = (float*)t->data;
            int64_t num_elements = ggml_nelements(t);
            int64_t zeros = 0;
            float min_v = 999.f, max_v = -999.f;

            for(int64_t i = 0; i < num_elements; ++i) {
                if (std::abs(data[i]) < 1e-5) zeros++;
                if (data[i] < min_v) min_v = data[i];
                if (data[i] > max_v) max_v = data[i];
            }
            metric.sparsity = (float)zeros / num_elements * 100.0f;
            metric.min_val = min_v;
            metric.max_val = max_v;
        } else {
            metric.sparsity = 0.0f; metric.min_val = 0.0f; metric.max_val = 0.0f;
        }

        ring_buf->push(metric);
    }
    return true;
}

void TelemetryHook::init_model_and_run(const std::string& model_path, TelemetryRingBuffer& ring_buffer) {
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    llama_model* model = llama_load_model_from_file(model_path.c_str(), model_params);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_threads = std::max(1U, std::thread::hardware_concurrency() - 2);
    llama_context* ctx = llama_new_context_with_model(model, ctx_params);

    std::cout << "[Backend] Model loaded. Simulating continuous inference execution..." << std::endl;

    while(true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(800));
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
}