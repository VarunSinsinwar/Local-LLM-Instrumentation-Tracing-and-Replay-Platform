// src/main.cpp
#include "ring_buffer.hpp"
#include "telemetry_hook.hpp"
#include "tui_app.hpp"
#include <thread>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./telemetry_platform <path_to_gguf_model>" << std::endl;
        return 1;
    }
    std::string model_path = argv[1];

    // Allocate ring buffer tracking up to the last 50 execution layers/tensors safely
    TelemetryRingBuffer global_buffer(50);

    // Run Backend Inference pipeline on a secondary Thread
    std::thread backend_thread([&]() {
        TelemetryHook::init_model_and_run(model_path, global_buffer);
    });

    // Start Foreground Terminal UI interaction on Main thread
    TuiApp::run(global_buffer);

    // Cleanup when user leaves UI
    backend_thread.detach(); 
    return 0;
}

