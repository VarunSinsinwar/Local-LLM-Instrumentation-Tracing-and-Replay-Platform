#include "tui_app.hpp"
#include "ftxui/component/component.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"

void TuiApp::run(TelemetryRingBuffer& ring_buffer) {
    using namespace ftxui;
    auto screen = ScreenInteractive::TerminalOutput();
    int selected_layer = 0;

    auto renderer = Renderer([& selected_layer, &ring_buffer] {
        auto metrics_snapshot = ring_buffer.get_all_snapshot();

        Elements layer_elements;
        for (size_t i = 0; i < metrics_snapshot.size(); ++i) {
            if (i == selected_layer) {
                layer_elements.push_back(text("-> [ " + metrics_snapshot[i].name + " ]") | bold | color(Color::Green));
            } else {
                layer_elements.push_back(text("   " + metrics_snapshot[i].name));
            }
        }

        if(layer_elements.empty()) {
            layer_elements.push_back(text("Awaiting telemetry events from backend loop..."));
        }

        Component metrics_view = Filler();
        if (!metrics_snapshot.empty() && selected_layer < metrics_snapshot.size()) {
            auto current_layer = metrics_snapshot[selected_layer];
            metrics_view = vbox({
                text("Layer Name: " + current_layer.name) | bold,
                text("Shape     : " + current_layer.shape),
                text("Latency   : " + std::to_string(current_layer.latency_ms) + " ms"),
                text("Sparsity  : " + std::to_string(current_layer.sparsity) + "%"),
                text("Min / Max : " + std::to_string(current_layer.min_val) + " / " + std::to_string(current_layer.max_val))
            });
        }

        return hbox({
            vbox(std::move(layer_elements)) | borderDouble | size(WIDTH, EQUAL, 40),
            vbox({
                text("Telemetry Dashboard (j/k: Navigate)") | bold | center,
                separator(),
                metrics_view
            }) | border | flex
        });
    });

    auto input_handler = CatchEvent(renderer, [&](Event event) {
        if (event == Event::Character('j')) {
            selected_layer++; return true;
        }
        if (event == Event::Character('k')) {
            if (selected_layer > 0) selected_layer--; return true;
        }
        if (event == Event::Escape) {
            screen.ExitLoopClosure()(); return true;
        }
        return false;
    });

    std::thread refresh_thread([&screen]() {
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            screen.PostEvent(Event::Custom);
        }
    });
    refresh_thread.detach();

    screen.Loop(input_handler);
}