cmake_minimum_required(VERSION 3.24)
project(LLMTelemetryPlatform LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Fetch FTXUI (TUI Library)
include(FetchContent)
FetchContent_Declare(
    ftxui
    GIT_REPOSITORY https://github.com/ArthurSonson/FTXUI.git
    GIT_TAG v5.0.0
)
FetchContent_MakeAvailable(ftxui)

# 2. Add llama.cpp (Assumes it's cloned inside third_party/llama.cpp)
# Disabling GPU features explicitly for your CPU-only environment
set(LLAMA_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/llama.cpp)

# 3. Project Executable Target
add_executable(telemetry_platform
    src/main.cpp
    src/ring_buffer.cpp
    src/telemetry_hook.cpp
    src/tui_app.cpp
)

# 4. Target Include Directories
target_include_directories(telemetry_platform PRIVATE 
    include
    third_party/llama.cpp/include
    third_party/llama.cpp/ggml/include
)

# 5. Link Dependencies (llama is the core inference library from llama.cpp)
target_link_libraries(telemetry_platform PRIVATE 
    llama
    ftxui::screen
    ftxui::dom
    ftxui::component
    threads
)