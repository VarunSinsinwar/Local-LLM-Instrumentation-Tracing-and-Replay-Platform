#pragma once
#include <string>
#include <vector>
#include <mutex>

struct LayerMetric {
    std::string name;
    std::string shape;
    float latency_ms;
    float sparsity;
    float min_val;
    float max_val;
};

class TelemetryRingBuffer {
public:
    explicit TelemetryRingBuffer(size_t capacity);
    void push(const LayerMetric& metric);
    std::vector<LayerMetric> get_all_snapshot();

private:
    size_t capacity_;
    size_t head_ = 0;
    std::vector<LayerMetric> buffer_;
    std::mutex mtx_;
};

// src/ring_buffer.cpp
#include "ring_buffer.hpp"

TelemetryRingBuffer::TelemetryRingBuffer(size_t capacity) 
    : capacity_(capacity), buffer_() {}

void TelemetryRingBuffer::push(const LayerMetric& metric) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (buffer_.size() < capacity_) {
        buffer_.push_back(metric);
    } else {
        buffer_[head_] = metric;
        head_ = (head_ + 1) % capacity_;
    }
}

std::vector<LayerMetric> TelemetryRingBuffer::get_all_snapshot() {
    std::lock_guard<std::mutex> lock(mtx_);
    return buffer_; // Returns a safe copy to the TUI thread
}