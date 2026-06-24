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
    return buffer_;
}