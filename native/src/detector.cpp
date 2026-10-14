#include "columba/detector.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace columba {
namespace {

using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::duration<double, std::milli>;

}

Detector::Detector(const std::filesystem::path& model_directory, ModelSettings settings, int threads,
                   std::optional<Compute> compute)
    : settings_(std::move(settings)),
      compute_(compute.value_or(settings_.quantisation == Quantisation::fp32
                                   ? Compute::fp32 : Compute::fp16)) {
    if (threads <= 0) {
        throw std::invalid_argument("threads must be positive");
    }
    network_.opt.num_threads = threads;
    network_.opt.use_vulkan_compute = false;
    const bool use_fp16 = compute_ == Compute::fp16;
    network_.opt.use_fp16_packed = use_fp16;
    network_.opt.use_fp16_storage = use_fp16;
    network_.opt.use_fp16_arithmetic = use_fp16;
    if (network_.load_param((model_directory / "model.ncnn.param").string().c_str()) != 0) {
        throw std::runtime_error("cannot load model.ncnn.param");
    }
    if (network_.load_model((model_directory / "model.ncnn.bin").string().c_str()) != 0) {
        throw std::runtime_error("cannot load model.ncnn.bin");
    }
}

InferenceResult Detector::detect(const std::uint8_t* pixels, ImageSize original, float confidence) const {
    if (!std::isfinite(confidence) || confidence < 0.0F || confidence > 1.0F) {
        throw std::invalid_argument("confidence must be finite and between 0 and 1");
    }
    const auto started = Clock::now();
    const auto geometry = letterbox_geometry(original, settings_.input);
    const cv::Mat frame(original.height_px, original.width_px, CV_8UC3,
                        const_cast<std::uint8_t*>(pixels));
    const auto input = preprocess(frame, geometry, settings_.rgb);
    auto extractor = network_.create_extractor();
    if (extractor.input("in0", input) != 0) {
        throw std::runtime_error("cannot supply NCNN input blob in0");
    }
    ncnn::Mat output;
    const auto extract_started = Clock::now();
    if (extractor.extract("out0", output) != 0) {
        throw std::runtime_error("NCNN extraction of out0 failed");
    }
    const auto extract_finished = Clock::now();
    auto detections = decode(output, settings_, geometry, confidence);
    const auto finished = Clock::now();
    return {std::move(detections), {
        Milliseconds(extract_started - started).count(),
        Milliseconds(extract_finished - extract_started).count(),
        Milliseconds(finished - extract_finished).count(),
        Milliseconds(finished - started).count()
    }};
}

const ModelSettings& Detector::settings() const noexcept {
    return settings_;
}

Compute Detector::compute() const noexcept {
    return compute_;
}

const ncnn::Option& Detector::options() const noexcept {
    return network_.opt;
}

}
