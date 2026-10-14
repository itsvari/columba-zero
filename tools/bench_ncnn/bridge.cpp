#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <net.h>
#include <cpu.h>

namespace {
constexpr int colourChannels = 3;
constexpr double millisecondsPerSecond = 1000.0;
std::unique_ptr<ncnn::Net> network;
}

// pointers across the C boundary borrow Python buffers for this call only.
extern "C" int cz_load(const char* parameters, const char* weights, int precision, int threads) {
    try {
        auto candidate = std::make_unique<ncnn::Net>();
        candidate->opt.num_threads = threads;
        candidate->opt.use_vulkan_compute = false;
        candidate->opt.use_fp16_packed = precision == 16;
        candidate->opt.use_fp16_storage = precision == 16;
        candidate->opt.use_fp16_arithmetic = precision == 16;
        candidate->opt.use_bf16_storage = false;
        candidate->opt.use_int8_inference = precision == 8;
        if (candidate->load_param(parameters) != 0 || candidate->load_model(weights) != 0) {
            return -1;
        }
        network = std::move(candidate);
        return 0;
    } catch (...) {
        return -2;
    }
}

extern "C" void cz_close() {
    network.reset();
}

extern "C" int cz_features() {
    return ncnn::cpu_support_arm_asimdhp() + 2 * ncnn::cpu_support_arm_asimddp();
}

// capacity is measured in floats; output shape and elapsed time are borrowed scalars.
extern "C" int cz_infer(const float* input, int height, int width, float* output,
                        int capacity, int* rows, int* columns, double* elapsedMs) {
    if (!network || height <= 0 || width <= 0 || capacity <= 0) {
        return -1;
    }
    try {
        const auto started = std::chrono::steady_clock::now();
        ncnn::Mat image(width, height, colourChannels);
        if (image.empty()) {
            return -2;
        }
        const auto channelPixels = static_cast<std::size_t>(height) * width;
        for (int channel = 0; channel < colourChannels; ++channel) {
            std::memcpy(image.channel(channel), input + channel * channelPixels,
                        channelPixels * sizeof(float));
        }
        auto extractor = network->create_extractor();
        if (extractor.input(network->input_names().front(), image) != 0) {
            return -3;
        }
        ncnn::Mat prediction;
        if (extractor.extract(network->output_names().front(), prediction) != 0) {
            return -4;
        }
        *elapsedMs = std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                         .count() * millisecondsPerSecond;
        if (prediction.dims != 2 || prediction.elempack != 1 || prediction.elembits() != 32 ||
            static_cast<std::size_t>(prediction.w) * prediction.h > static_cast<std::size_t>(capacity)) {
            return -5;
        }
        *rows = prediction.h;
        *columns = prediction.w;
        std::copy_n(static_cast<const float*>(prediction.data), prediction.w * prediction.h, output);
        return 0;
    } catch (...) {
        return -6;
    }
}
