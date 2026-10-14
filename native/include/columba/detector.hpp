#pragma once

#include <filesystem>
#include <optional>

#include <net.h>

#include "columba/detection.hpp"

namespace columba {

enum class Compute { fp16, fp32 };

class Detector final {
public:
    Detector(const std::filesystem::path& model_directory, ModelSettings settings, int threads,
             std::optional<Compute> compute = std::nullopt);
    InferenceResult detect(const std::uint8_t* pixels, ImageSize original, float confidence) const;
    const ModelSettings& settings() const noexcept;
    Compute compute() const noexcept;
    const ncnn::Option& options() const noexcept;

private:
    ModelSettings settings_;
    Compute compute_;
    ncnn::Net network_;
};

}
