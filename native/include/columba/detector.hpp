#pragma once

#include <filesystem>

#include <net.h>

#include "columba/detection.hpp"

namespace columba {

class Detector final {
public:
    Detector(const std::filesystem::path& model_directory, ModelSettings settings, int threads);
    InferenceResult detect(const std::uint8_t* pixels, ImageSize original, float confidence) const;
    const ModelSettings& settings() const noexcept;

private:
    ModelSettings settings_;
    ncnn::Net network_;
};

}
