#include "columba/detection.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace columba {
namespace {

constexpr double rounding_bias = 0.1;
constexpr float half = 0.5F;
constexpr int first_class_row = box_coordinates;

int round_to_even(double value) {
    const double lower = std::floor(value);
    const double fraction = value - lower;
    if (fraction < 0.5 || (fraction == 0.5 && std::fmod(lower, 2.0) == 0.0)) {
        return static_cast<int>(lower);
    }
    return static_cast<int>(lower + 1.0);
}

Detection best_candidate(const ncnn::Mat& output, int anchor_index, int class_count) {
    int class_id = 0;
    float score = output.row(first_class_row)[anchor_index];
    for (int index = 1; index < class_count; ++index) {
        const float candidate_score = output.row(first_class_row + index)[anchor_index];
        if (candidate_score > score) {
            score = candidate_score;
            class_id = index;
        }
    }
    const float centre_x_px = output.row(0)[anchor_index];
    const float centre_y_px = output.row(1)[anchor_index];
    const float half_width_px = output.row(2)[anchor_index] * half;
    const float half_height_px = output.row(3)[anchor_index] * half;
    return {
        centre_x_px - half_width_px, centre_y_px - half_height_px,
        centre_x_px + half_width_px, centre_y_px + half_height_px, score, class_id
    };
}

Detection scale_detection(Detection detection, const Letterbox& geometry) {
    detection.left_px = std::clamp(
        (detection.left_px - geometry.left_px) / geometry.gain_x,
        0.0F, static_cast<float>(geometry.original.width_px)
    );
    detection.right_px = std::clamp(
        (detection.right_px - geometry.left_px) / geometry.gain_x,
        0.0F, static_cast<float>(geometry.original.width_px)
    );
    detection.top_px = std::clamp(
        (detection.top_px - geometry.top_px) / geometry.gain_y,
        0.0F, static_cast<float>(geometry.original.height_px)
    );
    detection.bottom_px = std::clamp(
        (detection.bottom_px - geometry.top_px) / geometry.gain_y,
        0.0F, static_cast<float>(geometry.original.height_px)
    );
    return detection;
}

}

Letterbox letterbox_geometry(ImageSize original, ImageSize input) {
    if (original.width_px <= 0 || original.height_px <= 0 ||
        input.width_px <= 0 || input.height_px <= 0) {
        throw std::invalid_argument("image dimensions must be positive");
    }
    const double gain = std::min(
        static_cast<double>(input.height_px) / original.height_px,
        static_cast<double>(input.width_px) / original.width_px
    );
    const ImageSize resized{
        round_to_even(original.height_px * gain), round_to_even(original.width_px * gain)
    };
    if (resized.height_px <= 0 || resized.width_px <= 0) {
        throw std::invalid_argument("letterboxed image dimension rounds to zero");
    }
    const double horizontal_padding_px = (input.width_px - resized.width_px) / 2.0;
    const double vertical_padding_px = (input.height_px - resized.height_px) / 2.0;
    return {
        original, resized, input,
        round_to_even(horizontal_padding_px - rounding_bias),
        round_to_even(vertical_padding_px - rounding_bias),
        round_to_even(horizontal_padding_px + rounding_bias),
        round_to_even(vertical_padding_px + rounding_bias),
        static_cast<float>(resized.width_px) / original.width_px,
        static_cast<float>(resized.height_px) / original.height_px
    };
}

cv::Mat letterbox_image(const cv::Mat& frame, const Letterbox& geometry) {
    cv::Mat resized;
    if (frame.cols == geometry.resized.width_px && frame.rows == geometry.resized.height_px) {
        resized = frame;
    } else {
        cv::resize(frame, resized, cv::Size(geometry.resized.width_px, geometry.resized.height_px),
                   0.0, 0.0, cv::INTER_LINEAR);
    }
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, geometry.top_px, geometry.bottom_px,
                      geometry.left_px, geometry.right_px, cv::BORDER_CONSTANT,
                      cv::Scalar::all(padding_value));
    return padded;
}

ncnn::Mat preprocess(const cv::Mat& frame, const Letterbox& geometry, bool rgb) {
    const cv::Mat padded = letterbox_image(frame, geometry);
    ncnn::Mat input = ncnn::Mat::from_pixels(
        padded.data, rgb ? ncnn::Mat::PIXEL_BGR2RGB : ncnn::Mat::PIXEL_BGR,
        padded.cols, padded.rows
    );
    const float normalisation[colour_channels] = {
        pixel_normalisation, pixel_normalisation, pixel_normalisation
    };
    input.substract_mean_normalize(nullptr, normalisation);
    return input;
}

float intersection_over_union(const Detection& left, const Detection& right) {
    const float intersection_width_px = std::max(
        0.0F, std::min(left.right_px, right.right_px) - std::max(left.left_px, right.left_px)
    );
    const float intersection_height_px = std::max(
        0.0F, std::min(left.bottom_px, right.bottom_px) - std::max(left.top_px, right.top_px)
    );
    const float intersection_area_px = intersection_width_px * intersection_height_px;
    const float left_area_px = (left.right_px - left.left_px) * (left.bottom_px - left.top_px);
    const float right_area_px = (right.right_px - right.left_px) * (right.bottom_px - right.top_px);
    const float union_area_px = left_area_px + right_area_px - intersection_area_px;
    return union_area_px > 0.0F ? intersection_area_px / union_area_px : 0.0F;
}

std::vector<Detection> suppress(std::vector<Detection> candidates, const ModelSettings& settings) {
    std::stable_sort(candidates.begin(), candidates.end(), [](const Detection& left, const Detection& right) {
        return left.score > right.score;
    });
    if (settings.decode == Decode::yolo26) {
        candidates.resize(std::min(candidates.size(), static_cast<std::size_t>(settings.max_det)));
        return candidates;
    }
    candidates.resize(std::min(candidates.size(), static_cast<std::size_t>(maximum_nms_candidates)));
    std::vector<Detection> selected;
    selected.reserve(std::min(candidates.size(), static_cast<std::size_t>(settings.max_det)));
    for (const auto& candidate : candidates) {
        const bool overlaps = std::any_of(selected.begin(), selected.end(), [&](const Detection& kept) {
            return (settings.agnostic || kept.class_id == candidate.class_id) &&
                intersection_over_union(kept, candidate) > settings.nms_iou;
        });
        if (overlaps) {
            continue;
        }
        selected.push_back(candidate);
        if (selected.size() == static_cast<std::size_t>(settings.max_det)) {
            break;
        }
    }
    return selected;
}

std::vector<Detection> decode(
    const ncnn::Mat& output, const ModelSettings& settings, const Letterbox& geometry, float confidence
) {
    const int class_count = static_cast<int>(settings.names.size());
    if (output.dims != 2 || output.h != box_coordinates + class_count || output.w <= 0 ||
        output.elempack != 1 || output.elemsize != sizeof(float)) {
        throw std::runtime_error("expected out0 float32 tensor (4 + nc, N)");
    }
    std::vector<Detection> candidates;
    for (int anchor_index = 0; anchor_index < output.w; ++anchor_index) {
        auto candidate = best_candidate(output, anchor_index, class_count);
        if (settings.decode == Decode::yolo26 && !settings.agnostic) {
            for (int class_id = 0; class_id < class_count; ++class_id) {
                candidate.score = output.row(first_class_row + class_id)[anchor_index];
                candidate.class_id = class_id;
                if (candidate.score > confidence) {
                    candidates.push_back(candidate);
                }
            }
            continue;
        }
        if (candidate.score > confidence) {
            candidates.push_back(candidate);
        }
    }
    auto selection_settings = settings;
    if (settings.decode == Decode::yolo26) {
        selection_settings.max_det = std::min(settings.max_det, output.w);
    }
    auto selected = suppress(std::move(candidates), selection_settings);
    for (auto& detection : selected) {
        detection = scale_detection(detection, geometry);
    }
    return selected;
}

}
