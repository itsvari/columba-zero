#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <mat.h>
#include <opencv2/core.hpp>

namespace columba {

constexpr int colour_channels = 3;
constexpr int box_coordinates = 4;
constexpr int detection_columns = 6;
constexpr int padding_value = 114;
constexpr int maximum_detections = 300;
constexpr int maximum_nms_candidates = 30000;
constexpr float default_nms_iou = 0.7F;
constexpr float pixel_normalisation = 1.0F / 255.0F;

enum class Decode { yolo11, yolo26 };
enum class Quantisation { fp32, fp16, int8 };

struct ImageSize {
    int height_px;
    int width_px;
};

struct ModelSettings {
    ImageSize input;
    std::map<int, std::string> names;
    Decode decode = Decode::yolo11;
    float nms_iou = default_nms_iou;
    bool agnostic = false;
    bool rgb = true;
    int max_det = maximum_detections;
    Quantisation quantisation = Quantisation::fp32;
};

struct Letterbox {
    ImageSize original;
    ImageSize resized;
    ImageSize input;
    int left_px;
    int top_px;
    int right_px;
    int bottom_px;
    float gain_x;
    float gain_y;
};

struct Detection {
    float left_px;
    float top_px;
    float right_px;
    float bottom_px;
    float score;
    int class_id;
};

struct Timings {
    double preprocess_ms;
    double extract_ms;
    double postprocess_ms;
    double total_ms;
};

struct InferenceResult {
    std::vector<Detection> detections;
    Timings timings;
};

Letterbox letterbox_geometry(ImageSize original, ImageSize input);
cv::Mat letterbox_image(const cv::Mat& frame, const Letterbox& geometry);
ncnn::Mat preprocess(const cv::Mat& frame, const Letterbox& geometry, bool rgb = true);
float intersection_over_union(const Detection& left, const Detection& right);
std::vector<Detection> suppress(std::vector<Detection> candidates, const ModelSettings& settings);
std::vector<Detection> decode(
    const ncnn::Mat& output, const ModelSettings& settings, const Letterbox& geometry, float confidence
);

}
