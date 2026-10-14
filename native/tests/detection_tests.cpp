#include "columba/detection.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(float actual, float expected, const std::string& message) {
    constexpr float tolerance = 0.0001F;
    require(std::abs(actual - expected) < tolerance, message);
}

void test_letterbox() {
    const auto square = columba::letterbox_geometry({480, 640}, {640, 640});
    require(square.top_px == 80 && square.bottom_px == 80 && square.left_px == 0, "centred padding");
    const auto odd = columba::letterbox_geometry({321, 640}, {384, 640});
    require(odd.top_px == 31 && odd.bottom_px == 32, "odd padding rounds asymmetrically");
    const auto ties = columba::letterbox_geometry({10, 13}, {5, 10});
    require(ties.resized.width_px == 6, "resize uses Python ties-to-even rounding");
    const auto rounded = columba::letterbox_geometry({333, 517}, {384, 640});
    near(rounded.gain_y, 384.0F / 333.0F, "rounded y gain");
    near(rounded.gain_x, 596.0F / 517.0F, "rounded x gain");
    const cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(10, 20, 30));
    const auto input = columba::preprocess(frame, square);
    near(input.channel(0).row(80)[0], 30.0F / 255.0F, "RGB channel order");
    near(input.channel(2).row(80)[0], 10.0F / 255.0F, "BGR input");
    near(input.channel(0).row(0)[0], 114.0F / 255.0F, "normalised padding");
}

void test_nms() {
    const std::vector<columba::Detection> candidates{
        {0, 0, 10, 10, 0.8F, 0}, {0, 0, 10, 10, 0.9F, 0},
        {0, 0, 10, 10, 0.7F, 1}, {20, 20, 30, 30, 0.6F, 0}
    };
    columba::ModelSettings settings{{640, 640}, {{0, "pigeon"}, {1, "other_bird"}}};
    const auto selected = columba::suppress(candidates, settings);
    require(selected.size() == 3 && selected[0].score == 0.9F, "per-class score-ordered NMS");
    settings.agnostic = true;
    require(columba::suppress(candidates, settings).size() == 2, "agnostic NMS");
    settings.decode = columba::Decode::yolo26;
    settings.max_det = 2;
    const auto top = columba::suppress(candidates, settings);
    require(top.size() == 2 && top[1].score == 0.8F, "top-k retains duplicate boxes without NMS");
    near(columba::intersection_over_union(candidates[0], candidates[1]), 1.0F, "identical IoU");
    near(columba::intersection_over_union(candidates[0], candidates[3]), 0.0F, "disjoint IoU");
    near(columba::intersection_over_union({0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 1, 0}), 0, "zero area");
    settings.decode = columba::Decode::yolo11;
    settings.agnostic = false;
    settings.max_det = 1;
    require(columba::suppress(candidates, settings).size() == 1, "max_det truncates NMS output");
}

void test_decode() {
    constexpr int anchors = 4;
    constexpr int rows = 6;
    ncnn::Mat output(anchors, rows);
    const float values[rows][anchors] = {
        {320, 320, 1000, 5}, {320, 320, 1000, 5}, {200, 200, 40, 20},
        {100, 100, 40, 20}, {0.9F, 0.8F, 0.1F, 0.5F}, {0.1F, 0.2F, 0.7F, 0.1F}
    };
    for (int row = 0; row < rows; ++row) {
        for (int anchor = 0; anchor < anchors; ++anchor) {
            output.row(row)[anchor] = values[row][anchor];
        }
    }
    columba::ModelSettings settings{{640, 640}, {{0, "pigeon"}, {1, "other_bird"}}};
    const auto geometry = columba::letterbox_geometry({480, 640}, settings.input);
    const auto decoded = columba::decode(output, settings, geometry, 0.5F);
    require(decoded.size() == 2, "best class, strict confidence threshold, and NMS");
    near(decoded[0].left_px, 220, "cxcywh to xyxy");
    near(decoded[0].top_px, 190, "subtract letterbox padding");
    require(decoded[1].class_id == 1, "class ID preserved");
    near(decoded[1].right_px, 640, "clip x to frame");
    near(decoded[1].bottom_px, 480, "clip y to frame");
    require(columba::decode(output, settings, geometry, 1.0F).empty(), "empty output");
    settings.decode = columba::Decode::yolo26;
    require(columba::decode(output, settings, geometry, 0.5F).size() == 3, "one-to-one head skips NMS");
    output.row(5)[0] = 0.85F;
    const auto multiple_classes = columba::decode(output, settings, geometry, 0.5F);
    require(multiple_classes.size() == 4, "class-aware top-k may keep two classes at one anchor");
    settings.agnostic = true;
    require(columba::decode(output, settings, geometry, 0.5F).size() == 3,
            "agnostic top-k selects the best class at each anchor");
    bool rejected = false;
    try {
        columba::decode(ncnn::Mat(anchors, rows + 1), settings, geometry, 0.5F);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "unexpected tensor layout rejected");
}

}

int main() {
    try {
        test_letterbox();
        test_nms();
        test_decode();
        std::cout << "letterbox, decode, NMS and top-k passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
