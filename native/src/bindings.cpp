#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/pair.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "columba/detector.hpp"

namespace nb = nanobind;
namespace {

using Frame = nb::ndarray<nb::numpy, const std::uint8_t, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
using Predictions = nb::ndarray<nb::numpy, float, nb::ndim<2>, nb::c_contig>;

nb::dict mapping(nb::handle value, const char* description) {
    if (!nb::isinstance<nb::dict>(value)) {
        throw std::invalid_argument(std::string(description) + " must be a mapping");
    }
    return nb::borrow<nb::dict>(value);
}

nb::object field(const nb::dict& values, const char* key, nb::object fallback = nb::none()) {
    return values.contains(key) ? nb::borrow<nb::object>(values[key]) : std::move(fallback);
}

int integer(nb::handle value, const char* description) {
    if (!nb::isinstance<nb::int_>(value) || nb::isinstance<nb::bool_>(value)) {
        throw std::invalid_argument(std::string(description) + " must be an integer");
    }
    return nb::cast<int>(value);
}

bool boolean(nb::handle value, const char* description) {
    if (!nb::isinstance<nb::bool_>(value)) {
        throw std::invalid_argument(std::string(description) + " must be a boolean");
    }
    return nb::cast<bool>(value);
}

std::string string(nb::handle value, const char* description) {
    if (!nb::isinstance<nb::str>(value)) {
        throw std::invalid_argument(std::string(description) + " must be a string");
    }
    return nb::cast<std::string>(value);
}

columba::ImageSize image_size(nb::handle value) {
    if ((!nb::isinstance<nb::list>(value) && !nb::isinstance<nb::tuple>(value)) || nb::len(value) != 2) {
        throw std::invalid_argument("input_hw/imgsz must contain [height, width]");
    }
    const auto sequence = nb::borrow<nb::sequence>(value);
    const int height_px = integer(sequence[0], "input height");
    const int width_px = integer(sequence[1], "input width");
    constexpr int stride_px = 32;
    if (height_px <= 0 || width_px <= 0 || height_px % stride_px != 0 || width_px % stride_px != 0) {
        throw std::invalid_argument("input dimensions must be positive multiples of 32");
    }
    return {height_px, width_px};
}

std::map<int, std::string> class_names(nb::handle value) {
    std::map<int, std::string> names;
    if (nb::isinstance<nb::dict>(value)) {
        for (const auto& item : nb::borrow<nb::dict>(value)) {
            const int index = nb::isinstance<nb::str>(item.first)
                ? nb::cast<int>(nb::module_::import_("builtins").attr("int")(item.first))
                : integer(item.first, "class index");
            if (!names.emplace(index, string(item.second, "class name")).second) {
                throw std::invalid_argument("duplicate class index");
            }
        }
    } else if (nb::isinstance<nb::list>(value) || nb::isinstance<nb::tuple>(value)) {
        int index = 0;
        for (nb::handle name : nb::borrow<nb::iterable>(value)) {
            names.emplace(index++, string(name, "class name"));
        }
    } else {
        throw std::invalid_argument("names must be a mapping or sequence");
    }
    if (names.empty()) {
        throw std::invalid_argument("names must not be empty");
    }
    int expected_index = 0;
    for (const auto& item : names) {
        if (item.first != expected_index++ || item.second.empty()) {
            throw std::invalid_argument("class indices must be contiguous from zero, with non-empty names");
        }
    }
    return names;
}

void validate_preprocessing(const nb::dict& settings) {
    if (!boolean(field(settings, "letterbox", nb::bool_(true)), "letterbox") ||
        integer(field(settings, "pad_value", nb::int_(columba::padding_value)), "pad_value") !=
        columba::padding_value) {
        throw std::invalid_argument("native preprocessing requires letterbox=true and pad_value=114");
    }
    if (settings.contains("normalize")) {
        const auto normalisation = mapping(settings["normalize"], "normalize");
        const auto yaml = nb::module_::import_("yaml");
        const auto expected = yaml.attr("safe_load")("{mean: [0, 0, 0], std: [255, 255, 255]}");
        if (!nb::cast<bool>(normalisation.attr("__eq__")(expected))) {
            throw std::invalid_argument("native normalisation requires mean [0,0,0] and std [255,255,255]");
        }
    }
    if (settings.contains("output_layout") && string(settings["output_layout"], "output_layout") !=
        "1x(4+nc)xN cxcywh-px sigmoid-scores") {
        throw std::invalid_argument("unsupported output_layout");
    }
}

columba::ModelSettings read_settings(const std::filesystem::path& directory) {
    if (!std::filesystem::is_directory(directory)) {
        throw std::invalid_argument("model must be a complete NCNN directory");
    }
    for (const char* name : {"model.ncnn.param", "model.ncnn.bin", "metadata.yaml"}) {
        const auto path = directory / name;
        if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) == 0) {
            throw std::invalid_argument("incomplete NCNN export: missing or empty " + path.string());
        }
    }
    std::ifstream metadata_file(directory / "metadata.yaml");
    std::ostringstream contents;
    contents << metadata_file.rdbuf();
    const auto metadata = mapping(nb::module_::import_("yaml").attr("safe_load")(contents.str()), "metadata");
    if (string(field(metadata, "task"), "task") != "detect") {
        throw std::invalid_argument("metadata must describe a detection model");
    }
    columba::ModelSettings settings{image_size(field(metadata, "imgsz")), class_names(field(metadata, "names"))};
    if (!metadata.contains("columba")) {
        constexpr int legacy_input_px = 640;
        if (settings.input.height_px != legacy_input_px || settings.input.width_px != legacy_input_px) {
            throw std::invalid_argument("legacy NCNN export without columba metadata must be 640x640");
        }
        return settings;
    }
    const auto columba_metadata = mapping(metadata["columba"], "columba");
    const auto input = image_size(field(columba_metadata, "input_hw"));
    if (input.height_px != settings.input.height_px || input.width_px != settings.input.width_px) {
        throw std::invalid_argument("columba.input_hw differs from export imgsz");
    }
    const auto decode = string(field(columba_metadata, "decode"), "decode");
    if (decode != "yolo11" && decode != "yolo26") {
        throw std::invalid_argument("unsupported decode: " + decode);
    }
    settings.decode = decode == "yolo11" ? columba::Decode::yolo11 : columba::Decode::yolo26;
    validate_preprocessing(columba_metadata);
    const auto colour = string(field(columba_metadata, "color_order", nb::str("rgb")), "color_order");
    if (colour != "rgb" && colour != "bgr") {
        throw std::invalid_argument("color_order must be rgb or bgr");
    }
    settings.rgb = colour == "rgb";
    const auto quantisation = string(field(columba_metadata, "quantization", nb::str("fp32")), "quantization");
    if (quantisation != "fp32" && quantisation != "fp16" && quantisation != "int8") {
        throw std::invalid_argument("quantization must be fp32, fp16 or int8");
    }
    settings.quantisation = quantisation == "fp16" ? columba::Quantisation::fp16
        : quantisation == "int8" ? columba::Quantisation::int8 : columba::Quantisation::fp32;
    settings.max_det = integer(field(columba_metadata, "max_det", nb::int_(columba::maximum_detections)), "max_det");
    if (settings.max_det <= 0 || settings.max_det > columba::maximum_detections) {
        throw std::invalid_argument("max_det must be between 1 and 300");
    }
    if (columba_metadata.contains("nms")) {
        const auto nms = mapping(columba_metadata["nms"], "nms");
        const auto overlap = field(nms, "iou", nb::float_(columba::default_nms_iou));
        if ((!nb::isinstance<nb::float_>(overlap) && !nb::isinstance<nb::int_>(overlap)) ||
            nb::isinstance<nb::bool_>(overlap)) {
            throw std::invalid_argument("nms.iou must be a number");
        }
        settings.nms_iou = nb::cast<float>(overlap);
        settings.agnostic = boolean(field(nms, "agnostic", nb::bool_(false)), "nms.agnostic");
        if (!std::isfinite(settings.nms_iou) || settings.nms_iou < 0.0F || settings.nms_iou > 1.0F) {
            throw std::invalid_argument("nms.iou must be between 0 and 1");
        }
    }
    return settings;
}

columba::ImageSize frame_size(const Frame& frame) {
    const auto maximum_dimension_px = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (frame.shape(0) == 0 || frame.shape(1) == 0 || frame.shape(2) != columba::colour_channels ||
        frame.shape(0) > maximum_dimension_px || frame.shape(1) > maximum_dimension_px) {
        throw std::invalid_argument("frame must be a non-empty HxWx3 uint8 BGR C-contiguous array");
    }
    return {static_cast<int>(frame.shape(0)), static_cast<int>(frame.shape(1))};
}

Predictions prediction_array(const std::vector<columba::Detection>& detections) {
    auto storage = std::make_unique<std::vector<float>>(
        std::max(std::size_t{1}, detections.size() * columba::detection_columns)
    );
    std::size_t offset = 0;
    for (const auto& detection : detections) {
        const float row[columba::detection_columns] = {
            detection.left_px, detection.top_px, detection.right_px, detection.bottom_px,
            detection.score, static_cast<float>(detection.class_id)
        };
        std::copy(std::begin(row), std::end(row), storage->begin() + offset);
        offset += columba::detection_columns;
    }
    nb::capsule owner(storage.get(), [](void* pointer) noexcept {
        const std::unique_ptr<std::vector<float>> owned(static_cast<std::vector<float>*>(pointer));
    });
    auto* pixels = storage.release()->data();
    return Predictions(pixels, {detections.size(), columba::detection_columns}, owner);
}

columba::InferenceResult run(const columba::Detector& detector, const Frame& frame, float confidence) {
    const auto size = frame_size(frame);
    nb::gil_scoped_release release;
    return detector.detect(frame.data(), size, confidence);
}

nb::dict timing_values(const columba::Timings& timings) {
    nb::dict values;
    values["preprocess"] = timings.preprocess_ms;
    values["extract"] = timings.extract_ms;
    values["postprocess"] = timings.postprocess_ms;
    values["total"] = timings.total_ms;
    return values;
}

std::optional<columba::Compute> parse_compute(const std::optional<std::string>& compute) {
    if (!compute) {
        return std::nullopt;
    }
    if (*compute == "fp16") {
        return columba::Compute::fp16;
    }
    if (*compute == "fp32") {
        return columba::Compute::fp32;
    }
    throw std::invalid_argument("compute must be fp16, fp32 or None");
}

}

NB_MODULE(columba_native, module) {
    module.doc() = "CPU NCNN detector with C8 metadata and Ultralytics-compatible preprocessing.";
    nb::class_<columba::Detector>(module, "Detector")
        .def("__init__", [](columba::Detector* self, const std::string& directory, int threads,
                           const std::optional<std::string>& compute) {
            const auto mode = parse_compute(compute);
            auto settings = read_settings(directory);
            nb::gil_scoped_release release;
            new (self) columba::Detector(directory, std::move(settings), threads, mode);
        }, nb::arg("model_dir"), nb::arg("threads") = 4, nb::arg("compute") = nb::none())
        .def("detect", [](const columba::Detector& detector, const Frame& frame, float confidence) {
            return prediction_array(run(detector, frame, confidence).detections);
        }, nb::arg("frame").noconvert(), nb::arg("conf"))
        .def("detect_with_timings", [](const columba::Detector& detector, const Frame& frame, float confidence) {
            const auto result = run(detector, frame, confidence);
            return nb::make_tuple(prediction_array(result.detections), timing_values(result.timings));
        }, nb::arg("frame").noconvert(), nb::arg("conf"))
        .def_prop_ro("input_hw", [](const columba::Detector& detector) {
            const auto input = detector.settings().input;
            return std::make_pair(input.height_px, input.width_px);
        })
        .def_prop_ro("names", [](const columba::Detector& detector) { return detector.settings().names; })
        .def_prop_ro("compute", [](const columba::Detector& detector) {
            return detector.compute() == columba::Compute::fp16 ? "fp16" : "fp32";
        })
        .def_prop_ro("compute_flags", [](const columba::Detector& detector) {
            const auto& options = detector.options();
            nb::dict flags;
            flags["use_fp16_packed"] = options.use_fp16_packed;
            flags["use_fp16_storage"] = options.use_fp16_storage;
            flags["use_fp16_arithmetic"] = options.use_fp16_arithmetic;
            return flags;
        })
        .def_prop_ro("decode", [](const columba::Detector& detector) {
            return detector.settings().decode == columba::Decode::yolo11 ? "yolo11" : "yolo26";
        });
    module.def("_letterbox", [](const Frame& frame, int height_px, int width_px) {
        const auto size = frame_size(frame);
        auto image = std::make_unique<cv::Mat>();
        {
            nb::gil_scoped_release release;
            const cv::Mat input(size.height_px, size.width_px, CV_8UC3, const_cast<std::uint8_t*>(frame.data()));
            *image = columba::letterbox_image(input, columba::letterbox_geometry(size, {height_px, width_px}));
        }
        nb::capsule owner(image.get(), [](void* pointer) noexcept {
            const std::unique_ptr<cv::Mat> owned(static_cast<cv::Mat*>(pointer));
        });
        auto* pixels = image.release()->data;
        return nb::ndarray<nb::numpy, std::uint8_t, nb::ndim<3>, nb::c_contig>(
            pixels, {static_cast<std::size_t>(height_px), static_cast<std::size_t>(width_px),
                     columba::colour_channels}, owner
        );
    }, nb::arg("frame").noconvert(), nb::arg("height_px"), nb::arg("width_px"));
}
