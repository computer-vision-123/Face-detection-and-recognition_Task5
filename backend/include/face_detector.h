/*
 * backend/face_detector.h
 * -----------------------
 * Public interface for Haar-cascade face detection.
 */

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <array>
#include <string>
#include <vector>

namespace py = pybind11;

struct DetectionResult {
    std::vector<std::array<int, 4>> faces;  // [{x,y,w,h}, ...]
    py::array_t<uint8_t> visualisation;
    double computation_time_ms{0};
    int num_faces{0};
};

DetectionResult detect_faces(
    py::array_t<uint8_t> img_np,
    double scale_factor   = 1.3,
    int    min_neighbors  = 5,
    const std::string& cascade_path = "");
