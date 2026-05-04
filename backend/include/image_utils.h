/*
 * backend/image_utils.h
 * ---------------------
 * Image I/O and channel-conversion helpers (OpenCV → numpy).
 */

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>

namespace py = pybind11;

py::array_t<uint8_t> imread_rgb (const std::string& path);
py::array_t<uint8_t> imread_gray(const std::string& path);
py::array_t<uint8_t> imresize   (py::array_t<uint8_t> img, int new_h, int new_w);
py::array_t<uint8_t> bgr_to_rgb (py::array_t<uint8_t> img);
py::array_t<uint8_t> rgb_to_gray(py::array_t<uint8_t> img);
