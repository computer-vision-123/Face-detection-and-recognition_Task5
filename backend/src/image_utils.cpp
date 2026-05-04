/*
 * backend/image_utils.cpp
 * -----------------------
 * Lightweight image I/O and channel-conversion helpers.
 * Wraps OpenCV, exposes numpy arrays to Python via pybind11.
 */

#include "image_utils.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

py::array_t<uint8_t> imread_rgb(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty())
        throw std::runtime_error("imread_rgb: cannot read " + path);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    int H = img.rows, W = img.cols;
    py::array_t<uint8_t> out({(py::ssize_t)H, (py::ssize_t)W, (py::ssize_t)3});
    std::memcpy(out.request().ptr, img.data, H * W * 3);
    return out;
}

py::array_t<uint8_t> imread_gray(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty())
        throw std::runtime_error("imread_gray: cannot read " + path);
    int H = img.rows, W = img.cols;
    py::array_t<uint8_t> out({(py::ssize_t)H, (py::ssize_t)W});
    std::memcpy(out.request().ptr, img.data, H * W);
    return out;
}

py::array_t<uint8_t> imresize(
    py::array_t<uint8_t> img_np, int new_h, int new_w)
{
    auto buf  = img_np.request();
    int  H    = static_cast<int>(buf.shape[0]);
    int  W    = static_cast<int>(buf.shape[1]);
    bool color = (buf.ndim == 3);
    int  C    = color ? static_cast<int>(buf.shape[2]) : 1;

    cv::Mat src(H, W, color ? CV_8UC3 : CV_8UC1, buf.ptr);
    cv::Mat dst;
    cv::resize(src, dst, {new_w, new_h});

    if (color) {
        py::array_t<uint8_t> out({(py::ssize_t)new_h,
                                   (py::ssize_t)new_w,
                                   (py::ssize_t)C});
        std::memcpy(out.request().ptr, dst.data, new_h * new_w * C);
        return out;
    } else {
        py::array_t<uint8_t> out({(py::ssize_t)new_h, (py::ssize_t)new_w});
        std::memcpy(out.request().ptr, dst.data, new_h * new_w);
        return out;
    }
}

py::array_t<uint8_t> bgr_to_rgb(py::array_t<uint8_t> img_np) {
    auto buf = img_np.request();
    if (buf.ndim != 3 || buf.shape[2] != 3)
        throw std::runtime_error("bgr_to_rgb: expected (H,W,3)");
    int H = static_cast<int>(buf.shape[0]);
    int W = static_cast<int>(buf.shape[1]);
    cv::Mat src(H, W, CV_8UC3, buf.ptr);
    cv::Mat dst;
    cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);
    py::array_t<uint8_t> out({(py::ssize_t)H, (py::ssize_t)W, (py::ssize_t)3});
    std::memcpy(out.request().ptr, dst.data, H * W * 3);
    return out;
}

py::array_t<uint8_t> rgb_to_gray(py::array_t<uint8_t> img_np) {
    auto buf = img_np.request();
    if (buf.ndim != 3)
        throw std::runtime_error("rgb_to_gray: expected (H,W,3)");
    int H = static_cast<int>(buf.shape[0]);
    int W = static_cast<int>(buf.shape[1]);
    cv::Mat src(H, W, CV_8UC3, buf.ptr);
    cv::Mat dst;
    cv::cvtColor(src, dst, cv::COLOR_RGB2GRAY);
    py::array_t<uint8_t> out({(py::ssize_t)H, (py::ssize_t)W});
    std::memcpy(out.request().ptr, dst.data, H * W);
    return out;
}
