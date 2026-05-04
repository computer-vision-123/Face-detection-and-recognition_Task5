/*
 * backend/face_detector.cpp
 */

#include "face_detector.h"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <dlfcn.h>
#endif

// ---- resolve path relative to this .so/.pyd ----------------------------
static std::string get_default_cascade_path() {
    std::filesystem::path lib_dir;

#ifdef _WIN32
    HMODULE hm = nullptr;
    GetModuleHandleExA(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCSTR>(&get_default_cascade_path), &hm);
    char buf[MAX_PATH];
    GetModuleFileNameA(hm, buf, MAX_PATH);
    lib_dir = std::filesystem::path(buf).parent_path();
#else
    Dl_info info;
    dladdr(reinterpret_cast<void*>(&get_default_cascade_path), &info);
    lib_dir = std::filesystem::path(info.dli_fname).parent_path();
#endif

    return (lib_dir / "data" / "haarcascade_frontalface_default.xml").string();
}

// ---- singleton cascade loader ------------------------------------------
static cv::CascadeClassifier& get_cascade(const std::string& xml_path = "") {
    static cv::CascadeClassifier cascade;
    if (cascade.empty()) {
        std::string path = xml_path.empty() ? get_default_cascade_path() : xml_path;
        if (!cascade.load(path))
            throw std::runtime_error(
                "face_detector: cannot load Haar cascade from " + path);
    }
    return cascade;
}

// ---- public function (unchanged below this line) -----------------------
DetectionResult detect_faces(
    py::array_t<uint8_t> img_np,
    double scale_factor,
    int    min_neighbors,
    const std::string& cascade_path)
{
    auto buf = img_np.request();
    if (buf.ndim != 3)
        throw std::runtime_error("detect_faces: expected HxWx3 uint8 array");

    int H = static_cast<int>(buf.shape[0]);
    int W = static_cast<int>(buf.shape[1]);

    cv::Mat img(H, W, CV_8UC3, buf.ptr);
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);

    auto& cascade = get_cascade(cascade_path);

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<cv::Rect> rects;
    cascade.detectMultiScale(gray, rects, scale_factor, min_neighbors);
    auto t1 = std::chrono::high_resolution_clock::now();

    cv::Mat vis = img.clone();
    for (auto& r : rects)
        cv::rectangle(vis, r, cv::Scalar(50, 200, 80), 2);

    py::array_t<uint8_t> vis_np({(py::ssize_t)H, (py::ssize_t)W, (py::ssize_t)3});
    std::memcpy(vis_np.request().ptr, vis.data, H * W * 3);

    DetectionResult res;
    for (auto& r : rects)
        res.faces.push_back({r.x, r.y, r.width, r.height});
    res.visualisation       = vis_np;
    res.computation_time_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.num_faces = static_cast<int>(rects.size());
    return res;
}