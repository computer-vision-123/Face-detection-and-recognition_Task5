/*
 * backend/cv_backend_module.cpp
 * -----------------------------
 * pybind11 module definition.
 * Stitches together pca_engine, face_detector, image_utils, roc_plotter
 * into a single importable Python extension: cv_backend
 *
 * Build: see CMakeLists.txt at project root
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "pca_engine.h"
#include "face_detector.h"
#include "image_utils.h"
#include "roc_plotter.h"

namespace py = pybind11;

PYBIND11_MODULE(cv_backend, m) {
    m.doc() = "FaceVision C++ backend: PCA/Eigenfaces · Face Detection · ROC Analysis";

    // ── PCAModel ────────────────────────────────────────────────────
    py::class_<PCAModel>(m, "PCAModel")
        .def(py::init<>())
        .def("fit",    &PCAModel::fit,
             py::arg("X"), py::arg("n_components"),
             "Fit PCA on training matrix X (N, D). "
             "Uses the covariance trick + power-iteration eigen-decomposition.")
        .def("transform",             &PCAModel::transform,
             py::arg("X"),
             "Project X (M, D) into eigenface space → (M, k).")
        .def("get_train_projections", &PCAModel::get_train_projections,
             "Return training-set projections (N_train, k).")
        .def("get_mean",              &PCAModel::get_mean,
             "Return mean face vector (D,).")
        .def("get_eigenfaces",        &PCAModel::get_eigenfaces,
             "Return eigenface matrix (D, k). Column i = i-th eigenface.")
        .def("save", &PCAModel::save, py::arg("path"),
             "Persist model to a compact binary file.")
        .def("load", &PCAModel::load, py::arg("path"),
             "Restore model from binary file.")
        .def_readonly("n_components", &PCAModel::n_components);

    // ── DetectionResult ─────────────────────────────────────────────
    py::class_<DetectionResult>(m, "DetectionResult")
        .def_readonly("faces",               &DetectionResult::faces)
        .def_readonly("visualisation",       &DetectionResult::visualisation)
        .def_readonly("computation_time_ms", &DetectionResult::computation_time_ms)
        .def_readonly("num_faces",           &DetectionResult::num_faces);

    // ── ROCResult ───────────────────────────────────────────────────
    py::class_<ROCResult>(m, "ROCResult")
        .def_readonly("fpr",         &ROCResult::fpr)
        .def_readonly("tpr",         &ROCResult::tpr)
        .def_readonly("thresholds",  &ROCResult::thresholds)
        .def_readonly("auc",         &ROCResult::auc)
        .def_readonly("n_genuine",   &ROCResult::n_genuine)
        .def_readonly("n_impostor",  &ROCResult::n_impostor);

    // ── Free functions ──────────────────────────────────────────────
    m.def("detect_faces", &detect_faces,
          py::arg("image"),
          py::arg("scale_factor")  = 1.3,
          py::arg("min_neighbors") = 5,
          py::arg("cascade_path")  = "",
          "Run Haar-cascade face detection on an RGB (H,W,3) uint8 array.");

    m.def("compute_roc", &compute_roc,
          py::arg("probe_proj"),
          py::arg("gallery_proj"),
          py::arg("probe_labels"),
          py::arg("gallery_labels"),
          py::arg("n_thresholds") = 200,
          "Compute ROC curve (FPR, TPR) and AUC in eigenface space.");

    m.def("imread_rgb",  &imread_rgb,  py::arg("path"),
          "Read image from disk → RGB (H,W,3) uint8.");
    m.def("imread_gray", &imread_gray, py::arg("path"),
          "Read grayscale image from disk → (H,W) uint8.");
    m.def("imresize",    &imresize,
          py::arg("img"), py::arg("new_h"), py::arg("new_w"),
          "Resize image.");
    m.def("bgr_to_rgb",  &bgr_to_rgb,  py::arg("img"),
          "Swap BGR channels to RGB.");
    m.def("rgb_to_gray", &rgb_to_gray, py::arg("img"),
          "Convert RGB image to grayscale.");
}
