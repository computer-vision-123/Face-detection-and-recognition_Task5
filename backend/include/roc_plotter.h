/*
 * backend/roc_plotter.h
 * ---------------------
 * Public interface for ROC curve computation.
 */

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

struct ROCResult {
    py::array_t<double> fpr;         // false-positive rate curve
    py::array_t<double> tpr;         // true-positive rate curve
    py::array_t<double> thresholds;  // distance thresholds used
    double auc{0.0};                 // area under the curve
    int n_genuine{0};
    int n_impostor{0};
};

ROCResult compute_roc(
    py::array_t<double> probe_proj,    // (N_probe,   k)
    py::array_t<double> gallery_proj,  // (N_gallery, k)
    py::array_t<int>    probe_labels,  // (N_probe,)
    py::array_t<int>    gallery_labels,// (N_gallery,)
    int n_thresholds = 200);
