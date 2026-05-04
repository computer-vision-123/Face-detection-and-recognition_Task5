/*
 * backend/pca_engine.h
 */

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <vector>

namespace py = pybind11;

// ── Mat must be fully defined here so PCAModel members are well-formed ──
struct Mat {
    int rows, cols;
    std::vector<double> data;

    Mat() : rows(0), cols(0) {}
    Mat(int r, int c, double fill = 0.0)
        : rows(r), cols(c), data(static_cast<size_t>(r) * c, fill) {}

    double&       at(int r, int c)       { return data[r * cols + c]; }
    const double& at(int r, int c) const { return data[r * cols + c]; }

    std::vector<double> col(int c) const {
        std::vector<double> v(rows);
        for (int r = 0; r < rows; ++r) v[r] = at(r, c);
        return v;
    }
    void set_col(int c, const std::vector<double>& v) {
        for (int r = 0; r < rows; ++r) at(r, c) = v[r];
    }
};

struct PCAModel {
    int n_components{0};
    std::vector<double> mean;   // length = D
    Mat eigenfaces;             // D × k
    Mat train_projections;      // N_train × k

    void fit(py::array_t<double> X_np, int n_comp);
    py::array_t<double> transform(py::array_t<double> X_np) const;
    py::array_t<double> get_train_projections() const;
    py::array_t<double> get_mean() const;
    py::array_t<double> get_eigenfaces() const;
    void save(const std::string& path) const;
    void load(const std::string& path);
};