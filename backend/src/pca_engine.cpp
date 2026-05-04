/*
 * backend/pca_engine.cpp
 */

#include "pca_engine.h"   // Mat is now defined here

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================
//  Linear algebra helpers  (Mat is already defined in the header)
// ============================================================
static Mat matmul(const Mat& A, const Mat& B) {
    if (A.cols != B.rows) throw std::runtime_error("matmul: dimension mismatch");
    Mat C(A.rows, B.cols);
    for (int i = 0; i < A.rows; ++i)
        for (int k = 0; k < A.cols; ++k) {
            double aik = A.at(i, k);
            if (aik == 0.0) continue;
            for (int j = 0; j < B.cols; ++j)
                C.at(i, j) += aik * B.at(k, j);
        }
    return C;
}

static Mat transpose(const Mat& A) {
    Mat B(A.cols, A.rows);
    for (int r = 0; r < A.rows; ++r)
        for (int c = 0; c < A.cols; ++c)
            B.at(c, r) = A.at(r, c);
    return B;
}

static double vec_norm(const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x * x;
    return std::sqrt(s);
}

static void vec_normalize(std::vector<double>& v) {
    double n = vec_norm(v);
    if (n < 1e-12) return;
    for (double& x : v) x /= n;
}

static double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

static std::vector<double> matvec(const Mat& M, const std::vector<double>& x) {
    std::vector<double> y(M.rows, 0.0);
    for (int r = 0; r < M.rows; ++r)
        for (int c = 0; c < M.cols; ++c)
            y[r] += M.at(r, c) * x[c];
    return y;
}

static std::pair<std::vector<double>, Mat>
eigen_power(const Mat& S, int k, int max_iter = 1000, double tol = 1e-8) {
    int n = S.rows;
    if (k > n) k = n;

    Mat deflated = S;
    std::vector<double> eigenvalues(k);
    Mat eigenvectors(n, k);

    for (int comp = 0; comp < k; ++comp) {
        std::vector<double> v(n);
        for (int i = 0; i < n; ++i) v[i] = static_cast<double>(i + 1);
        vec_normalize(v);

        double lambda = 0;
        for (int iter = 0; iter < max_iter; ++iter) {
            std::vector<double> w = matvec(deflated, v);
            double new_lambda = dot(v, w);
            vec_normalize(w);
            double diff = 0;
            for (int i = 0; i < n; ++i) diff += (w[i] - v[i]) * (w[i] - v[i]);
            v = w;
            if (std::sqrt(diff) < tol && iter > 10) { lambda = new_lambda; break; }
            lambda = new_lambda;
        }
        eigenvalues[comp] = lambda;
        eigenvectors.set_col(comp, v);

        for (int r = 0; r < n; ++r)
            for (int c2 = 0; c2 < n; ++c2)
                deflated.at(r, c2) -= lambda * v[r] * v[c2];
    }
    return {eigenvalues, eigenvectors};
}

// ============================================================
//  PCAModel implementation
// ============================================================
void PCAModel::fit(py::array_t<double> X_np, int n_comp) {
    auto buf = X_np.request();
    if (buf.ndim != 2)
        throw std::runtime_error("fit: expected 2-D array (N, D)");

    int N = static_cast<int>(buf.shape[0]);
    int D = static_cast<int>(buf.shape[1]);
    n_components = std::min(n_comp, N - 1);

    auto* ptr = static_cast<double*>(buf.ptr);

    mean.assign(D, 0.0);
    for (int i = 0; i < N; ++i)
        for (int d = 0; d < D; ++d)
            mean[d] += ptr[i * D + d];
    for (double& m : mean) m /= N;

    Mat Xc(N, D);
    for (int i = 0; i < N; ++i)
        for (int d = 0; d < D; ++d)
            Xc.at(i, d) = ptr[i * D + d] - mean[d];

    Mat S(N, N);
    for (int i = 0; i < N; ++i)
        for (int j = i; j < N; ++j) {
            double s = 0;
            for (int d = 0; d < D; ++d)
                s += Xc.at(i, d) * Xc.at(j, d);
            S.at(i, j) = S.at(j, i) = s;
        }

    auto [evals, evecs_small] = eigen_power(S, n_components);

    Mat XcT = transpose(Xc);
    eigenfaces = matmul(XcT, evecs_small);

    for (int c = 0; c < eigenfaces.cols; ++c) {
        double n = 0;
        for (int r = 0; r < eigenfaces.rows; ++r)
            n += eigenfaces.at(r, c) * eigenfaces.at(r, c);
        n = std::sqrt(n);
        if (n < 1e-12) continue;
        for (int r = 0; r < eigenfaces.rows; ++r)
            eigenfaces.at(r, c) /= n;
    }

    train_projections = matmul(Xc, eigenfaces);
}

py::array_t<double> PCAModel::transform(py::array_t<double> X_np) const {
    auto buf = X_np.request();
    if (buf.ndim != 2)
        throw std::runtime_error("transform: expected 2-D array");
    int M = static_cast<int>(buf.shape[0]);
    int D = static_cast<int>(buf.shape[1]);
    auto* ptr = static_cast<double*>(buf.ptr);

    Mat Xc(M, D);
    for (int i = 0; i < M; ++i)
        for (int d = 0; d < D; ++d)
            Xc.at(i, d) = ptr[i * D + d] - mean[d];

    Mat proj = matmul(Xc, eigenfaces);

    py::array_t<double> out({(py::ssize_t)M, (py::ssize_t)proj.cols});
    // FIX: std::copy needs begin, end, destination — not just begin, end
    std::copy(proj.data.begin(), proj.data.end(),
              static_cast<double*>(out.request().ptr));
    return out;
}

py::array_t<double> PCAModel::get_train_projections() const {
    py::array_t<double> out({(py::ssize_t)train_projections.rows,
                              (py::ssize_t)train_projections.cols});
    // FIX: same — destination pointer was missing in the original
    std::copy(train_projections.data.begin(), train_projections.data.end(),
              static_cast<double*>(out.request().ptr));
    return out;
}

py::array_t<double> PCAModel::get_mean() const {
    py::array_t<double> out(mean.size());
    std::copy(mean.begin(), mean.end(),
              static_cast<double*>(out.request().ptr));
    return out;
}

py::array_t<double> PCAModel::get_eigenfaces() const {
    py::array_t<double> out({(py::ssize_t)eigenfaces.rows,
                              (py::ssize_t)eigenfaces.cols});
    std::copy(eigenfaces.data.begin(), eigenfaces.data.end(),
              static_cast<double*>(out.request().ptr));
    return out;
}

void PCAModel::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save: cannot open " + path);

    auto write_i   = [&](int v)    { f.write(reinterpret_cast<const char*>(&v), sizeof(v)); };
    auto write_vec = [&](const std::vector<double>& v) {
        int sz = static_cast<int>(v.size());
        write_i(sz);
        f.write(reinterpret_cast<const char*>(v.data()), sz * sizeof(double));
    };
    auto write_mat = [&](const Mat& m) {
        write_i(m.rows); write_i(m.cols);
        f.write(reinterpret_cast<const char*>(m.data.data()),
                m.data.size() * sizeof(double));
    };

    write_i(n_components);
    write_vec(mean);
    write_mat(eigenfaces);        // FIX: now compiles — Mat is fully known
    write_mat(train_projections); // FIX: same
}

void PCAModel::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load: cannot open " + path);

    auto read_i   = [&]() { int v; f.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; };
    auto read_vec = [&]() {
        int sz = read_i();
        std::vector<double> v(sz);
        f.read(reinterpret_cast<char*>(v.data()), sz * sizeof(double));
        return v;
    };
    auto read_mat = [&]() {
        int r = read_i(), c = read_i();
        Mat m(r, c);
        f.read(reinterpret_cast<char*>(m.data.data()),
               m.data.size() * sizeof(double));
        return m;
    };

    n_components      = read_i();
    mean              = read_vec();
    eigenfaces        = read_mat();
    train_projections = read_mat();
}