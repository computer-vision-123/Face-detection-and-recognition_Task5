/*
 * backend/roc_plotter.cpp
 * -----------------------
 * Generates ROC (Receiver Operating Characteristic) curve data
 * for the face-recognition pipeline.
 *
 * Algorithm
 * ---------
 * Given a set of probe projections and gallery projections:
 *  1. Compute all pairwise L2 distances in eigenface space.
 *  2. At each distance threshold, count TP, FP, FN, TN.
 *  3. Compute TPR = TP/(TP+FN)  and  FPR = FP/(FP+TN).
 *  4. Export the (FPR, TPR) curve and AUC.
 *
 * Exposed to Python as  cv_backend.compute_roc(...)
 */

#include "roc_plotter.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

// ---- pairwise distance matrix -----------------------------------------
static std::vector<double> pairwise_l2(
    const double* A, int Na, int k,
    const double* B, int Nb)
{
    // Returns Na × Nb distances (row-major: dist[i*Nb + j])
    std::vector<double> D(Na * Nb, 0.0);
    for (int i = 0; i < Na; ++i)
        for (int j = 0; j < Nb; ++j) {
            double s = 0;
            for (int d = 0; d < k; ++d) {
                double diff = A[i * k + d] - B[j * k + d];
                s += diff * diff;
            }
            D[i * Nb + j] = std::sqrt(s);
        }
    return D;
}

// ---- trapezoid AUC ---------------------------------------------------
static double trapz_auc(const std::vector<double>& fpr,
                         const std::vector<double>& tpr)
{
    double auc = 0;
    for (size_t i = 1; i < fpr.size(); ++i)
        auc += (fpr[i] - fpr[i - 1]) * 0.5 * (tpr[i] + tpr[i - 1]);
    return std::abs(auc);
}

// ---- public function --------------------------------------------------
ROCResult compute_roc(
    py::array_t<double> probe_proj_np,    // (N_probe, k)
    py::array_t<double> gallery_proj_np,  // (N_gallery, k)
    py::array_t<int>    probe_labels_np,  // (N_probe,)
    py::array_t<int>    gallery_labels_np,// (N_gallery,)
    int n_thresholds)
{
    // ── unpack inputs ──────────────────────────────────────────────
    auto pb = probe_proj_np.request();
    auto gb = gallery_proj_np.request();
    auto pl = probe_labels_np.request();
    auto gl = gallery_labels_np.request();

    if (pb.ndim != 2 || gb.ndim != 2)
        throw std::runtime_error("compute_roc: projections must be 2-D (N, k)");
    if (pl.ndim != 1 || gl.ndim != 1)
        throw std::runtime_error("compute_roc: labels must be 1-D");
    if (pb.shape[1] != gb.shape[1])
        throw std::runtime_error("compute_roc: k-dimension mismatch");

    int Np = static_cast<int>(pb.shape[0]);
    int Ng = static_cast<int>(gb.shape[0]);
    int k  = static_cast<int>(pb.shape[1]);

    const double* P  = static_cast<const double*>(pb.ptr);
    const double* G  = static_cast<const double*>(gb.ptr);
    const int*    PL = static_cast<const int*>(pl.ptr);
    const int*    GL = static_cast<const int*>(gl.ptr);

    // ── pairwise distances ─────────────────────────────────────────
    std::vector<double> dist = pairwise_l2(P, Np, k, G, Ng);

    // ── genuine / impostor score sets ─────────────────────────────
    std::vector<double> genuine, impostor;
    genuine.reserve(Np);
    impostor.reserve(Np * Ng);

    for (int i = 0; i < Np; ++i)
        for (int j = 0; j < Ng; ++j) {
            double d = dist[i * Ng + j];
            if (PL[i] == GL[j]) genuine.push_back(d);
            else                 impostor.push_back(d);
        }

    // ── threshold sweep ───────────────────────────────────────────
    double d_min = *std::min_element(dist.begin(), dist.end());
    double d_max = *std::max_element(dist.begin(), dist.end());
    double step  = (d_max - d_min) / (n_thresholds - 1);

    std::vector<double> fpr_vec, tpr_vec, thresholds;
    thresholds.reserve(n_thresholds);
    fpr_vec.reserve(n_thresholds);
    tpr_vec.reserve(n_thresholds);

    for (int t = 0; t < n_thresholds; ++t) {
        double thresh = d_min + t * step;

        // Accept pair as "same" if distance <= threshold
        long long TP = 0, FP = 0, FN = 0, TN = 0;
        for (double d : genuine)  (d <= thresh ? TP : FN)++;
        for (double d : impostor) (d <= thresh ? FP : TN)++;

        double tpr = (TP + FN > 0) ? static_cast<double>(TP) / (TP + FN) : 0.0;
        double fpr_val = (FP + TN > 0) ? static_cast<double>(FP) / (FP + TN) : 0.0;

        thresholds.push_back(thresh);
        tpr_vec.push_back(tpr);
        fpr_vec.push_back(fpr_val);
    }

    // Sort by FPR ascending (required for AUC)
    std::vector<size_t> idx(fpr_vec.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](size_t a, size_t b){ return fpr_vec[a] < fpr_vec[b]; });

    std::vector<double> fpr_s(fpr_vec.size()), tpr_s(tpr_vec.size());
    for (size_t i = 0; i < idx.size(); ++i) {
        fpr_s[i] = fpr_vec[idx[i]];
        tpr_s[i] = tpr_vec[idx[i]];
    }

    // ── pack results ──────────────────────────────────────────────
    int n = static_cast<int>(fpr_s.size());

    auto to_np = [&](const std::vector<double>& v) {
        py::array_t<double> arr(v.size());
        std::copy(v.begin(), v.end(),
                  static_cast<double*>(arr.request().ptr));
        return arr;
    };

    ROCResult result;
    result.fpr        = to_np(fpr_s);
    result.tpr        = to_np(tpr_s);
    result.thresholds = to_np(thresholds);
    result.auc        = trapz_auc(fpr_s, tpr_s);
    result.n_genuine  = static_cast<int>(genuine.size());
    result.n_impostor = static_cast<int>(impostor.size());
    return result;
}
