#pragma once

#include <Eigen/Geometry>
#include <nlohmann/json.hpp>

#include "calib/io/json.h"

// Make Eigen types natively serializable for nlohmann::json
namespace nlohmann {

template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct adl_serializer<Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>> {
    static void to_json(json& j,
                        const Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& m) {
        if (m.rows() == 1 || m.cols() == 1) {
            j = json::array();
            const auto n = static_cast<size_t>(m.size());
            for (size_t i = 0; i < n; ++i) j.push_back(m(Eigen::Index(i)));
        } else {
            j = json::array();
            for (Eigen::Index r = 0; r < m.rows(); ++r) {
                json row = json::array();
                for (Eigen::Index c = 0; c < m.cols(); ++c) row.push_back(m(r, c));
                j.push_back(row);
            }
        }
    }

    static void from_json(const json& j,
                          Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& m) {
        if (!j.is_array()) throw std::runtime_error("Eigen JSON: expected array");
        if (!j.empty() && j.front().is_array()) {
            const Eigen::Index rows = static_cast<Eigen::Index>(j.size());
            const Eigen::Index cols = static_cast<Eigen::Index>(j.front().size());
            m.resize(rows, cols);
            for (Eigen::Index r = 0; r < rows; ++r)
                for (Eigen::Index c = 0; c < cols; ++c) m(r, c) = j[r][c].get<Scalar>();
        } else {
            const Eigen::Index n = static_cast<Eigen::Index>(j.size());
            if constexpr (Cols == 1 || Rows == 1) {
                m.resize((Rows == 1 ? 1 : n), (Cols == 1 ? 1 : n));
            } else {
                m.resize(n, 1);
            }
            for (Eigen::Index i = 0; i < n; ++i) m(Eigen::Index(i)) = j[i].get<Scalar>();
        }
    }
};

template <typename Scalar, int Dim, int Mode, int Options>
struct adl_serializer<Eigen::Transform<Scalar, Dim, Mode, Options>> {
    static void to_json(json& j, const Eigen::Transform<Scalar, Dim, Mode, Options>& T) {
        const auto M = T.matrix();
        j = M;
    }
    static void from_json(const json& j, Eigen::Transform<Scalar, Dim, Mode, Options>& T) {
        Eigen::Matrix<Scalar, Dim + 1, Dim + 1> M =
            j.get<Eigen::Matrix<Scalar, Dim + 1, Dim + 1>>();
        T.matrix() = M;
    }
};

}  // namespace nlohmann
