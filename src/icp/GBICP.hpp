//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <Eigen/Core>
#include <tuple>
#include <vector>
#include "voxel_map_util.h"
#include <sophus/se3.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

typedef std::vector<Eigen::Vector3d> Vector3dVector;
typedef std::tuple<Vector3dVector, Vector3dVector> Vector3dVectorTuple;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 3, 6> Matrix3_6d;

// map params
extern double voxel_size_;
extern double max_range_;
extern double min_range_;
extern int max_points_per_voxel_;

extern double convergence_criterion;
extern int max_num_iterations;

// th parms
extern double min_motion_th_;
extern double initial_threshold_;

extern bool is_match_success_;

struct ResultTuple
{
    ResultTuple(std::size_t n)
    {
        source.reserve(n);
        target.reserve(n);
    }
    std::vector<Eigen::Vector3d> source;
    std::vector<Eigen::Vector3d> target;
};

struct ResultTupleLinear
{
    ResultTupleLinear()
    {
        JTJ.setZero();
        JTr.setZero();
    }

    ResultTupleLinear operator+(const ResultTupleLinear &other)
    {
        this->JTJ += other.JTJ;
        this->JTr += other.JTr;
        return *this;
    }

    Matrix6d JTJ;
    Vector6d JTr;
};

struct AdaptiveThreshold
{
    explicit AdaptiveThreshold(double initial_threshold, double min_motion_th, double max_range)
        : initial_threshold_(initial_threshold),
          min_motion_th_(min_motion_th),
          max_range_(max_range) {}

    /// Update the current belief of the deviation from the prediction model
    inline void UpdateModelDeviation(const Sophus::SE3d &current_deviation)
    {
        model_deviation_ = current_deviation;
    }

    /// Returns the GB-ICP adaptive threshold used in registration
    double ComputeThreshold();
    double ComputeModelError(const Sophus::SE3d &model_deviation, double max_range);
    // configurable parameters
    double initial_threshold_;
    double min_motion_th_;
    double max_range_;

    // Local cache for ccomputation
    double model_error_sse2_ = 0;
    int num_samples_ = 0;
    Sophus::SE3d model_deviation_ = Sophus::SE3d();
};

class GbICP
{

    inline double square(double x) { return x * x; }

public:
    explicit GbICP()
        : adaptive_threshold_(initial_threshold_, min_motion_th_, max_range_) {}
    Vector3dVectorTuple GetCorrespondences(
        const Vector3dVector &points, double max_correspondance_distance, const std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map);
    void TransformPoints(const Sophus::SE3d &T, std::vector<Eigen::Vector3d> &points);
    Sophus::SE3d PoseEstimate(const std::vector<Eigen::Vector3d> &frame,
                              const std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map,
                              const Sophus::SE3d &initial_guess,
                              double max_correspondence_distance,
                              double kernel);
    std::tuple<Matrix6d, Vector6d> BuildLinearSystem(
        const std::vector<Eigen::Vector3d> &source,
        const std::vector<Eigen::Vector3d> &target,
        double kernel);
    std::vector<Eigen::Vector3d> loadCloud(const std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map) const;
    bool RegisterFrame(const std::vector<Eigen::Vector3d> &frame, std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map);
    std::vector<Eigen::Vector3d> Preprocess(const std::vector<Eigen::Vector3d> &frame,
                                            double max_range,
                                            double min_range);
    double GetAdaptiveThreshold();
    Sophus::SE3d GetPredictionModel() const;
    bool HasMoved();

public:
    std::vector<Sophus::SE3d> poses() const { return poses_; };
    void setPoses(Sophus::SE3d pose)
    {
        if (poses_.empty())
        {
            poses_.push_back(pose);
        }
        else
        {
            poses_.back() = pose;
        }
    };

private:
    // GB-ICP pipeline modules
    std::vector<Sophus::SE3d> poses_;
    AdaptiveThreshold adaptive_threshold_;
};
