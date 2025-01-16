// MIT License
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

#include "GBICP.hpp"

#include <Eigen/Core>
#include <tuple>
#include <vector>

// map params
double voxel_size_ = 1.0;
double max_range_ = 100.0;
double min_range_ = 5.0;
int max_points_per_voxel_ = 20;
bool is_match_success_ = false;
double convergence_criterion = 0.0001;
int max_num_iterations = 500;

// th parms
double min_motion_th_ = 0.1;
double initial_threshold_ = 2.0;

// Motion compensation
double AdaptiveThreshold::ComputeModelError(const Sophus::SE3d &model_deviation, double max_range)
{
    const double theta = Eigen::AngleAxisd(model_deviation.rotationMatrix()).angle();
    const double delta_rot = 2.0 * max_range * std::sin(theta / 2.0);
    const double delta_trans = model_deviation.translation().norm();
    return delta_trans + delta_rot;
}
double AdaptiveThreshold::ComputeThreshold()
{
    double model_error = ComputeModelError(model_deviation_, max_range_);
    if (model_error > min_motion_th_)
    {
        model_error_sse2_ += model_error * model_error;
        num_samples_++;
    }

    if (num_samples_ < 1)
    {
        return initial_threshold_;
    }
    return std::sqrt(model_error_sse2_ / num_samples_);
}

Vector3dVectorTuple GbICP::GetCorrespondences(
    const Vector3dVector &frame, double max_correspondance_distance, const std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map)
{
    // Lambda Function to obtain the KNN of one point, maybe refactor
    auto GetClosestNeighboor = [&](const Eigen::Vector3d &point)
    {
        auto kx = static_cast<int>(point[0] / voxel_size_);
        auto ky = static_cast<int>(point[1] / voxel_size_);
        auto kz = static_cast<int>(point[2] / voxel_size_);
        std::vector<VOXEL_LOC> voxels;
        voxels.reserve(27);
        for (int i = kx - 1; i < kx + 1 + 1; ++i)
        {
            for (int j = ky - 1; j < ky + 1 + 1; ++j)
            {
                for (int k = kz - 1; k < kz + 1 + 1; ++k)
                {
                    voxels.emplace_back(i, j, k);
                }
            }
        }

        using Vector3dVector = std::vector<Eigen::Vector3d>;
        Vector3dVector neighboors;
        neighboors.reserve(27 * max_points_per_voxel_);
        std::for_each(voxels.cbegin(), voxels.cend(), [&](const auto &voxel)
                      {
            auto search = feat_map.find(voxel);
            if (search != feat_map.end() && search->second->plane_ptr_->is_plane) {
                const auto &points = search->second->all_points;
                if (!points.empty()) {
                    for (const auto &point : points) {
                        neighboors.emplace_back(point);
                    }
                }
            } });

        Eigen::Vector3d closest_neighbor;
        double closest_distance2 = std::numeric_limits<double>::max();
        std::for_each(neighboors.cbegin(), neighboors.cend(), [&](const auto &neighbor)
                      {
            double distance = (neighbor - point).squaredNorm();
            if (distance < closest_distance2) {
                closest_neighbor = neighbor;
                closest_distance2 = distance;
            } });

        return closest_neighbor;
    };
    using points_iterator = std::vector<Eigen::Vector3d>::const_iterator;
    const auto [source, target] = tbb::parallel_reduce(
        // Range
        tbb::blocked_range<points_iterator>{frame.cbegin(), frame.cend()},
        // Identity
        ResultTuple(frame.size()),
        // 1st lambda: Parallel computation
        [max_correspondance_distance, &GetClosestNeighboor](
            const tbb::blocked_range<points_iterator> &r, ResultTuple res) -> ResultTuple
        {
            auto &[src, tgt] = res;
            src.reserve(r.size());
            tgt.reserve(r.size());
            for (const auto &point : r)
            {
                Eigen::Vector3d closest_neighboors = GetClosestNeighboor(point);
                if ((closest_neighboors - point).norm() < max_correspondance_distance)
                {
                    src.emplace_back(point);
                    tgt.emplace_back(closest_neighboors);
                }
            }
            return res;
        },
        // 2nd lambda: Parallel reduction
        [](ResultTuple a, const ResultTuple &b) -> ResultTuple
        {
            auto &[src, tgt] = a;
            const auto &[srcp, tgtp] = b;
            src.insert(src.end(), //
                       std::make_move_iterator(srcp.begin()), std::make_move_iterator(srcp.end()));
            tgt.insert(tgt.end(), //
                       std::make_move_iterator(tgtp.begin()), std::make_move_iterator(tgtp.end()));
            return a;
        });

    return std::make_tuple(source, target);
}

void GbICP::TransformPoints(const Sophus::SE3d &T, std::vector<Eigen::Vector3d> &points)
{
    std::transform(points.cbegin(), points.cend(), points.begin(),
                   [&](const auto &point)
                   { return T * point; });
}

Sophus::SE3d GbICP::PoseEstimate(const std::vector<Eigen::Vector3d> &frame,
                                 const std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map,
                                 const Sophus::SE3d &initial_guess,
                                 double max_correspondence_distance,
                                 double kernel)
{
    if (feat_map.empty())
        return initial_guess;

    // Equation (9)
    std::vector<Eigen::Vector3d> source = frame;
    TransformPoints(initial_guess, source);

    // ICP-loop
    Sophus::SE3d T_icp = Sophus::SE3d();
    int j = 0;
    Vector6d dx;
    // for (; j < max_num_iterations; ++j)
    while (true)
    {
        j++;
        // Equation (10)
        const auto &[src, tgt] = GetCorrespondences(source, max_correspondence_distance, feat_map);
        if (j >= max_num_iterations)
        {
            is_match_success_ = false;
            SavePointCloudToPLY(source, "/home/guowenwu/workspace/Indoor_SLAM/gb_ws/output_source_no.ply");
            SavePointCloudToPLY(src, "/home/guowenwu/workspace/Indoor_SLAM/gb_ws/output_src_no.ply");
            SavePointCloudToPLY(tgt, "/home/guowenwu/workspace/Indoor_SLAM/gb_ws/output_tgt_no.ply");
            break;
        }
        // Equation (11)
        const auto &[JTJ, JTr] = BuildLinearSystem(src, tgt, kernel);
        dx = JTJ.ldlt().solve(-JTr);
        const Sophus::SE3d estimation = Sophus::SE3d::exp(dx);
        // Equation (12)
        TransformPoints(estimation, source);
        // Update iterations
        T_icp = estimation * T_icp;
        // Termination criteria
        // std::cout << "ICP converged after false   norm = " << dx.norm() << std::endl;
        if (dx.norm() < convergence_criterion)
        {
            // SavePointCloudToPLY(source, "/home/guowenwu/workspace/Indoor_SLAM/gb_ws/output_source.ply");
            // SavePointCloudToPLY(src, "/home/guowenwu/workspace/Indoor_SLAM/gb_ws/output_src.ply");
            // SavePointCloudToPLY(tgt, "/home/guowenwu/workspace/Indoor_SLAM/gb_ws/output_tgt.ply");
            break;
        }
    }
    // std::cout << "ICP converged after false   j = " << j << std::endl;
    // Spit the final transformation
    return T_icp * initial_guess;
}

std::vector<Eigen::Vector3d> GbICP::loadCloud(const std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map) const
{
    std::vector<Eigen::Vector3d> a_points;
    a_points.reserve(max_points_per_voxel_ * feat_map.size());
    int i = 0;
    for (const auto &pair : feat_map)
    {
        const auto &points = pair.second->all_points;
        i++;
        if (i > 5000)
            break;
        if (!points.empty())
        {
            for (const auto &point : points)
            {
                a_points.emplace_back(point);
            }
        }
    }
    return a_points;
}

std::tuple<Matrix6d, Vector6d> GbICP::BuildLinearSystem(const std::vector<Eigen::Vector3d> &source, const std::vector<Eigen::Vector3d> &target, double kernel)
{
    auto compute_jacobian_and_residual = [&](auto i)
    {
        const Eigen::Vector3d residual = source[i] - target[i];
        Matrix3_6d J_r;
        J_r.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat(source[i]);
        return std::make_tuple(J_r, residual);
    };

    const auto &[JTJ, JTr] = tbb::parallel_reduce(
        // Range
        tbb::blocked_range<size_t>{0, source.size()},
        // Identity
        ResultTupleLinear(),
        // 1st Lambda: Parallel computation
        [&](const tbb::blocked_range<size_t> &r, ResultTupleLinear J) -> ResultTupleLinear
        {
            auto Weight = [&](double residual2)
            {
                return square(kernel) / square(kernel + residual2);
            };
            auto &[JTJ_private, JTr_private] = J;
            for (auto i = r.begin(); i < r.end(); ++i)
            {
                const auto &[J_r, residual] = compute_jacobian_and_residual(i);
                const double w = Weight(residual.squaredNorm());
                JTJ_private.noalias() += J_r.transpose() * w * J_r;
                JTr_private.noalias() += J_r.transpose() * w * residual;
            }
            return J;
        },
        // 2nd Lambda: Parallel reduction of the private Jacboians
        [&](ResultTupleLinear a, const ResultTupleLinear &b) -> ResultTupleLinear
        { return a + b; });

    return std::make_tuple(JTJ, JTr);
}

std::vector<Eigen::Vector3d> GbICP::Preprocess(const std::vector<Eigen::Vector3d> &frame,
                                               double max_range,
                                               double min_range)
{
    std::vector<Eigen::Vector3d> inliers;
    std::copy_if(frame.cbegin(), frame.cend(), std::back_inserter(inliers), [&](const auto &pt)
                 {
        const double norm = pt.norm();
        return norm < max_range && norm > min_range; });
    return inliers;
}

bool GbICP::RegisterFrame(const std::vector<Eigen::Vector3d> &frame, std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map)
{
    const auto &cropped_frame = Preprocess(frame, max_range_, min_range_);
    // const double sigma = GetAdaptiveThreshold();
    const double sigma = 0.1;
    // Compute initial_guess for ICP
    // const auto prediction = GetPredictionModel();
    // const auto last_pose = !poses_.empty() ? poses_.back() : Sophus::SE3d();
    // const auto initial_guess = last_pose * prediction;
    const auto initial_guess = !poses_.empty() ? poses_.back() : Sophus::SE3d();
    is_match_success_ = true;
    // Run icp
    const Sophus::SE3d new_pose = PoseEstimate(cropped_frame, //
                                               feat_map,      //
                                               initial_guess, //
                                               3.0 * sigma,   //
                                               sigma / 3.0);
    // std::cout << "SE3: \n"
    //           << new_pose.matrix() << std::endl; // 4x4 变换矩阵
    // std::cout << "Rotation matrix: \n"
    //           << new_pose.rotationMatrix() << std::endl; // 3x3 旋转矩阵
    // std::cout << "Translation vector: \n"
    //           << new_pose.translation().transpose() << std::endl; // 平移向量
    const auto model_deviation = initial_guess.inverse() * new_pose;
    adaptive_threshold_.UpdateModelDeviation(model_deviation);
    // local_map_.Update(frame, new_pose);
    // local_map_.Update(frame, last_pose);
    // std::cout<<" point size "<<local_map_.Pointcloud().size()<<std::endl;
    poses_.push_back(new_pose);
    return is_match_success_;
}

double GbICP::GetAdaptiveThreshold()
{
    if (!HasMoved())
    {
        return initial_threshold_;
    }
    return adaptive_threshold_.ComputeThreshold();
}

Sophus::SE3d GbICP::GetPredictionModel() const
{
    Sophus::SE3d pred = Sophus::SE3d();
    const size_t N = poses_.size();
    if (N < 2)
        return pred;
    return poses_[N - 2].inverse() * poses_[N - 1];
}

bool GbICP::HasMoved()
{
    if (poses_.empty())
        return false;
    const double motion = (poses_.front().inverse() * poses_.back()).translation().norm();
    return motion > 5.0 * min_motion_th_;
}
