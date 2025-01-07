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
#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cstddef>
#include <memory>
#include <regex>
#include <sophus/se3.hpp>
#include <string>
#include <vector>
#include <pcl/point_cloud.h> // For pcl::PointCloud
#include <pcl/point_types.h> // For pcl::PointXYZI
// ROS 1 headers
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Transform.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

namespace tf2
{

    inline geometry_msgs::Transform sophusToTransform(const Sophus::SE3d &T)
    {
        geometry_msgs::Transform t;
        t.translation.x = T.translation().x();
        t.translation.y = T.translation().y();
        t.translation.z = T.translation().z();

        Eigen::Quaterniond q(T.so3().unit_quaternion());
        t.rotation.x = q.x();
        t.rotation.y = q.y();
        t.rotation.z = q.z();
        t.rotation.w = q.w();

        return t;
    }

    template <typename T>
    void set_posestamp(T &out, geometry_msgs::PoseStamped &pose_stamped)
    {
        out.pose.position.x = pose_stamped.pose.position.x;
        out.pose.position.y = pose_stamped.pose.position.y;
        out.pose.position.z = pose_stamped.pose.position.z;
        out.pose.orientation.x = pose_stamped.pose.orientation.x;
        out.pose.orientation.y = pose_stamped.pose.orientation.y;
        out.pose.orientation.z = pose_stamped.pose.orientation.z;
        out.pose.orientation.w = pose_stamped.pose.orientation.w;
    }

    inline geometry_msgs::Pose sophusToPose(const Sophus::SE3d &T)
    {
        geometry_msgs::Pose t;
        t.position.x = T.translation().x();
        t.position.y = T.translation().y();
        t.position.z = T.translation().z();

        Eigen::Quaterniond q(T.so3().unit_quaternion());
        t.orientation.x = q.x();
        t.orientation.y = q.y();
        t.orientation.z = q.z();
        t.orientation.w = q.w();

        return t;
    }

    inline Sophus::SE3d poseStampedToSE3d(const geometry_msgs::PoseStamped &pose_stamped)
    {
        Eigen::Vector3d translation(
            pose_stamped.pose.position.x,
            pose_stamped.pose.position.y,
            pose_stamped.pose.position.z);

        Eigen::Quaterniond quaternion(
            pose_stamped.pose.orientation.w,
            pose_stamped.pose.orientation.x,
            pose_stamped.pose.orientation.y,
            pose_stamped.pose.orientation.z);

        return Sophus::SE3d(Sophus::SO3d(quaternion), translation);
    }

    inline Sophus::SE3d transformToSophus(const geometry_msgs::TransformStamped &transform)
    {
        const auto &t = transform.transform;
        return Sophus::SE3d(
            Sophus::SE3d::QuaternionType(t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z),
            Sophus::SE3d::Point(t.translation.x, t.translation.y, t.translation.z));
    }

    inline void pointCloudToEigenVector(const PointCloudXYZI::Ptr &cloud,
                                        std::vector<Eigen::Vector3d> &eigen_points)
    {
        // 清空目标向量
        eigen_points.clear();

        // 遍历点云中的每个点
        for (const auto &point : cloud->points)
        {
            // 将每个点的 (x, y, z) 坐标转换为 Eigen::Vector3d
            Eigen::Vector3d vec(point.x, point.y, point.z);
            eigen_points.push_back(vec);
        }
    }

    inline double computeTranslationError(const Sophus::SE3d &pose1, const Sophus::SE3d &pose2)
    {
        Eigen::Vector3d translation_error = pose1.translation() - pose2.translation();
        return translation_error.norm();
    }

    inline double computeRotationError(const Sophus::SE3d &pose1, const Sophus::SE3d &pose2)
    {
        Eigen::Quaterniond q1(pose1.rotationMatrix());
        Eigen::Quaterniond q2(pose2.rotationMatrix());
        double angle = q1.angularDistance(q2);
        return angle;
    }

    inline double computePoseSimilarity(const Sophus::SE3d &pose1, const Sophus::SE3d &pose2)
    {

        double translation_error = computeTranslationError(pose1, pose2);
        double rotation_error = computeRotationError(pose1, pose2);

        std::cout << "平移误差: " << translation_error << std::endl;
        std::cout << "旋转误差 (弧度): " << rotation_error << std::endl;

        double similarity = 0.5 * translation_error + 0.5 * rotation_error;

        return similarity;
    }

} // namespace tf2

namespace gb_icp_ros::utils
{
    using PointCloud2 = sensor_msgs::PointCloud2;
    using PointField = sensor_msgs::PointField;
    using Header = std_msgs::Header;

    inline std::string FixFrameId(const std::string &frame_id)
    {
        return std::regex_replace(frame_id, std::regex("^/"), "");
    }

    inline auto GetTimestampField(const PointCloud2::ConstPtr msg)
    {
        PointField timestamp_field;
        for (const auto &field : msg->fields)
        {
            if ((field.name == "t" || field.name == "timestamp" || field.name == "time"))
            {
                timestamp_field = field;
            }
        }
        if (!timestamp_field.count)
        {
            throw std::runtime_error("Field 't', 'timestamp', or 'time'  does not exist");
        }
        return timestamp_field;
    }

    // Normalize timestamps from 0.0 to 1.0
    inline auto NormalizeTimestamps(const std::vector<double> &timestamps)
    {
        const auto [min_it, max_it] = std::minmax_element(timestamps.cbegin(), timestamps.cend());
        const double min_timestamp = *min_it;
        const double max_timestamp = *max_it;

        std::vector<double> timestamps_normalized(timestamps.size());
        std::transform(timestamps.cbegin(), timestamps.cend(), timestamps_normalized.begin(),
                       [&](const auto &timestamp)
                       {
                           return (timestamp - min_timestamp) / (max_timestamp - min_timestamp);
                       });
        return timestamps_normalized;
    }

    inline auto ExtractTimestampsFromMsg(const PointCloud2::ConstPtr msg, const PointField &field)
    {
        auto extract_timestamps =
            [&msg]<typename T>(sensor_msgs::PointCloud2ConstIterator<T> &&it) -> std::vector<double>
        {
            const size_t n_points = msg->height * msg->width;
            std::vector<double> timestamps;
            timestamps.reserve(n_points);
            for (size_t i = 0; i < n_points; ++i, ++it)
            {
                timestamps.emplace_back(static_cast<double>(*it));
            }
            return NormalizeTimestamps(timestamps);
        };

        // Get timestamp field that must be one of the following : {t, timestamp, time}
        auto timestamp_field = GetTimestampField(msg);

        // According to the type of the timestamp == type, return a PointCloud2ConstIterator<type>
        using sensor_msgs::PointCloud2ConstIterator;
        if (timestamp_field.datatype == PointField::UINT32)
        {
            return extract_timestamps(PointCloud2ConstIterator<uint32_t>(*msg, timestamp_field.name));
        }
        else if (timestamp_field.datatype == PointField::FLOAT32)
        {
            return extract_timestamps(PointCloud2ConstIterator<float>(*msg, timestamp_field.name));
        }
        else if (timestamp_field.datatype == PointField::FLOAT64)
        {
            return extract_timestamps(PointCloud2ConstIterator<double>(*msg, timestamp_field.name));
        }

        // timestamp type not supported, please open an issue :)
        throw std::runtime_error("timestamp field type not supported");
    }

    inline std::unique_ptr<PointCloud2> CreatePointCloud2Msg(const size_t n_points,
                                                             const Header &header,
                                                             bool timestamp = false)
    {
        auto cloud_msg = std::make_unique<PointCloud2>();
        sensor_msgs::PointCloud2Modifier modifier(*cloud_msg);
        cloud_msg->header = header;
        cloud_msg->header.frame_id = FixFrameId(cloud_msg->header.frame_id);
        cloud_msg->fields.clear();
        int offset = 0;
        offset = addPointField(*cloud_msg, "x", 1, PointField::FLOAT32, offset);
        offset = addPointField(*cloud_msg, "y", 1, PointField::FLOAT32, offset);
        offset = addPointField(*cloud_msg, "z", 1, PointField::FLOAT32, offset);
        offset += sizeOfPointField(PointField::FLOAT32);
        if (timestamp)
        {
            // assuming timestamp on a velodyne fashion for now (between 0.0 and 1.0)
            offset = addPointField(*cloud_msg, "time", 1, PointField::FLOAT64, offset);
            offset += sizeOfPointField(PointField::FLOAT64);
        }

        // Resize the point cloud accordingly
        cloud_msg->point_step = offset;
        cloud_msg->row_step = cloud_msg->width * cloud_msg->point_step;
        cloud_msg->data.resize(cloud_msg->height * cloud_msg->row_step);
        modifier.resize(n_points);
        return cloud_msg;
    }

    inline void FillPointCloud2XYZ(const std::vector<Eigen::Vector3d> &points, PointCloud2 &msg)
    {
        sensor_msgs::PointCloud2Iterator<float> msg_x(msg, "x");
        sensor_msgs::PointCloud2Iterator<float> msg_y(msg, "y");
        sensor_msgs::PointCloud2Iterator<float> msg_z(msg, "z");
        for (size_t i = 0; i < points.size(); i++, ++msg_x, ++msg_y, ++msg_z)
        {
            const Eigen::Vector3d &point = points[i];
            *msg_x = point.x();
            *msg_y = point.y();
            *msg_z = point.z();
        }
    }

    inline void FillPointCloud2Timestamp(const std::vector<double> &timestamps, PointCloud2 &msg)
    {
        sensor_msgs::PointCloud2Iterator<double> msg_t(msg, "time");
        for (size_t i = 0; i < timestamps.size(); i++, ++msg_t)
            *msg_t = timestamps[i];
    }

    inline std::vector<double> GetTimestamps(const PointCloud2::ConstPtr msg)
    {
        auto timestamp_field = GetTimestampField(msg);

        // Extract timestamps from cloud_msg
        std::vector<double> timestamps = ExtractTimestampsFromMsg(msg, timestamp_field);

        return timestamps;
    }

    inline std::vector<Eigen::Vector3d> PointCloud2ToEigen(const PointCloud2::ConstPtr msg)
    {
        std::vector<Eigen::Vector3d> points;
        points.reserve(msg->height * msg->width);
        sensor_msgs::PointCloud2ConstIterator<float> msg_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> msg_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> msg_z(*msg, "z");
        for (size_t i = 0; i < msg->height * msg->width; ++i, ++msg_x, ++msg_y, ++msg_z)
        {
            points.emplace_back(*msg_x, *msg_y, *msg_z);
        }
        return points;
    }

    inline std::unique_ptr<PointCloud2> EigenToPointCloud2(const std::vector<Eigen::Vector3d> &points,
                                                           const Header &header)
    {
        auto msg = CreatePointCloud2Msg(points.size(), header);
        FillPointCloud2XYZ(points, *msg);
        return msg;
    }

    inline std::unique_ptr<PointCloud2> EigenToPointCloud2(const std::vector<Eigen::Vector3d> &points,
                                                           const Sophus::SE3d &T,
                                                           const Header &header)
    {
        std::vector<Eigen::Vector3d> points_t;
        points_t.resize(points.size());
        std::transform(points.cbegin(), points.cend(), points_t.begin(),
                       [&](const auto &point)
                       { return T * point; });
        return EigenToPointCloud2(points_t, header);
    }

    inline std::unique_ptr<PointCloud2> EigenToPointCloud2(const std::vector<Eigen::Vector3d> &points,
                                                           const std::vector<double> &timestamps,
                                                           const Header &header)
    {
        auto msg = CreatePointCloud2Msg(points.size(), header, true);
        FillPointCloud2XYZ(points, *msg);
        FillPointCloud2Timestamp(timestamps, *msg);
        return msg;
    }
} // namespace gb_icp_ros::utils
