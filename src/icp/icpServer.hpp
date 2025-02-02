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
#include <memory>
#include <utility>
#include <vector>

#include "Utils.hpp"

// GB-ICP
#include "GBICP.hpp"

// ROS 1 headers
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Transform.h>
#include <ros/init.h>
#include <ros/node_handle.h>
#include <tf2_ros/static_transform_broadcaster.h>

// ROS
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <pcl/point_cloud.h> // For pcl::PointCloud
#include <pcl/point_types.h> // For pcl::PointXYZI

#include <string>
#include <atomic>

namespace gb_icp_ros
{
    struct PoseData
    {
        ros::Time stamp;
        Sophus::SE3d pose;
    };

    class IcpServer
    {
    public:
        /// OdometryServer constructor
        IcpServer(ros::NodeHandle &nh);
        /// Register new frame
        bool RegisterFrame(const PointCloudXYZI::Ptr &input_cloud, geometry_msgs::PoseStamped &esk_pose, const ros::Time &stamp, std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map);
        std::vector<PoseData> Poses();

    private:
        /// Stream the estimated pose to ROS
        void PublishOdometry(const Sophus::SE3d &pose,
                             const ros::Time &stamp);

        /// Stream the debugging point clouds for visualization (if required)
        void PublishClouds(const std::vector<Eigen::Vector3d> frame,
                           const std::vector<Eigen::Vector3d> keypoints,
                           const ros::Time &stamp,
                           Sophus::SE3d cloud2odom, std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map);
        void publish_odometry(const ros::Publisher &pubOdomAftMapped, geometry_msgs::PoseStamped &esk_pose);

        /// Utility function to compute transformation using tf tree
        Sophus::SE3d LookupTransform(const std::string &target_frame,
                                     const std::string &source_frame) const;

        /// Ros node stuff
        ros::NodeHandle nh_;
        ros::NodeHandle pnh_;
        int queue_size_ = 100000;
        std::vector<PoseData> poses_with_data;

        /// Tools for broadcasting TFs.
        tf2_ros::TransformBroadcaster *tf_broadcaster_;
        tf2_ros::Buffer *tf2_buffer_;
        tf2_ros::TransformListener *tf2_listener_;
        nav_msgs::Odometry odomAftMapped;

        bool publish_odom_tf_;
        bool publish_debug_clouds_;
        /// Data publishers.
        ros::Publisher odom_publisher_;
        ros::Publisher frame_publisher_;
        ros::Publisher kpoints_publisher_;
        ros::Publisher map_publisher_;
        ros::Publisher traj_publisher_;
        nav_msgs::Path path_msg_;

        /// GB-ICP
        GbICP odometry_;

        /// Global/map coordinate frame.
        std::string odom_frame_{"camera_init"};
        std::string base_frame_{"st_body"};
    };

} // namespace gb_icp_ros
