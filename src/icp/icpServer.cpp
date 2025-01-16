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

// GB-ICP-ROS
#include "icpServer.hpp"
#include <iostream>
#include <chrono>
#include <thread>
#include <ctime>

namespace gb_icp_ros
{

    using utils::EigenToPointCloud2;
    using utils::GetTimestamps;
    using utils::PointCloud2ToEigen;
    std::atomic<bool> show_map_one(true);
    void debugThread(std::atomic<bool> *is_show_map)
    {
        // while (ros::ok())
        // {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        is_show_map->store(true, std::memory_order_relaxed);
        std::cout << "debug_clouds one" << std::endl;
        // }
    }

    IcpServer::IcpServer(ros::NodeHandle &nh)
        : nh_(nh)
    {
        publish_debug_clouds_ = true;
        nh.param<double>("gb_icp/max_range", max_range_, 100.0);
        nh.param<double>("gb_icp/min_range", min_range_, 0.0);
        nh.param<double>("mapping/voxel_size", voxel_size_, max_range_ / 100.0);
        nh.param<int>("gb_icp/max_points_per_voxel", max_points_per_voxel_, 20);
        nh.param<double>("gb_icp/convergence_criterion", convergence_criterion, 0.0001);
        nh.param<int>("gb_icp/max_num_iterations", max_num_iterations, 500);
        nh.param<double>("gb_icp/initial_threshold", initial_threshold_, 2.0);
        nh.param<double>("gb_icp/min_motion_th", min_motion_th_, 0.1);
        if (max_range_ < min_range_)
        {
            ROS_WARN("[WARNING] max_range is smaller than min_range, setting min_range to 0.0");
            min_range_ = 0.0;
        }
        tf2_buffer_ = new tf2_ros::Buffer();
        tf2_listener_ = new tf2_ros::TransformListener(*tf2_buffer_);
        // Construct the main GB-ICP odometry node
        odometry_ = GbICP();

        // Initialize publishers
        odom_publisher_ = nh.advertise<nav_msgs::Odometry>("/gb/odometry", queue_size_);
        traj_publisher_ = nh.advertise<nav_msgs::Path>("/gb/trajectory", queue_size_);
        if (publish_debug_clouds_)
        {
            frame_publisher_ = nh.advertise<sensor_msgs::PointCloud2>("/gb/frame", queue_size_);
            kpoints_publisher_ =
                nh.advertise<sensor_msgs::PointCloud2>("/gb/keypoints", queue_size_);
            map_publisher_ = nh.advertise<sensor_msgs::PointCloud2>("/gb/local_map", queue_size_);
        }
        // Initialize the transform buffer
        tf2_buffer_->setUsingDedicatedThread(true);
        path_msg_.header.frame_id = odom_frame_;
        // publish odometry msg
        ROS_INFO("GB-ICP ROS 1 Odometry Node Initialized");
    }

    Sophus::SE3d IcpServer::LookupTransform(const std::string &target_frame,
                                            const std::string &source_frame) const
    {
        std::string err_msg;
        std::cout << "source_frameExists: " << tf2_buffer_->_frameExists(source_frame) << " target_frameExists: " << tf2_buffer_->_frameExists(target_frame) << std::endl;
        // if (tf2_buffer_->_frameExists(source_frame) &&  //
        //     tf2_buffer_->_frameExists(target_frame)) {
        try
        {
            auto tf = tf2_buffer_->lookupTransform(target_frame, source_frame, ros::Time(0));
            return tf2::transformToSophus(tf);
        }
        catch (tf2::TransformException &ex)
        {
            ROS_WARN("%s", ex.what());
        }
        // }
        ROS_WARN("Failed to find tf between %s and %s. Reason=%s", target_frame.c_str(),
                 source_frame.c_str(), err_msg.c_str());
        return {};
    }

    std::vector<PoseData> IcpServer::Poses()
    {
        return poses_with_data;
    }

    bool IcpServer::RegisterFrame(const PointCloudXYZI::Ptr &input_cloud, geometry_msgs::PoseStamped &esk_pose, const ros::Time &stamp, std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map)
    {

        if (esk_pose.pose.orientation.w == 0.0 &&
            esk_pose.pose.orientation.x == 0.0 &&
            esk_pose.pose.orientation.y == 0.0 &&
            esk_pose.pose.orientation.z == 0.0)
        {
            return false;
        }

        std::vector<Eigen::Vector3d> points;
        tf2::pointCloudToEigenVector(input_cloud, points);
        // true
        const auto egocentric_estimation = base_frame_.empty();

        odometry_.setPoses(tf2::poseStampedToSE3d(esk_pose));
        // Register frame, main entry point to GB-ICP pipeline
        bool isSuccess = odometry_.RegisterFrame(points, feat_map);
        // double error = tf2::computePoseSimilarity(tf2::poseStampedToSE3d(esk_pose), odometry_.poses().back());

        // std::cout << "tol error: " << error << std::endl;
        // odometry_.setPoses(tf2::poseStampedToSE3d(esk_pose));

        // Compute the pose using GB, ego-centric to the LiDAR
        const Sophus::SE3d gb_pose = odometry_.poses().back();
        struct PoseData posedata;
        posedata.pose = gb_pose;
        posedata.stamp = stamp;
        poses_with_data.push_back(posedata);
        // If necessary, transform the ego-centric pose to the specified base_link/base_footprint frame
        // const auto pose = [&]() -> Sophus::SE3d
        // {
        //     if (egocentric_estimation)
        //         return gb_pose;
        //     // const Sophus::SE3d cloud2base = LookupTransform(base_frame_, cloud_frame_id);
        //     // return cloud2base * gb_pose * cloud2base.inverse();
        // }();

        // Spit the current estimated pose to ROS msgs
        PublishOdometry(gb_pose, stamp);
        // Publishing this clouds is a bit costly, so do it only if we are debugging
        if (publish_debug_clouds_)
        {
            PublishClouds(points, points, stamp, tf2::poseStampedToSE3d(esk_pose), feat_map);
            if (show_map_one.load(std::memory_order_relaxed))
            {

                show_map_one.store(false, std::memory_order_relaxed);
                std::thread map_thread(debugThread, &show_map_one);
                map_thread.detach();
                std_msgs::Header odom_header;
                odom_header.stamp = stamp;
                odom_header.frame_id = odom_frame_;
                // const auto cloud2odom = LookupTransform(odom_frame_, cloud_frame_id);

                const auto gb_map = odometry_.loadCloud(feat_map);
                // SavePointCloudToPLY(gb_map, "/home/guowenwu/workspace/Indoor_SLAM/gb_ws/output_map.ply");
                map_publisher_.publish(*EigenToPointCloud2(gb_map, odom_header));
            }
        }
        return isSuccess;
    }

    void IcpServer::PublishOdometry(const Sophus::SE3d &pose,
                                    const ros::Time &stamp)
    {
        // Header for point clouds and stuff seen from desired odom_frame

        // Broadcast the tf
        if (publish_odom_tf_)
        {
            geometry_msgs::TransformStamped transform_msg;
            transform_msg.header.stamp = stamp;
            transform_msg.header.frame_id = odom_frame_;
            transform_msg.child_frame_id = base_frame_.empty();
            transform_msg.transform = tf2::sophusToTransform(pose);
            tf_broadcaster_->sendTransform(transform_msg);
        }

        // publish trajectory msg
        geometry_msgs::PoseStamped pose_msg;
        pose_msg.header.stamp = stamp;
        pose_msg.header.frame_id = odom_frame_;
        pose_msg.pose = tf2::sophusToPose(pose);
        path_msg_.poses.push_back(pose_msg);
        traj_publisher_.publish(path_msg_);

        // publish odometry msg
        nav_msgs::Odometry odom_msg;
        odom_msg.header.stamp = stamp;
        odom_msg.header.frame_id = odom_frame_;
        odom_msg.pose.pose = tf2::sophusToPose(pose);
        odom_publisher_.publish(odom_msg);
    }

    void IcpServer::publish_odometry(const ros::Publisher &pubOdomAftMapped, geometry_msgs::PoseStamped &esk_pose)
    {
        odomAftMapped.header.frame_id = "camera_init";
        odomAftMapped.child_frame_id = "body";
        odomAftMapped.header.stamp = ros::Time::now(); // ros::Time().fromSec(lidar_end_time);
        tf2::set_posestamp(odomAftMapped.pose, esk_pose);
        pubOdomAftMapped.publish(odomAftMapped);
        // auto P = kf.get_P();
        // for (int i = 0; i < 6; i ++)
        // {
        //     int k = i < 3 ? i + 3 : i - 3;
        //     odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        //     odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        //     odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        //     odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        //     odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        //     odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
        // }

        static tf::TransformBroadcaster br;
        tf::Transform transform;
        tf::Quaternion q;
        transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
                                        odomAftMapped.pose.pose.position.y,
                                        odomAftMapped.pose.pose.position.z));
        q.setW(odomAftMapped.pose.pose.orientation.w);
        q.setX(odomAftMapped.pose.pose.orientation.x);
        q.setY(odomAftMapped.pose.pose.orientation.y);
        q.setZ(odomAftMapped.pose.pose.orientation.z);
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "body"));

        static tf::TransformBroadcaster br_world;
        transform.setOrigin(tf::Vector3(0, 0, 0));
        q.setValue(1, 0, 0, 0);
        transform.setRotation(q);

        std::cout << " gb_pose: cloud2odom = " << std::endl;
        br_world.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "world", "camera_init"));
    }

    void IcpServer::PublishClouds(const std::vector<Eigen::Vector3d> frame,
                                  const std::vector<Eigen::Vector3d> keypoints,
                                  const ros::Time &stamp,
                                  Sophus::SE3d cloudodom, std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map)
    {
        std_msgs::Header odom_header;
        odom_header.stamp = stamp;
        odom_header.frame_id = odom_frame_;
        // const auto cloud2odom = LookupTransform(odom_frame_, cloud_frame_id);
        frame_publisher_.publish(*EigenToPointCloud2(frame,cloudodom,odom_header));
        // Publish map

        // if (!publish_odom_tf_) {
        //     // debugging happens in an egocentric world
        //     std_msgs::Header cloud_header;
        //     cloud_header.stamp = stamp;
        //     cloud_header.frame_id = cloud_frame_id;

        //     frame_publisher_.publish(*EigenToPointCloud2(frame, odom_header));
        //     kpoints_publisher_.publish(*EigenToPointCloud2(keypoints, odom_header));
        //     map_publisher_.publish(*EigenToPointCloud2(gb_map, odom_header));

        //     return;
        // }

        // If transmitting to tf tree we know where the clouds are exactly
        // const auto cloud2odom = LookupTransform(odom_frame_, cloud_frame_id);
        // std::cout<<" gb_pose: cloud2odom = "<<cloud2odom.matrix()<<std::endl;
        // publish_odometry(frame_publisher_,esk_pose,cloud_frame_id);
        // kpoints_publisher_.publish(*EigenToPointCloud2(keypoints, cloud2odom, odom_header));

        // if (!base_frame_.empty()) {
        //     // const Sophus::SE3d cloud2base = LookupTransform(base_frame_, cloud_frame_id);
        //     // map_publisher_.publish(*EigenToPointCloud2(gb_map, cloud2base, odom_header));
        // } else {
        // }
    }

} // namespace gb_icp_ros
