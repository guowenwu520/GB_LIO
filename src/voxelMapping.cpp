// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/random_sample.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/String.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <queue>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>
#include "icp/icpServer.hpp"
#include "voxel_map_util.h"
#define foreach BOOST_FOREACH

#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define MAXN (720000)
#define PUBFRAME_PERIOD (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool time_sync_en = false, extrinsic_est_en = true, path_en = true, encoder_fusion_en = true;
double lidar_time_offset = 0.0;
double encoder_zeropoint_offset_deg = 0.0;
double mean_division_error_points = 0;
double max_error_points = 5.0;
double max_distance = 50.0;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double intensity_th = 1.0f;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string lid_topic, imu_topic, encoder_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_surf_min = 0;
double total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0, lidar_end_time_prev = 0;
int effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_index = 0;
int scan_index = 0;
double total_time_ = 0;
int total_num_ = 0;
bool point_selected_surf[100000] = {0};
bool lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false, icp_en = true;
double publish_limit_z = 1000000.0;
bool isRepositioning = false;
int publish_dense_skip = 1;
int publish_downsample_points = 1000000;
int publish_path_skip = 1;
bool adaptive_voxelization = false;
std::vector<double> adaptive_threshold;
std::vector<double> adaptive_multiple_factor;
bool init_gravity_with_pose;

vector<vector<int>> pointSearchInd_surf;
vector<PointVector> Nearest_Points;
vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);
vector<double> extrinT_encoder(3, 0.0);
vector<double> extrinR_encoder(9, 0.0);
deque<double> time_buffer;
deque<PointCloudXYZI::Ptr> lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr last_feats_down_body(new PointCloudXYZI()); // 记录上一帧的点云
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
// PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
// PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;
std::vector<M3D> var_down_body;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterAdaptive;
pcl::RandomSample<PointType> downSizeFilterVis;

std::vector<float> nn_dist_in_feats;
std::vector<float> nn_plane_std;
// PointCloudXYZI::Ptr feats_with_correspondence(new PointCloudXYZI());

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(zero3d);
V3D Lidar_T_wrt_IMU(zero3d);
M3D Lidar_R_wrt_IMU(eye3d);

SO3 R_lidar_o_encoder(eye3d);
V3D t_lidar_o_encoder(zero3d);
SO3 R_encoder_o_zero_point(eye3d);

// params for voxel mapping algorithm
double min_eigen_value = 0.003;
int max_layer = 0;

int max_cov_points_size = 50;
int max_points_size = 50;
double sigma_num = 2.0;
double max_voxel_size = 1.0;
std::vector<int> layer_size;

// record point usage
double mean_effect_points = 0;
double mean_ds_points = 0;
double mean_raw_points = 0;

// record time
double undistort_time_mean = 0;
double down_sample_time_mean = 0;
double calc_cov_time_mean = 0;
double scan_match_time_mean = 0;
double ekf_solve_time_mean = 0;
double map_update_time_mean = 0;

double ranging_cov = 0.0;
double angle_cov = 0.0;
std::vector<double> layer_point_size;

bool publish_voxel_map = true;
int publish_max_voxel_layer = 0;

std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map;
state_ikfom current_state_point;

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point, last_state_point;
// vect3 pos_lid;

// 思路二
esekfom::esekf<state_ikfom, 12, input_ikfom> hide_kf;
std::unordered_map<VOXEL_LOC, OctoTree *> hide_voxel_map;
state_ikfom hide_state_point;
PointCloudXYZI::Ptr hide_feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr hide_feats_undistort(new PointCloudXYZI());
bool hide_flg_first_scan = true;
geometry_msgs::Quaternion hide_geoQuat;

// 思路三
gb_icp_ros::IcpServer *icpServer;

nav_msgs::Path path;
nav_msgs::Path keyPath;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());
shared_ptr<ImuProcess> hide_p_imu(new ImuProcess());

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

const bool var_contrast(pointWithCov &x, pointWithCov &y)
{
    return (x.cov.diagonal().norm() < y.cov.diagonal().norm());
};

void pointBodyToWorld_ikfom(PointType const *const pi, PointType *const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void showStatus(state_ikfom &state, int a)
{
    if (a == 1)
    {
        cout << "pre pos: x = " << state.rot.coeffs()[0] << " y = " << state.rot.coeffs()[1] << " z = " << state.rot.coeffs()[2] << " w = " << state.rot.coeffs()[3] << endl;
    }
    else
    {
        cout << "chang pos: x = " << state.rot.coeffs()[0] << " y = " << state.rot.coeffs()[1] << " z = " << state.rot.coeffs()[2] << " w = " << state.rot.coeffs()[3] << endl;
    }
}

void pointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    //    po->intensity = pi->intensity;
}

template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;

    po->curvature = pi->curvature;
    po->normal_x = pi->normal_x;
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    auto time_offset = lidar_time_offset;
    //    std::printf("lidar offset:%f\n", lidar_time_offset);
    mtx_buffer.lock();
    scan_count++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() + time_offset < last_timestamp_lidar)
    {
        //        ROS_ERROR("lidar loop back, clear buffer");
        ROS_ERROR("lidar loop back, skip this scan!!!");
        //        lidar_buffer.clear();
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        return;
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    PointCloudXYZI::Ptr ptr2(new PointCloudXYZI());
    p_pre->intensity_th = intensity_th;
    p_pre->process(msg, ptr, ptr2);
    // 删除过少的点
    // std::printf("points: %ld\n", ptr->size());
    if (ptr->size() < 120)
    {
        //        ROS_ERROR("lidar loop back, clear buffer");
        ROS_ERROR("Too few points, skip this scan!!!");
        //        lidar_buffer.clear();
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        return;
    }

    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec() + time_offset);
    // lidar_buffer.push_back(ptr2);
    // time_buffer.push_back(msg->header.stamp.toSec() + time_offset+ max(ptr->points.back().curvature,ptr2->points.back().curvature)/2000.0f);
    last_timestamp_lidar = msg->header.stamp.toSec() + time_offset;
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        //        ROS_ERROR("lidar loop back, clear buffer");
        ROS_ERROR("lidar loop back, skip this scan!!!");
        //        lidar_buffer.clear();
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        return;
    }
    last_timestamp_lidar = msg->header.stamp.toSec();

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty())
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);

    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    publish_count++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp =
            ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    if (timestamp < last_timestamp_imu)
    {
        //        ROS_WARN("imu loop back, clear buffer");
        //        imu_buffer.clear();
        ROS_WARN("imu loop back, ignoring!!!");
        ROS_WARN("current T: %f, last T: %f", timestamp, last_timestamp_imu);
        return;
    }
    // 剔除异常数据
    if (std::abs(msg->angular_velocity.x) > 10 || std::abs(msg->angular_velocity.y) > 10 || std::abs(msg->angular_velocity.z) > 10)
    {
        ROS_WARN("Large IMU measurement!!! Drop Data!!! %.3f  %.3f  %.3f",
                 msg->angular_velocity.x,
                 msg->angular_velocity.y,
                 msg->angular_velocity.z);
        return;
    }

    //    // 如果是第一帧 拿过来做重力对齐
    //    // TODO 用多帧平均的重力
    //    if (is_first_imu) {
    //        double acc_vec[3] = {msg_in->linear_acceleration.x, msg_in->linear_acceleration.y, msg_in->linear_acceleration.z};
    //
    //        R__world__o__initial = SO3(g2R(Eigen::Vector3d(acc_vec)));
    //
    //        is_first_imu = false;
    //    }

    last_timestamp_imu = timestamp;

    mtx_buffer.lock();

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty())
    {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {

            //            std::printf("\nFirst 100 points: \n");
            //            for(int i=0; i < 100; ++i){
            //                std::printf("%f ", meas.lidar->points[i].curvature  / double(1000));
            //            }
            //
            //            std::printf("\n Last 100 points: \n");
            //            for(int i=100; i >0; --i){
            //                std::printf("%f ", meas.lidar->points[meas.lidar->size() - i - 1].curvature / double(1000));
            //            }
            //            std::printf("last point offset time: %f\n", meas.lidar->points.back().curvature / double(1000));
            scan_num++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            //            lidar_end_time = meas.lidar_beg_time + (meas.lidar->points[meas.lidar->points.size() - 20]).curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
            //            std::printf("pcl_bag_time: %f\n", meas.lidar_beg_time);
            //            std::printf("lidar_end_time: %f\n", lidar_end_time);
        }

        meas.lidar_end_time = lidar_end_time;
        //        std::printf("Scan start timestamp: %f, Scan end time: %f\n", meas.lidar_beg_time, meas.lidar_end_time);

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time)
            break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());

void savePointCloud2ToPCD(const sensor_msgs::PointCloud2 &input_cloud_msg, const std::string &filename)
{
    // 创建 PCL 点云对象
    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;

    // 将 sensor_msgs::PointCloud2 转换为 pcl::PointCloud
    pcl::fromROSMsg(input_cloud_msg, pcl_cloud);

    // 保存到 PCD 文件
    if (pcl::io::savePCDFileASCII(filename, pcl_cloud) == 0)
    {
        ROS_INFO("Successfully saved point cloud to %s", filename.c_str());
    }
    else
    {
        ROS_ERROR("Failed to save point cloud to %s", filename.c_str());
    }
}

void publish_frame_world(const ros::Publisher &pubLaserCloudFull)
{
    if (scan_pub_en)
    {
        if (scan_index <= 1)
        {
            return;
        }
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        // 随机下采样 只用于可视化
        if (!dense_pub_en)
        {
            downSizeFilterVis.setSample(publish_downsample_points);
            downSizeFilterVis.setSeed(std::rand());
            downSizeFilterVis.setInputCloud(feats_down_body);
            downSizeFilterVis.filter(*laserCloudFullRes);
        }
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI laserCloudWorld;
        for (int i = 0; i < size; i++)
        {
            if (i % publish_dense_skip != 0)
            {
                continue;
            }
            PointType const *const p = &laserCloudFullRes->points[i];
            // 转换到gravity aligned系 删除过高的点, 保证平面的可视化效果
            V3D p_body;
            p_body << p->x, p->y, p->z;
            V3D p_gravaity = p_imu->Initial_R_wrt_G * state_point.rot * state_point.offset_R_L_I * p_body;
            if (p_gravaity.z() > publish_limit_z)
            {
                continue;
            }

            //            if (p->x < 0 and p->x > -4
            //                    and p->y < 1.5 and p->y > -1.5
            //                            and p->z < 2 and p->z > -1) {
            //                continue;
            //            }
            PointType p_world;

            RGBpointBodyToWorld(p, &p_world);
            //            if (p_world.z > 1) {
            //                continue;
            //            }
            laserCloudWorld.push_back(p_world);
            //            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
//                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(laserCloudWorld, laserCloudmsg);
        // savePointCloud2ToPCD(laserCloudmsg, "/home/guowenwu/workspace/Indoor_SLAM/alpha_ws/voxel_plane.pcd");
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }
}

void publish_frame_body(const ros::Publisher &pubLaserCloudFull_body)
{
    //    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));
    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&laserCloudFullRes->points[i],
                               &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_map(const ros::Publisher &pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template <typename T>
void set_posestamp(T &out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
}

template <typename T>
void set_keyposestamp(T &out)
{
    out.pose.position.x = hide_state_point.pos(0);
    out.pose.position.y = hide_state_point.pos(1);
    out.pose.position.z = hide_state_point.pos(2);
    out.pose.orientation.x = hide_geoQuat.x;
    out.pose.orientation.y = hide_geoQuat.y;
    out.pose.orientation.z = hide_geoQuat.z;
    out.pose.orientation.w = hide_geoQuat.w;
}

void publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time); // ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

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
    q.setValue(p_imu->Initial_R_wrt_G.x(), p_imu->Initial_R_wrt_G.y(), p_imu->Initial_R_wrt_G.z(), p_imu->Initial_R_wrt_G.w());
    transform.setRotation(q);
    br_world.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "world", "camera_init"));
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % publish_path_skip == 0)
    {
        path.header.stamp = msg_body_pose.header.stamp;
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

void publish_key_path(const ros::Publisher pubkeyPath)
{
    set_keyposestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if keyPath is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % publish_path_skip == 0)
    {
        keyPath.header.stamp = msg_body_pose.header.stamp;
        keyPath.poses.push_back(msg_body_pose);
        pubkeyPath.publish(keyPath);
    }
}

void transformLidar(const state_ikfom &state_point, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud)
{
    trans_cloud->clear();
    for (size_t i = 0; i < input_cloud->size(); i++)
    {
        pcl::PointXYZINormal p_c = input_cloud->points[i];
        Eigen::Vector3d p_lidar(p_c.x, p_c.y, p_c.z);
        // HACK we need to specify p_body as a V3D type!!!
        V3D p_body = state_point.rot * (state_point.offset_R_L_I * p_lidar + state_point.offset_T_L_I) + state_point.pos;
        PointType pi;
        pi.x = p_body(0);
        pi.y = p_body(1);
        pi.z = p_body(2);
        pi.intensity = p_c.intensity;
        trans_cloud->points.push_back(pi);
    }
}

M3D transformLiDARCovToWorld(Eigen::Vector3d &p_lidar, const esekfom::esekf<state_ikfom, 12, input_ikfom> &kf, const Eigen::Matrix3d &COV_lidar)
{
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(p_lidar);
    auto state = kf.get_x();

    // lidar到body的方差传播
    // 注意外参的var是先rot 后pos
    M3D il_rot_var = kf.get_P().block<3, 3>(6, 6);
    M3D il_t_var = kf.get_P().block<3, 3>(9, 9);

    M3D COV_body =
        state.offset_R_L_I * COV_lidar * state.offset_R_L_I.conjugate() + state.offset_R_L_I * (-point_crossmat) * il_rot_var * (-point_crossmat).transpose() * state.offset_R_L_I.conjugate() + il_t_var;

    // body的坐标
    V3D p_body = state.offset_R_L_I * p_lidar + state.offset_T_L_I;

    // body到world的方差传播
    // 注意pose的var是先pos 后rot
    point_crossmat << SKEW_SYM_MATRX(p_body);
    M3D rot_var = kf.get_P().block<3, 3>(3, 3);
    M3D t_var = kf.get_P().block<3, 3>(0, 0);

    // Eq. (3)
    M3D COV_world =
        state.rot * COV_body * state.rot.conjugate() + state.rot * (-point_crossmat) * rot_var * (-point_crossmat).transpose() * state.rot.conjugate() + t_var;

    return COV_world;
    // Voxel map 真实实现
    //    M3D cov_world = R_body * COV_lidar * R_body.conjugate() +
    //          (-point_crossmat) * rot_var * (-point_crossmat).transpose() + t_var;
}

void observation_model_share(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{

    // =================================================================================================================
    // 用当前迭代轮最新的位姿估计值 将点云转换到world地图系
    vector<pointWithCov> pv_list;
    PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI);
    // FIXME stupid mistake 这里应该用迭代的最新线性化点
    // FIXME stupid mistake 这里应该用迭代的最新线性化点
    //    transformLidar(state_point, feats_down_body, world_lidar);
    transformLidar(s, feats_down_body, world_lidar);
    pv_list.resize(feats_down_body->size());
    for (size_t i = 0; i < feats_down_body->size(); i++)
    {
        // 保存body系和world系坐标
        pointWithCov pv;
        pv.point << feats_down_body->points[i].x, feats_down_body->points[i].y, feats_down_body->points[i].z;
        pv.point_world << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
        // 计算lidar点的cov
        // 注意这个在每次迭代时是存在重复计算的 因为lidar系的点云covariance是不变的
        // M3D cov_lidar = calcBodyCov(pv.point, ranging_cov, angle_cov);
        M3D cov_lidar = var_down_body[i];
        // 将body系的var转换到world系
        M3D cov_world = transformLiDARCovToWorld(pv.point, kf, cov_lidar);
        pv.cov = cov_world;
        pv.cov_lidar = cov_lidar;
        pv_list[i] = pv;
    }

    // ===============================================================================================================
    // 查找最近点 并构建residual
    double match_start = omp_get_wtime();
    std::vector<ptpl, Eigen::aligned_allocator<ptpl>> ptpl_list;
    std::vector<V3D, Eigen::aligned_allocator<V3D>> non_match_list;

    current_state_point = s;
    BuildResidualListOMP(voxel_map, max_voxel_size, 3.0, max_layer, pv_list,
                         ptpl_list, non_match_list);
    double match_end = omp_get_wtime();
    // std::printf("Match Time: %f\n", match_end - match_start);

    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    // 根据匹配结果 设置H和R的维度
    // h_x是观测值对状态量的导数 TODO 为什么不加上状态量对状态量误差的导数？？？？像quaternion那本书？
    effct_feat_num = ptpl_list.size();
    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); // 23 因为点面距离只和位姿 外参有关 对其他状态量的导数都是0
    ekfom_data.h.resize(effct_feat_num);
    ekfom_data.R.resize(effct_feat_num, 1); // 把R作为向量 用的时候转换成diag
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < effct_feat_num; i++)
    {

        //        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(ptpl_list[i].point);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        //        const PointType &norm_p = corr_normvect->points[i];
        //        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);
        V3D norm_vec(ptpl_list[i].normal);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() * norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); // s.rot.conjugate()*norm_vec);
            // ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec.x(), norm_vec.y(), norm_vec.z(), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            // ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec.x(), norm_vec.y(), norm_vec.z(), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        //        ekfom_data.h(i) = -norm_p.intensity;
        float pd2 = norm_vec.x() * ptpl_list[i].point_world.x() + norm_vec.y() * ptpl_list[i].point_world.y() + norm_vec.z() * ptpl_list[i].point_world.z() + ptpl_list[i].d;
        ekfom_data.h(i) = -pd2;

        // norm_p中存了匹配的平面法向 还有点面距离
        // V3D point_world = s.rot * (s.offset_R_L_I * ptpl_list[i].point + s.offset_T_L_I) + s.pos;
        V3D point_world = ptpl_list[i].point_world;
        // /*** get the normal vector of closest surface/corner ***/
        Eigen::Matrix<double, 1, 6> J_nq;
        J_nq.block<1, 3>(0, 0) = point_world - ptpl_list[i].center;
        J_nq.block<1, 3>(0, 3) = -ptpl_list[i].normal;
        double sigma_l = J_nq * ptpl_list[i].plane_cov * J_nq.transpose();

        // M3D cov_lidar = calcBodyCov(ptpl_list[i].point, ranging_cov, angle_cov);
        M3D cov_lidar = ptpl_list[i].cov_lidar;
        M3D R_cov_Rt = s.rot * s.offset_R_L_I * cov_lidar * s.offset_R_L_I.conjugate() * s.rot.conjugate();
        // HACK 1. 因为是标量 所以求逆直接用1除
        // HACK 2. 不同分量的方差用加法来合成 因为公式(12)中的Sigma是对角阵，逐元素运算之后就是对角线上的项目相加
        double R_inv = 1.0 / (sigma_l + norm_vec.transpose() * R_cov_Rt * norm_vec);

        // 计算测量方差R并赋值 目前暂时使用固定值
        // ekfom_data.R(i) = 1.0 / LASER_POINT_COV;
        ekfom_data.R(i) = R_inv;
    }
}

void observation_hide_model_share(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{

    // =================================================================================================================
    // 用当前迭代轮最新的位姿估计值 将点云转换到world地图系
    vector<pointWithCov> pv_list;
    PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI);
    // FIXME stupid mistake 这里应该用迭代的最新线性化点
    // FIXME stupid mistake 这里应该用迭代的最新线性化点
    //    transformLidar(state_point, hide_feats_down_body, world_lidar);
    transformLidar(s, hide_feats_down_body, world_lidar);
    pv_list.resize(hide_feats_down_body->size());
    for (size_t i = 0; i < hide_feats_down_body->size(); i++)
    {
        // 保存body系和world系坐标
        pointWithCov pv;
        pv.point << hide_feats_down_body->points[i].x, hide_feats_down_body->points[i].y, hide_feats_down_body->points[i].z;
        pv.point_world << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
        // 计算lidar点的cov
        // 注意这个在每次迭代时是存在重复计算的 因为lidar系的点云covariance是不变的
        // M3D cov_lidar = calcBodyCov(pv.point, ranging_cov, angle_cov);
        M3D cov_lidar = var_down_body[i];
        // 将body系的var转换到world系
        M3D cov_world = transformLiDARCovToWorld(pv.point, hide_kf, cov_lidar);
        pv.cov = cov_world;
        pv.cov_lidar = cov_lidar;
        pv_list[i] = pv;
    }

    // ===============================================================================================================
    // 查找最近点 并构建residual
    double match_start = omp_get_wtime();
    std::vector<ptpl, Eigen::aligned_allocator<ptpl>> ptpl_list;
    std::vector<V3D, Eigen::aligned_allocator<V3D>> non_match_list;

    current_state_point = s;
    BuildResidualListOMP(hide_voxel_map, max_voxel_size, 3.0, max_layer, pv_list,
                         ptpl_list, non_match_list);
    double match_end = omp_get_wtime();
    // std::printf("Match Time: %f\n", match_end - match_start);

    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    // 根据匹配结果 设置H和R的维度
    // h_x是观测值对状态量的导数 TODO 为什么不加上状态量对状态量误差的导数？？？？像quaternion那本书？
    effct_feat_num = ptpl_list.size();
    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); // 23 因为点面距离只和位姿 外参有关 对其他状态量的导数都是0
    ekfom_data.h.resize(effct_feat_num);
    ekfom_data.R.resize(effct_feat_num, 1); // 把R作为向量 用的时候转换成diag
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < effct_feat_num; i++)
    {

        //        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(ptpl_list[i].point);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        //        const PointType &norm_p = corr_normvect->points[i];
        //        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);
        V3D norm_vec(ptpl_list[i].normal);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() * norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); // s.rot.conjugate()*norm_vec);
            // ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec.x(), norm_vec.y(), norm_vec.z(), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            // ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec.x(), norm_vec.y(), norm_vec.z(), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        //        ekfom_data.h(i) = -norm_p.intensity;
        float pd2 = norm_vec.x() * ptpl_list[i].point_world.x() + norm_vec.y() * ptpl_list[i].point_world.y() + norm_vec.z() * ptpl_list[i].point_world.z() + ptpl_list[i].d;
        ekfom_data.h(i) = -pd2;

        // norm_p中存了匹配的平面法向 还有点面距离
        // V3D point_world = s.rot * (s.offset_R_L_I * ptpl_list[i].point + s.offset_T_L_I) + s.pos;
        V3D point_world = ptpl_list[i].point_world;
        // /*** get the normal vector of closest surface/corner ***/
        Eigen::Matrix<double, 1, 6> J_nq;
        J_nq.block<1, 3>(0, 0) = point_world - ptpl_list[i].center;
        J_nq.block<1, 3>(0, 3) = -ptpl_list[i].normal;
        double sigma_l = J_nq * ptpl_list[i].plane_cov * J_nq.transpose();

        // M3D cov_lidar = calcBodyCov(ptpl_list[i].point, ranging_cov, angle_cov);
        M3D cov_lidar = ptpl_list[i].cov_lidar;
        M3D R_cov_Rt = s.rot * s.offset_R_L_I * cov_lidar * s.offset_R_L_I.conjugate() * s.rot.conjugate();
        // HACK 1. 因为是标量 所以求逆直接用1除
        // HACK 2. 不同分量的方差用加法来合成 因为公式(12)中的Sigma是对角阵，逐元素运算之后就是对角线上的项目相加
        double R_inv = 1.0 / (sigma_l + norm_vec.transpose() * R_cov_Rt * norm_vec);

        // 计算测量方差R并赋值 目前暂时使用固定值
        // ekfom_data.R(i) = 1.0 / LASER_POINT_COV;
        ekfom_data.R(i) = R_inv;
    }
}

/*** ROS subscribe initialization ***/
ros::Subscriber sub_pcl;
ros::Subscriber sub_imu;
ros::Publisher pubLaserCloudFull;
ros::Publisher pubLaserCloudFull_body;
ros::Publisher pubLaserCloudEffect;
ros::Publisher pubLaserCloudMap;
ros::Publisher pubOdomAftMapped;
ros::Publisher pubExtrinsic;
ros::Publisher pubKeyPath;
ros::Publisher pubPath;
ros::Publisher voxel_map_pub;
ros::Publisher marker_cov_pub;
ros::Publisher stats_pub;

// for Plane Map
bool init_map = false;

// statistic
double sum_optimize_time = 0, sum_update_time = 0;
std::fstream stat_latency("/tmp/latency.csv", std::ios::out);

double calculateErrorPoint(const PointCloudXYZI::Ptr &current_cloud, PointCloudXYZI::Ptr &last_cloud)
{
    if (last_cloud->size() == 0 || current_cloud->size() == 0)
        return 1.0f;
    // 使用 KD-Tree 找到重叠点
    pcl::KdTreeFLANN<pcl::PointXYZINormal> kdtree;
    kdtree.setInputCloud(current_cloud);

    int overlap_count = 0;
    float search_radius = filter_size_surf_min * 1.5;

    for (const auto &point : last_cloud->points)
    {
        std::vector<int> indices;
        std::vector<float> distances;
        if (kdtree.radiusSearch(point, search_radius, indices, distances) > 0)
        {
            overlap_count++;
        }
    }

    // 计算重叠比例
    float overlap_ratio = static_cast<float>(overlap_count) / (current_cloud->size() * 1.0f);
    return 1.0f - overlap_ratio;
}

void fusion_status_update(const PointCloudXYZI::Ptr &current_cloud, PointCloudXYZI::Ptr &last_cloud, bool is_success)
{
    if (!is_success)
    {
        return;
    }
    mean_division_error_points = calculateErrorPoint(feats_down_body, last_feats_down_body);

    double weight_kf = max_error_points - min(mean_division_error_points, max_error_points);
    double weight_icp = min(mean_division_error_points, max_error_points);
    Sophus::SE3d icp_state_point = icpServer->Poses().back().pose;

    Eigen::Vector3d translation = icp_state_point.translation();
    Eigen::Quaterniond quaternion(icp_state_point.unit_quaternion());

    // state_ikfom avg_st;

    state_point.pos[0] = (translation.x() * weight_icp + state_point.pos(0) * weight_kf) / max_error_points;
    state_point.pos[1] = (translation.y() * weight_icp + state_point.pos(1) * weight_kf) / max_error_points;
    state_point.pos[2] = (translation.z() * weight_icp + state_point.pos(2) * weight_kf) / max_error_points;
    state_point.rot.coeffs()[0] = (quaternion.x() * weight_icp + geoQuat.x * weight_kf) / max_error_points;
    state_point.rot.coeffs()[1] = (quaternion.y() * weight_icp + geoQuat.y * weight_kf) / max_error_points;
    state_point.rot.coeffs()[2] = (quaternion.z() * weight_icp + geoQuat.z * weight_kf) / max_error_points;
    state_point.rot.coeffs()[3] = (quaternion.w() * weight_icp + geoQuat.w * weight_kf) / max_error_points;

    kf.change_x(state_point);
    pcl::copyPointCloud(*feats_down_body, *last_feats_down_body);
}

bool isPoseStable(const state_ikfom &last_state_point, const state_ikfom &state_point,
                  double pos_threshold = 0.01, double rot_threshold = 0.1)
{
    // 计算位置的差异（欧几里得距离）
    Eigen::Vector3d pos1(last_state_point.pos(0), last_state_point.pos(1), last_state_point.pos(2));
    Eigen::Vector3d pos2(state_point.pos(0), state_point.pos(1), state_point.pos(2));
    Eigen::Vector3d pos_diff = pos1 - pos2;
    double pos_distance = pos_diff.norm();

    // 计算旋转的差异（四元数的角度差异）
    geometry_msgs::Quaternion q1;
    q1.x = state_point.rot.coeffs()[0];
    q1.y = state_point.rot.coeffs()[1];
    q1.z = state_point.rot.coeffs()[2];
    q1.w = state_point.rot.coeffs()[3];
    geometry_msgs::Quaternion q2;
    q2.x = last_state_point.rot.coeffs()[0];
    q2.y = last_state_point.rot.coeffs()[1];
    q2.z = last_state_point.rot.coeffs()[2];
    q2.w = last_state_point.rot.coeffs()[3];

    // 计算四元数的相对旋转（四元数点积）
    double dot_product = q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w;

    // 归一化点积范围 [-1, 1]
    dot_product = std::max(-1.0, std::min(1.0, dot_product));

    // 计算旋转角度差（弧度）
    double angle_difference = 2.0 * std::acos(dot_product);

    // 判断位置和旋转差异是否都小于给定阈值
    if (pos_distance < pos_threshold && std::fabs(angle_difference) < rot_threshold)
    {
        return true; // 位姿波动不大
    }
    else
    {
        return false; // 位姿波动较大
    }
}

void execute()
{
    // execute one step of state estimation and mapping
    if (flg_first_scan)
    {
        first_lidar_time = Measures.lidar_beg_time;
        p_imu->first_lidar_time = first_lidar_time;
        flg_first_scan = false;
        // continue;
        return;
    }

    double t_total_start = omp_get_wtime();

    double t_optimize_start = omp_get_wtime();
    p_imu->Process(Measures, kf, feats_undistort);
    state_point = kf.get_x();
    // pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

    if (feats_undistort->empty() || (feats_undistort == NULL))
    {
        ROS_WARN("No point, skip this scan!\n");
        // continue;
        return;
    }

    flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;
    // ===============================================================================================================
    // 第一帧 如果ekf初始化了 就初始化voxel地图
    if (flg_EKF_inited && !init_map)
    {
        PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI);
        transformLidar(state_point, feats_undistort, world_lidar);
        std::vector<pointWithCov> pv_list;

        // std::cout << kf.get_P() << std::endl;
        // 计算第一帧所有点的covariance 并用于构建初始地图
        for (size_t i = 0; i < world_lidar->size(); i++)
        {
            pointWithCov pv;
            pv.point << world_lidar->points[i].x, world_lidar->points[i].y,
                world_lidar->points[i].z;
            V3D point_this(feats_undistort->points[i].x,
                           feats_undistort->points[i].y,
                           feats_undistort->points[i].z);
            // if z=0, error will occur in calcBodyCov. To be solved
            if (point_this[2] == 0)
            {
                point_this[2] = 0.001;
            }
            M3D cov_lidar = calcBodyCov(point_this, ranging_cov, angle_cov);
            // 转换到world系
            M3D cov_world = transformLiDARCovToWorld(point_this, kf, cov_lidar);

            pv.cov = cov_world;
            pv.distance = calcPointDistance(pv.point);
            pv.intensity = world_lidar->points[i].intensity;
            pv_list.push_back(pv);
            // Eigen::Vector3d sigma_pv = pv.cov.diagonal();
            // sigma_pv[0] = sqrt(sigma_pv[0]);
            // sigma_pv[1] = sqrt(sigma_pv[1]);
            // sigma_pv[2] = sqrt(sigma_pv[2]);
        }

        // 当前state point 赋值
        current_state_point = kf.get_x();
        buildVoxelMap(pv_list, max_voxel_size, max_layer, layer_size,
                      max_points_size, max_points_size, min_eigen_value,
                      voxel_map);
        std::cout << "build voxel map" << std::endl;

        if (publish_voxel_map)
        {
            // current_state_point = kf.get_x();
            pubVoxelMap(voxel_map, publish_max_voxel_layer, voxel_map_pub);
            pubVocVoxelMap(voxel_map, publish_max_voxel_layer, marker_cov_pub);
            publish_frame_world(pubLaserCloudFull);
            publish_frame_body(pubLaserCloudFull_body);
        }
        init_map = true;
        // continue;
        return;
    }

    /*** downsample the feature points in a scan ***/
    downSizeFilterSurf.setInputCloud(feats_undistort);
    downSizeFilterSurf.filter(*feats_down_body);
    // std::cout << "feats size:" << feats_undistort->size()
    //                   << ", down size:" << feats_down_body->size() << std::endl;

    sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);

    feats_down_size = feats_down_body->points.size();
    // 由于点云的body var是一直不变的 因此提前计算 在迭代时可以复用
    var_down_body.clear();
    for (auto &pt : feats_down_body->points)
    {
        V3D point_this(pt.x, pt.y, pt.z);
        var_down_body.push_back(calcBodyCov(point_this, ranging_cov, angle_cov));
    }

    /*** ICP and iterated Kalman filter update ***/
    if (feats_down_size < 5)
    {
        ROS_WARN("Too few points (<5 points), skip this scan!\n");
        // continue;
        return;
    }
    // ===============================================================================================================
    // 开始迭代滤波
    /*** iterated state estimation ***/
    double solve_H_time = 0;
    kf.update_iterated_dyn_share_diagonal();
    //            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
    double t_optimize_end = omp_get_wtime();
    sum_optimize_time += t_optimize_end - t_optimize_start;

    state_point = kf.get_x();

    //    // HACK 强行重置ba bg
    //    state_point.ba.setZero();
    //    state_point.bg.setZero();
    //    kf.change_x(state_point);

    euler_cur = SO3ToEuler(state_point.rot);
    // pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
    geoQuat.x = state_point.rot.coeffs()[0];
    geoQuat.y = state_point.rot.coeffs()[1];
    geoQuat.z = state_point.rot.coeffs()[2];
    geoQuat.w = state_point.rot.coeffs()[3];

    // if (isPoseStable(last_state_point, state_point))
    // {

    //     double weight_1 = 0.5; // kf的权重
    //     double weight_2 = 0.5; // hide_kf的权重
    //     state_point.pos[0] = (last_state_point.pos(0) * weight_1 + state_point.pos(0) * weight_2);
    //     state_point.pos[1] = (last_state_point.pos(1) * weight_1 + state_point.pos(1) * weight_2);
    //     state_point.pos[2] = (last_state_point.pos(2) * weight_1 + state_point.pos(2) * weight_2);
    //     state_point.rot.coeffs()[0] = (last_state_point.rot.coeffs()[0] * weight_1 + state_point.rot.coeffs()[0] * weight_2);
    //     state_point.rot.coeffs()[1] = (last_state_point.rot.coeffs()[1] * weight_1 + state_point.rot.coeffs()[1] * weight_2);
    //     state_point.rot.coeffs()[2] = (last_state_point.rot.coeffs()[2] * weight_1 + state_point.rot.coeffs()[2] * weight_2);
    //     state_point.rot.coeffs()[3] = (last_state_point.rot.coeffs()[3] * weight_1 + state_point.rot.coeffs()[3] * weight_2);
    //     kf.change_x(state_point);
    // }

    last_state_point = kf.get_x();

    // ===============================================================================================================
    // 更新地图
    /*** add the points to the voxel map ***/
    double t_update_start = omp_get_wtime();
    // 用最新的状态估计将点及点的covariance转换到world系
    std::vector<pointWithCov> pv_list;
    PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI);
    transformLidar(state_point, feats_down_body, world_lidar);
    for (size_t i = 0; i < feats_down_body->size(); i++)
    {
        // 保存body系和world系坐标
        pointWithCov pv;
        pv.point << feats_down_body->points[i].x, feats_down_body->points[i].y, feats_down_body->points[i].z;
        // 计算lidar点的cov
        // FIXME 这里错误的使用世界系的点来calcBodyCov时 反倒在某些seq（比如hilti2022的03 15）上效果更好 需要考虑是不是init_plane时使用更大的cov更好
        // 注意这个在每次迭代时是存在重复计算的 因为lidar系的点云covariance是不变的
        // M3D cov_lidar = calcBodyCov(pv.point, ranging_cov, angle_cov);
        M3D cov_lidar = var_down_body[i];
        // 将body系的var转换到world系
        M3D cov_world = transformLiDARCovToWorld(pv.point, kf, cov_lidar);

        // 最终updateVoxelMap需要用的是world系的point
        pv.cov = cov_world;
        pv.point << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
        pv.distance = calcPointDistance(pv.point);
        pv.intensity = world_lidar->points[i].intensity;
        pv_list.push_back(pv);
    }

    // 当前state point 赋值
    current_state_point = kf.get_x();
    std::sort(pv_list.begin(), pv_list.end(), var_contrast);
    updateVoxelMapOMP(pv_list, max_voxel_size, max_layer, layer_size,
                      max_points_size, max_points_size, min_eigen_value,
                      voxel_map);
    double t_update_end = omp_get_wtime();
    sum_update_time += t_update_end - t_update_start;
    scan_index++;

    // SaveIPointCloudToPLY(feats_undistort, "/home/guowenwu/workspace/Indoor_SLAM/gb_ws/output_cloud_addI_rgb.ply");
    if (icp_en)
    {
        set_posestamp(msg_body_pose);
        bool is_success = icpServer->RegisterFrame(feats_down_body, msg_body_pose, ros::Time().fromSec(lidar_end_time), voxel_map);
        fusion_status_update(feats_down_body, last_feats_down_body, is_success);
    }

    double t_total_end = omp_get_wtime();
    // ===============================================================================================================
    // 可视化相关
    /******* Publish odometry *******/
    double t_vis_start = omp_get_wtime();
    // publish_odometry(pubOdomAftMapped);
    //
    //            /*** add the feature points to map kdtree ***/
    //            map_incremental();
    //
    // TODO skip first few frames
    // TODO downsample dense point clouds
    /******* Publish points *******/
    if (path_en)
        publish_path(pubPath);
    if (scan_pub_en)
        publish_frame_world(pubLaserCloudFull);
    if (scan_pub_en && scan_body_pub_en)
        publish_frame_body(pubLaserCloudFull_body);
    if (publish_voxel_map && pubLaserCloudMap.getNumSubscribers() > 0)
    {
        pubColoredVoxels(voxel_map, publish_max_voxel_layer, pubLaserCloudMap, lidar_end_time);
        pubVocVoxelMap(voxel_map, publish_max_voxel_layer, marker_cov_pub);
    }

    if (publish_voxel_map && marker_cov_pub.getNumSubscribers() > 0)
    {
        pubVocVoxelMap(voxel_map, publish_max_voxel_layer, marker_cov_pub);
    }
    if (publish_voxel_map && voxel_map_pub.getNumSubscribers() > 0)
    {
        pubVoxelMap(voxel_map, publish_max_voxel_layer, voxel_map_pub);
        // pubVocVoxelMap(voxel_map, publish_max_voxel_layer, marker_cov_pub);
    }
    // pubVocVoxelMap(voxel_map, publish_max_voxel_layer, marker_cov_pub);
    double t_vis_end = omp_get_wtime();
    // nav_msgs::Odometry stat_msg;
    // stat_msg.header = odomAftMapped.header;
    // stat_msg.pose.pose.position.x = t_optimize_end - t_optimize_start;
    // stat_msg.pose.pose.position.y = t_update_end - t_update_start;
    // stats_pub.publish(stat_msg);
    // publish_effect_world(pubLaserCloudEffect);
    // publish_map(pubLaserCloudMap);
    //
    // std::printf("v: %.2f %.2f %.2f BA: %.4f %.4f %.4f   BG: %.4f %.4f %.4f   g: %.4f %.4f %.4f\n",
    //             kf.get_x().vel.x(),kf.get_x().vel.y(),kf.get_x().vel.z(),
    //             kf.get_x().ba.x(),kf.get_x().ba.y(),kf.get_x().ba.z(),
    //             kf.get_x().bg.x(),kf.get_x().bg.y(),kf.get_x().bg.z(),
    //             kf.get_x().grav.get_vect().x(), kf.get_x().grav.get_vect().y(), kf.get_x().grav.get_vect().z()
    // );

    // mean_raw_points = mean_raw_points * (scan_index - 1) / scan_index +
    //                   (double) (feats_undistort->size()) / scan_index;
    // mean_ds_points = mean_ds_points * (scan_index - 1) / scan_index +
    //                  (double) (feats_down_body->size()) / scan_index;
    // mean_effect_points = mean_effect_points * (scan_index - 1) / scan_index +
    //                      (double) effct_feat_num / scan_index;

    // undistort_time_mean = undistort_time_mean * (scan_index - 1) / scan_index +
    //                       (undistort_time) / scan_index;
    // down_sample_time_mean =
    //         down_sample_time_mean * (scan_index - 1) / scan_index +
    //         (t_downsample) / scan_index;
    // calc_cov_time_mean = calc_cov_time_mean * (scan_index - 1) / scan_index +
    //                      (calc_point_cov_time) / scan_index;
    // scan_match_time_mean =
    //         scan_match_time_mean * (scan_index - 1) / scan_index +
    //         (scan_match_time) / scan_index;
    // ekf_solve_time_mean = ekf_solve_time_mean * (scan_index - 1) / scan_index +
    //                       (solve_time) / scan_index;
    // map_update_time_mean =
    //         map_update_time_mean * (scan_index - 1) / scan_index +
    //         (map_incremental_time) / scan_index;

    // aver_time_consu = aver_time_consu * (scan_index - 1) / scan_index +
    //                   (total_time) / scan_index;

    // time_log_counter++;
    // cout << "pos:" << state.pos_end.transpose() << endl;
    // cout << "[ Time ]: "
    //      << "average undistort: " << undistort_time_mean << std::endl;
    // cout << "[ Time ]: "
    //      << "average down sample: " << down_sample_time_mean << std::endl;
    // cout << "[ Time ]: "
    //      << "average calc cov: " << calc_cov_time_mean << std::endl;
    // cout << "[ Time ]: "
    //      << "average scan match: " << scan_match_time_mean << std::endl;
    // cout << "[ Time ]: "
    //      << "average solve: " << ekf_solve_time_mean << std::endl;
    // cout << "[ Time ]: "
    //      << "average map incremental: " << map_update_time_mean << std::endl;
    // cout << "[ Time ]: "
    //      << " average total " << aver_time_consu << endl;
    // cout << "--------------------------------------------" << endl;

    // std::printf("Mean Latency: %.3fs |  Mean Topt: %.5fs   Tu: %.5fs   | Cur Topt: %.5fs   地图更新时间 : %.5fs   可视化时间: %.5fs\n",
    //             (sum_optimize_time + sum_update_time) / scan_index + (t_vis_end - t_vis_start),
    //             sum_optimize_time / scan_index, sum_update_time / scan_index,
    //             t_optimize_end - t_optimize_start,
    //             t_update_end - t_update_start,
    //             t_vis_end - t_vis_start);

    total_time_ += t_total_end - t_total_start;
    // total_num_ += feats_undistort->size();
    std::printf("总时间 %.5f\n", total_time_);
    stat_latency
        << lidar_end_time << ", "
        << t_optimize_end - t_optimize_start << ", "
        << t_update_end - t_update_start << ", "
        << t_vis_end - t_vis_start << ", "
        << std::endl;
}

void savePathAsTUM(const std::vector<gb_icp_ros::PoseData> &path, const std::string &filename)
{
    // 打开文件
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    // 遍历路径，输出每个位姿到文件
    for (size_t i = 0; i < path.size(); ++i)
    {
        const Sophus::SE3d &pose = path[i].pose;

        // 假设时间戳可以用索引代替
        double timestamp = path[i].stamp.toSec();

        // 获取平移向量和旋转四元数
        Eigen::Vector3d translation = pose.translation();
        Eigen::Quaterniond quaternion(pose.unit_quaternion());

        // 保存到文件，格式：timestamp tx ty tz qx qy qz qw
        file << std::fixed << timestamp << " "
             << translation.x() << " " << translation.y() << " " << translation.z() << " "
             << quaternion.x() << " " << quaternion.y() << " " << quaternion.z() << " " << quaternion.w()
             << std::endl;
    }

    // 关闭文件
    file.close();
    std::cout << "Path saved to " << filename << " in TUM format." << std::endl;
}
void savePathAsTUM(const nav_msgs::Path &path, const std::string &filename)
{
    // 打开文件，准备写入
    std::ofstream file(filename);

    // 检查文件是否成功打开
    if (!file.is_open())
    {
        ROS_ERROR("Unable to open file: %s", filename.c_str());
        return;
    }

    // 遍历路径中的每一个PoseStamped
    for (const auto &poseStamped : path.poses)
    {
        // 提取时间戳
        double timestamp = poseStamped.header.stamp.toSec();

        // 提取位姿信息：平移部分 (tx, ty, tz)
        const auto &position = poseStamped.pose.position;
        double tx = position.x;
        double ty = position.y;
        double tz = position.z;

        // 提取姿态信息：四元数部分 (qx, qy, qz, qw)
        const auto &orientation = poseStamped.pose.orientation;
        double qx = orientation.x;
        double qy = orientation.y;
        double qz = orientation.z;
        double qw = orientation.w;
        // double qx = 0.;
        // double qy = 0.;
        // double qz = 0.;
        // double qw = 1.;

        // 写入文件，格式: timestamp tx ty tz qx qy qz qw
        file << std::fixed << timestamp << " "
             << tx << " " << ty << " " << tz << " "
             << qx << " " << qy << " " << qz << " " << qw << std::endl;
    }

    // 关闭文件
    file.close();
    ROS_INFO("Path saved to %s", filename.c_str());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<double>("time_offset", lidar_time_offset, 0.0);

    nh.param<bool>("publish/path_en", path_en, true);
    nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en", dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, true);
    nh.param<double>("publish/intensity_th", intensity_th, 1.0);

    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<bool>("common/icp_en", icp_en, false);

    // mapping algorithm params
    nh.param<float>("mapping/det_range", DET_RANGE, 300.f);
    nh.param<int>("mapping/max_iteration", NUM_MAX_ITERATIONS, 4);
    nh.param<int>("mapping/max_points_size", max_points_size, 100);
    nh.param<int>("mapping/max_cov_points_size", max_cov_points_size, 100);
    nh.param<vector<double>>("mapping/layer_point_size", layer_point_size, vector<double>());
    nh.param<int>("mapping/max_layer", max_layer, 2);
    nh.param<double>("mapping/voxel_size", max_voxel_size, 1.0);
    nh.param<double>("mapping/down_sample_size", filter_size_surf_min, 0.5);
    std::cout << "filter_size_surf_min:" << filter_size_surf_min << std::endl;
    nh.param<double>("mapping/plannar_threshold", min_eigen_value, 0.01);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    nh.param<bool>("mapping/encoder_fusion_en", encoder_fusion_en, false);
    nh.param<vector<double>>("mapping/extrinsic_T_encoder_lidar", extrinT_encoder, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R_encoder_lidar", extrinR_encoder, vector<double>());
    nh.param<double>("mapping/encoder_offset_deg", encoder_zeropoint_offset_deg, 0);
    nh.param<bool>("mapping/adaptive_voxelization", adaptive_voxelization, false);
    nh.param<vector<double>>("mapping/adaptive_threshold", adaptive_threshold, vector<double>({1000}));
    nh.param<vector<double>>("mapping/adaptive_multiple_factor", adaptive_multiple_factor, vector<double>({2.0}));
    nh.param<bool>("mapping/init_gravity_with_pose", init_gravity_with_pose, false);
    nh.param<float>("mapping/merge_distance_threshold", MERGE_DISTANCE_THRESHOLD, 0.03f);
    nh.param<float>("mapping/merge_bias_threshold", MERGE_BIAS_THRESHOLD, 0.8f);
    nh.param<int>("mapping/merge_layers", MERGE_LAYERS, 5);
    nh.param<bool>("mapping/merge_mode", merge_mode, false);

    // noise model params
    nh.param<double>("noise_model/ranging_cov", ranging_cov, 0.02);
    nh.param<double>("noise_model/angle_cov", angle_cov, 0.05);
    nh.param<double>("noise_model/gyr_cov", gyr_cov, 0.1);
    nh.param<double>("noise_model/acc_cov", acc_cov, 0.1);
    nh.param<double>("noise_model/b_gyr_cov", b_gyr_cov, 0.0001);
    nh.param<double>("noise_model/b_acc_cov", b_acc_cov, 0.0001);

    // visualization params
    nh.param<bool>("publish/pub_voxel_map", publish_voxel_map, true);
    nh.param<int>("publish/publish_max_voxel_layer", publish_max_voxel_layer, 0);
    nh.param<int>("publish/publish_downsample_points", publish_downsample_points, 1000000);
    nh.param<int>("publish/publish_dense_skip", publish_dense_skip, 1);
    nh.param<int>("publish/publish_path_skip", publish_path_skip, 1);
    nh.param<double>("publish/publish_limit_z", publish_limit_z, 1000000.0);

    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 1);
    nh.param<bool>("preprocess/feature_extract_enable", p_pre->feature_enabled, false);
    cout << "p_pre->lidar_type " << p_pre->lidar_type << endl;
    // 全是[5,5,5,5,5]
    for (int i = 0; i < layer_point_size.size(); i++)
    {
        layer_size.push_back(layer_point_size[i]);
    }
    // path初始化，并指定坐标系
    path.header.stamp = ros::Time::now();
    path.header.frame_id = "camera_init";

    keyPath.header.stamp = ros::Time::now();
    keyPath.header.frame_id = "camera_init";

    /*** variables definition ***/
    int effect_feat_num = 0, scan_index = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    scan_index = 0;
    // 没有使用到
    _featsArray.reset(new PointCloudXYZI());

    // 初始化两个数组
    std::fill(std::begin(point_selected_surf), std::end(point_selected_surf), true);
    std::fill(std::begin(res_last), std::end(res_last), -1000.0f);
    // 初始化下采样滤波器
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    // memset(point_selected_surf, true, sizeof(point_selected_surf));
    // memset(res_last, -1000.0f, sizeof(res_last));
    // 初始化IMU的RT矩阵和
    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
    // 设置 IMU 的外部参数，包括位置（平移向量）和方向（旋转矩阵）
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    // 设置陀螺仪的协方差值。V3D 可能是一个三维向量类型，用于表示陀螺仪的噪声特性。
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    // 设置加速度计的协方差值，方式与陀螺仪相似。
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    // 设置陀螺仪偏置的协方差值，反映陀螺仪偏置的不确定性。
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    // 设置加速度计偏置的协方差值，反映加速度计偏置的不确定性
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    // 是否使用当前姿态来初始化重力
    p_imu->set_init_gravity_with_pose(init_gravity_with_pose);

    // 设置 IMU 的外部参数，包括位置（平移向量）和方向（旋转矩阵）
    hide_p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    // 设置陀螺仪的协方差值。V3D 可能是一个三维向量类型，用于表示陀螺仪的噪声特性。
    hide_p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    // 设置加速度计的协方差值，方式与陀螺仪相似。
    hide_p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    // 设置陀螺仪偏置的协方差值，反映陀螺仪偏置的不确定性。
    hide_p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    // 设置加速度计偏置的协方差值，反映加速度计偏置的不确定性
    hide_p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    // 是否使用当前姿态来初始化重力
    hide_p_imu->set_init_gravity_with_pose(init_gravity_with_pose);

    double epsi[23] = {0.001};
    fill(epsi, epsi + 23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, observation_model_share, NUM_MAX_ITERATIONS, epsi);
    hide_kf.init_dyn_share(get_f, df_dx, df_dw, observation_hide_model_share, NUM_MAX_ITERATIONS, epsi);

    if (icp_en)
    {
        icpServer = new gb_icp_ros::IcpServer(nh);
    }

    /*** ROS subscribe initialization ***/
    sub_pcl = p_pre->lidar_type == AVIA ? nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
    pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);
    pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);
    pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    pubExtrinsic = nh.advertise<nav_msgs::Odometry>("/Extrinsic", 100000);
    pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);
    pubKeyPath = nh.advertise<nav_msgs::Path>("/key_path", 100000);
    voxel_map_pub = nh.advertise<visualization_msgs::MarkerArray>("/planes", 10000);
    stats_pub = nh.advertise<nav_msgs::Odometry>("/stats", 10000);
    marker_cov_pub = nh.advertise<visualization_msgs::Marker>("/cov_marker", 100000);

    //------------------------------------------------------------------------------------------------------
    // statistic
    stat_latency.setf(ios::fixed);
    stat_latency.precision(10); // 精度为输出小数点后5位

    ////------------------------------------------------------------------------------------------------------
    //    // 用rosbag读取
    //    signal(SIGINT, SigHandle);
    //
    //    std::string bag_file = "/tmp/2024-03-05-17-21-56.bag";
    //    rosbag::Bag bag;
    //    try {
    //        bag.open(bag_file, rosbag::bagmode::Read);
    //    } catch (const rosbag::BagException& e) {
    //        ROS_ERROR("Could not open bag file: %s", e.what());
    //        return -1;
    //    }
    //
    //    std::vector<std::string> topics;
    //    topics.push_back(std::string("/velodyne_points"));
    //    topics.push_back(std::string("/imu/data"));
    //
    //    rosbag::View view(bag, rosbag::TopicQuery(topics));
    //        foreach(rosbag::MessageInstance const m, view) {
    //            // 中途退出
    //            if (flg_exit) break;
    //
    //            sensor_msgs::PointCloud2ConstPtr lidar_msg = m.instantiate<sensor_msgs::PointCloud2>();
    //            if (lidar_msg != NULL) {
    //                standard_pcl_cbk(lidar_msg);
    //            }
    //
    //            sensor_msgs::ImuConstPtr imu_msg = m.instantiate<sensor_msgs::Imu>();
    //            if (imu_msg != NULL) {
    //                imu_cbk(imu_msg);
    //            }
    //
    //            if (sync_packages(Measures)) {
    //                // execute one step
    //                execute();
    //            }
    //        }
    //    bag.close();

    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    float Similarity_rate = 1.0f;
    int index = 0;
    while (status)
    {
        if (flg_exit)
            break;
        ros::spinOnce();
        if (sync_packages(Measures))
        {
            execute();
        }

        status = ros::ok();
        rate.sleep();
    }
    stat_latency.close();
    savePathAsTUM(path, "/home/guowenwu/workspace/Indoor_SLAM/gb_ws/resutl.txt");
    savePathAsTUM(icpServer->Poses(), "/home/guowenwu/workspace/Indoor_SLAM/gb_ws/key_resutl.txt");
    return 0;
}
