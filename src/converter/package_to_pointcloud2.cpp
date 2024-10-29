#include <ros/ros.h>
#include <velodyne_msgs/VelodyneScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <velodyne_pointcloud/convert.h>
#include <velodyne_pointcloud/rawdata.h>

class VelodyneScanToPointCloud
{
public:
    VelodyneScanToPointCloud(ros::NodeHandle &nh)
    {
        // 设置 Velodyne 数据转换器
        rawdata_.setup(nh); // 初始化 Velodyne 原始数据配置
        pointcloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_points", 10);
        scan_sub_ = nh.subscribe("/velodyne_packets", 10, &VelodyneScanToPointCloud::scanCallback, this);
    }

private:
    void scanCallback(const velodyne_msgs::VelodyneScan::ConstPtr &scan_msg)
    {
        // 存储点云的 PCL 对象
        pcl::PointCloud<pcl::PointXYZI> pcl_cloud;

        // 逐个包解码并加入点云
        for (const auto &packet : scan_msg->packets)
        {
            rawdata_.unpack(packet, pcl_cloud); // 解码单个数据包
        }

        // 转换 PCL 点云到 ROS 点云消息
        sensor_msgs::PointCloud2 ros_cloud;
        pcl::toROSMsg(pcl_cloud, ros_cloud);
        ros_cloud.header.stamp = scan_msg->header.stamp;
        ros_cloud.header.frame_id = "velodyne";

        // 发布解码后的点云消息
        pointcloud_pub_.publish(ros_cloud);
    }

    ros::Publisher pointcloud_pub_;
    ros::Subscriber scan_sub_;
    velodyne_rawdata::RawData rawdata_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "velodyne_scan_to_pointcloud");
    ros::NodeHandle nh;
    VelodyneScanToPointCloud converter(nh);
    ros::spin();
    return 0;
}
