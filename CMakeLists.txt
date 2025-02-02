cmake_minimum_required(VERSION 3.10)
project(state_estimation)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -pthread")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")


# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/include/c++/9 -pthread -std=c++11 -std=c++14 -std=c++17 -fexceptions")
set(CMAKE_CXX_COMPILER g++) 
set(CMAKE_CXX_EXTENSIONS OFF)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -std=c++0x -std=c++17 -fexceptions")
set(livox_ros_driver_DIR "/home/guowenwu/workspace/Indoor_SLAM/livox_driver_ws/devel/.private/livox_ros_driver/share/livox_ros_driver/cmake/")
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.5)

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)|(aarch64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 14)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=12)
    message("core for MP: 14")
  elseif(N GREATER 10)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=8)
    message("core for MP: 8")
  elseif(N GREATER 7)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=6)
    message("core for MP: 6")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
    message("core for MP: 3")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(Boost REQUIRED COMPONENTS timer)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_package(PythonLibs REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  livox_ros_driver
  message_generation
  eigen_conversions
)
find_package(Sophus REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
#find_package(GTSAM REQUIRED QUIET)

message(Eigen: ${EIGEN3_INCLUDE_DIR})

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  include)

#add_message_files(
#  FILES
#  Pose6D.msg
#)

generate_messages(
 DEPENDENCIES
 geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS
)

add_executable(${PROJECT_NAME}_node src/voxelMapping.cpp src/preprocess.cpp src/icp/GBICP.cpp src/icp/icpServer.cpp src/util/voxel_map_util.cpp src/util/common_lib.cpp src/util/use-ikfom.cpp)
target_link_libraries(${PROJECT_NAME}_node  Sophus::Sophus   ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES}  ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME}_node PRIVATE ${PYTHON_INCLUDE_DIRS})

add_executable(pandar_to_velodyne src/converter/pandar_to_velodyne.cpp)
target_link_libraries(pandar_to_velodyne ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES}  ${OpenCV_LIBS})
target_include_directories(pandar_to_velodyne PRIVATE ${PYTHON_INCLUDE_DIRS})

# add_executable(rs_to_velodyne src/converter/rs_to_velodyne.cpp)
# target_link_libraries(rs_to_velodyne ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES})
# target_include_directories(rs_to_velodyne PRIVATE ${PYTHON_INCLUDE_DIRS})

# add_executable(package_to_pointcloud2 src/converter/package_to_pointcloud2.cpp)
# target_link_libraries(package_to_pointcloud2 ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES})
# target_include_directories(package_to_pointcloud2 PRIVATE ${PYTHON_INCLUDE_DIRS})
