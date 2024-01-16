#ifndef SOPHUS_CONVERSION_HPP
#define SOPHUS_CONVERSION_HPP

#include "EigenConversion.hpp"
#include <sophus/se3.hpp>

namespace ERLControl{

    
    geometry_msgs::Pose convert(const Sophus::SE3d& se3){
        geometry_msgs::Pose pose;

        pose.position = convert(se3.translation());
        pose.orientation = convert(se3.so3().unit_quaternion());
        
        return pose;
    }

    Sophus::SE3d convert(const geometry_msgs::Pose& pose){
        Sophus::SE3d se3;

        se3.translation() = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
        se3.setQuaternion(convert(pose.orientation));

        return se3;
    }

}

#endif