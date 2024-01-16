#ifndef EIGEN_CONVERSION_HPP
#define EIGEN_CONVERSION_HPP

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Vector3.h>
#include <Eigen/Geometry>

namespace ERLControl{
    
    // conversion for point and vectors
    geometry_msgs::Point convert(const Eigen::Vector3d& vec){
        geometry_msgs::Point p;

        p.x = vec.x();
        p.y = vec.y();
        p.z = vec.z();
        
        return p;
    }

    Eigen::Vector3d convert(const geometry_msgs::Point& vec){
        return Eigen::Vector3d(vec.x, vec.y, vec.z);
    }
    
    Eigen::Vector3d convert(const geometry_msgs::Vector3& vec){
        return Eigen::Vector3d(vec.x, vec.y, vec.z);
    }

    // conversiton for quaternions
    geometry_msgs::Quaternion convert(const Eigen::Quaterniond& quat){
        geometry_msgs::Quaternion q;

        q.w = quat.w();
        q.x = quat.x();
        q.y = quat.y();
        q.z = quat.z();
        
        return q;
    }

    Eigen::Quaterniond convert(const geometry_msgs::Quaternion& quat){
        return Eigen::Quaterniond(quat.w, quat.x, quat.y, quat.z);
    }

}

#endif
