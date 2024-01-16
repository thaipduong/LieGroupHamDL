// This implementation is based on code from the following repositories:
// https://github.com/Jaeyoung-Lim/mavros_controllers, original code was authored by: Jaeyoung Lim
// https://github.com/KumarRobotics/kr_mav_control, original code was authored by: Kartik Mohta

#ifndef ERL_ENERGY_BASED_CONTROL_HPP
#define ERL_ENERGY_BASED_CONTROL_HPP

#include <torch/script.h>
#include <torch/torch.h>
#include <ros/ros.h>
#include <sophus/se3.hpp>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <nav_msgs/Odometry.h>
#include <mavros_msgs/AttitudeTarget.h>
#include <kr_mav_msgs/PositionCommand.h>
#include <cmath>

namespace ERLControl{
    
    class EnergyBasedControl{
        public:
            void init(Eigen::Vector3d Kp, Eigen::Vector3d Kv, Eigen::Vector3d KR, Eigen::Vector3d Kw,
                      std::string Gnet_path, std::string Dvnet_path, std::string Dwnet_path);
            void setCurrentState(nav_msgs::Odometry &odom);
            void setTargetState(const kr_mav_msgs::PositionCommand);
            double getYawAngleFromQuat(Eigen::Vector4d &quat);
            mavros_msgs::AttitudeTarget getAttitudeTargetMsg(std::string frame_id, ros::Time stamp);
            
            void compute(const bool &start);
            // Utils 
            Eigen::Vector4d rot2Quaternion(const Eigen::Matrix3d &R); 
            Eigen::Matrix3d quat2RotMatrix(const Eigen::Vector4d &q);
            Eigen::Vector3d vee_map(const Eigen::Matrix3d &m);
            Eigen::Matrix3d hat_map(const Eigen::Vector3d &m);

        private:
            Eigen::Vector3d position_, velocity_, angular_velocity_, acceleration_; // velocities in body frame
            Eigen::Vector3d target_position_, target_velocity_, target_acceleration_; // world frame
            double target_yaw_, target_yaw_dot = 0.;
            double max_tilt_angle = M_PI*40/180;
            Eigen::Matrix3d rotmat_;
            Eigen::Vector4d cmdBodyRate_;  //{wx, wy, wz, Thrust}
            Eigen::Vector4d des_rotmat_;
            Eigen::Vector3d des_thrust_;
            Eigen::Quaterniond q_des_;
            Eigen::Vector4d control;

            // Control Gains 
            Eigen::Vector3d Kp_, Kv_, KR_, Kw_, Ki_;
            Eigen::Vector3d pos_int_ = Eigen::Vector3d::Zero();

            // Learned model
            at::Tensor atOutputg, atOutputV, atOutputdV, atOutputM1, atOutputM2, atoutputDv, atoutputDw, atoutputChuong;
            Eigen::Matrix<double, 6, 4> outputg;
            Eigen::Matrix<double, 3, 1> ouputdVdp, ouputdVdR1, ouputdVdR2, ouputdVdR3;
            Eigen::Matrix3d outputM1, outputM2, outputDv, outputDw;
            torch::jit::script::Module module_gnet;
            torch::jit::script::Module module_Dvnet;
            torch::jit::script::Module module_Dwnet;
            torch::jit::script::Module module_Vnet;
            torch::jit::script::Module module_Chuongnet;

    };
}

#endif
