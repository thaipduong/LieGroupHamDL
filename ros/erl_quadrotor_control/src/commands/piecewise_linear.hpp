#ifndef ERL_CONTROL_PIECEWISE_LINEAR_HPP
#define ERL_CONTROL_PIECEWISE_LINEAR_HPP

#include <ros/ros.h>
#include "command.hpp"
#include "math.h"

#include "../PX4_quadcoptor.hpp"

#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Pose.h>

#include <sophus/se3.hpp>
#include <sophus/interpolate.hpp>

namespace ERLControl{

    class PiecewiseLinear : public Command {
        public:
        PiecewiseLinear(Sophus::Vector3d aTarget_1, Sophus::Vector3d aTarget_2, Sophus::Vector3d aTarget_3, Sophus::Vector3d aTarget_4,
         double aYawTarget, double aMoveTime, double aTolerance, bool yaw_interpolation = false) :
         target_1(aTarget_1), target_2(aTarget_2), target_3(aTarget_3), target_4(aTarget_4),
         yaw_target(aYawTarget), move_time(aMoveTime), tolerance(aTolerance), timeout(aMoveTime + 5.0), interpolate_yaw(yaw_interpolation) {}

        void initialize() override {
            start_time = ros::Time::now();
            start_pose = quad->getLocalPose();
            
            // Set Euler angles (in radians)
            double roll = 0.0;
            double pitch = 0.0;
            double yaw = yaw_target;

            Eigen::Quaterniond q;
            q = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) *
                Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());

            target_pose.translation() = target_1;
            target_pose.setQuaternion(q);
        }

        void execute() override {
            double T = move_time / 4; 
            auto t = (ros::Time::now() - start_time).toSec();
            p = t / move_time;
            if (p >= 1.0) {p = 1.0;}

            // Target pose 
            if (p < 0.25) {
              target_pose.translation() = target_1;
              p_waypoint = p / 0.25;
            }
            else if (p < 0.5) {
              start_pose.translation() = target_1;
              target_pose.translation() = target_2;
              p_waypoint = (p - 0.25) / 0.25;
            }
            else if (p < 0.75) {
              start_pose.translation() = target_2;
              target_pose.translation() = target_3;
              p_waypoint = (p - 0.5) / 0.25;
            }
            else {
              start_pose.translation() = target_3;
              target_pose.translation() = target_4;
              p_waypoint = (p - 0.75) / 0.25;
            }

            double yaw_target_t = yaw_target;
            
            current_target_local = Sophus::interpolate(start_pose, target_pose, p_waypoint);
            target_position = current_target_local.translation();
            Eigen::Matrix3d R = current_target_local.rotationMatrix();
            
            if (p >= 1.0) {
              target_velocity = Eigen::Vector3d(0.0, 0.0, 0.0);
            }
            else {
              target_velocity = (target_pose.translation() - start_pose.translation()) / T;
            }

            yaw_target_t = 0.0; 
            yaw_target_dot = 0.0; 

            target_acceleration = Eigen::Vector3d(0.0, 0.0, 0.0);
            target_jerk = Eigen::Vector3d(0.0, 0.0, 0.0);
            //std::cout << target_position << std::endl;
            Sophus::SE3d currentTargetPose;
            currentTargetPose.translation() = target_position;
            currentTargetPose.setQuaternion(target_pose.unit_quaternion()); // why unit quarternion if we have a yaw target here?

            //quad->setTargetPose(currentTargetPose);
            quad->setTargetState(target_position, target_velocity, target_acceleration, target_jerk, yaw_target_t, yaw_target_dot); // The target position is in local frame, not world frame??
        }

        bool isFinished() override {
            double delta2 = 0;
            auto currentPose = quad->getLocalPose();
            return (ros::Time::now() - start_time) > ros::Duration(timeout);
        };

        std::string getName() override {
            return "piecewise_linear";
        }



        private:
        Sophus::Vector3d        target_1;
        Sophus::Vector3d        target_2;
        Sophus::Vector3d        target_3;
        Sophus::Vector3d        target_4;
        double                  yaw_target = 0.;
        double                  yaw_target_dot = 0.;
        double                  tolerance;
        double                  move_time;
        double                  timeout;
        double p;
        double p_waypoint;
        bool                    interpolate_yaw = false;

        ros::Time               start_time;
        Sophus::SE3d            start_pose;
        Sophus::SE3d            target_pose;
        Sophus::SE3d            current_target_local;
        Sophus::Vector3d        target_position;
        Sophus::Vector3d        target_velocity;
        Sophus::Vector3d        target_acceleration;
        Sophus::Vector3d        target_jerk; 
    };

}

#endif
