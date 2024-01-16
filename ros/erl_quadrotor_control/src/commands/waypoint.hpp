#ifndef ERL_CONTROL_WAYPOINT_HPP
#define ERL_CONTROL_WAYPOINT_HPP

#include <ros/ros.h>
#include "command.hpp"
#include "math.h"

#include "../PX4_quadcoptor.hpp"

#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Pose.h>

#include <sophus/se3.hpp>
#include <sophus/interpolate.hpp>

namespace ERLControl{

    class WayPoint : public Command {
        public:
        WayPoint(Sophus::Vector3d aTarget, double aYawTarget, double aMoveTime, double aTolerance, double aTimeout = 20.0, bool yaw_interpolation = false) :
         target(aTarget), yaw_target(aYawTarget), move_time(aMoveTime), tolerance(aTolerance), timeout(aTimeout), interpolate_yaw(yaw_interpolation) {}

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

            target_pose.translation() = target;
            target_pose.setQuaternion(q);
        }

        void execute() override {

            if (move_time == 0.0)
            {
                p = 1.0;
            } else {
                auto t = (ros::Time::now() - start_time).toSec();
                p = t / move_time;
                if (p >= 1.0) {p = 1.0;}
            }
            
            current_target_local = Sophus::interpolate(start_pose, target_pose, p);
            target_position = current_target_local.translation();
            Eigen::Matrix3d R = current_target_local.rotationMatrix();
            double yaw_target_t = yaw_target;
            if (p >= 1.0) {
              target_velocity = Eigen::Vector3d(0.0, 0.0, 0.0);
            }
            else {
              target_velocity = (target_pose.translation() - start_pose.translation())/move_time;
            }

            if (interpolate_yaw)
            {
              if (p >= 1.0) {
                yaw_target_t = yaw_target;
                yaw_target_dot = 0.;
              }
              else {
                yaw_target_t = yaw_target*p;
                yaw_target_dot = yaw_target/move_time;
              }
            }

            target_acceleration = Eigen::Vector3d(0.0, 0.0, 0.0);
            target_jerk = Eigen::Vector3d(0.0, 0.0, 0.0);
            Sophus::SE3d currentTargetPose;
            currentTargetPose.translation() = target_position;
            currentTargetPose.setQuaternion(target_pose.unit_quaternion()); // why unit quarternion if we have a yaw target here?

            quad->setTargetState(target_position, target_velocity, target_acceleration, target_jerk, yaw_target_t, yaw_target_dot); // The target position is in local frame, not world frame??
        }

        bool isFinished() override {
            double delta2 = 0;
            auto currentPose = quad->getLocalPose();

            Sophus::Vector3d diff = (currentPose.translation() - target_pose.translation());

            return (diff.norm() < tolerance) || (ros::Time::now() - start_time) > ros::Duration(timeout);
        };

        std::string getName() override {
        return "move";
        }



        private:
        Sophus::Vector3d        target;
        double                  yaw_target = 0.;
        double                  yaw_target_dot = 0.;
        double                  tolerance;
        double                  move_time;
        double                  timeout;
        double p;
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
