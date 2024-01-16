#ifndef ERL_CONTROL_VERTICAL_LEMINSCATE_HPP
#define ERL_CONTROL_VERTICAL_LEMINSCATE_HPP

#include <ros/ros.h>
#include "command.hpp"
#include "math.h"

#include "../PX4_quadcoptor.hpp"

#include <geometry_msgs/Pose.h>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

namespace ERLControl{

    class VerticalLemniscateCommand : public Command {
        public:
        VerticalLemniscateCommand(Eigen::Vector3d aCenter, double aHalfWidth, double aAngularVelocity, double ayaw, double aTime) :
         center(aCenter), a(aHalfWidth), angularVelocity(aAngularVelocity), yaw(ayaw), time(aTime) {}

        void initialize() override {
            start_time = ros::Time::now();
            omega = angularVelocity;

            start_pose = quad->getLocalPose(); // used to get target orientation
        }

        void execute() override {
            auto t = (ros::Time::now() - start_time).toSec();
            double th = t * omega;

            Eigen::Vector3d target_position, target_velocity, target_acceleration, target_jerk;
            target_position.x() = center.x() + 0;
            target_position.y() = center.y() + (a * cos(th)) / (1 + sin(th)*sin(th));
            target_position.z() = center.z() + (a * sin(th) * cos(th)) / (1 + sin(th)*sin(th));
            target_velocity.x() = 0.0;
            target_velocity.y() = -a * omega * sin(th) * (pow(sin(th), 2) + 2*pow(cos(th),2) + 1) / (pow(sin(th)*sin(th) + 1, 2));
            target_velocity.z() = -a * omega * (pow(sin(th), 4) + pow(sin(th), 2) + (pow(sin(th), 2) - 1)*pow(cos(th), 2)) / pow(pow(sin(th), 2) + 1, 2); 
            target_acceleration.x() = 0.0;
            target_acceleration.y() = a * pow(omega, 2) * cos(th) * (44 * cos(2*th) + cos(4*th) - 21) / pow(cos(2*th) - 3, 3);
            target_acceleration.z() = 4 * a * pow(omega, 2) * sin(2*th) * (3 * cos(2*th) + 7) / pow(cos(2*th) - 3, 3);
            target_jerk.x() = 0.0; // Neglect
            target_jerk.y() = 0.0; // Neglect
            target_jerk.z() = 0.0; // Neglect

            
            Sophus::SE3d currentTargetPose;
            currentTargetPose.translation() = target_position;
            currentTargetPose.setQuaternion(start_pose.unit_quaternion());  
            double yaw_target = yaw;

            //quad->setTargetPose(currentTargetPose);
            quad->setTargetState(target_position, target_velocity, target_acceleration, target_jerk, yaw_target);
        }

        bool isFinished() override {
            return (ros::Time::now() - start_time) > ros::Duration(time);
        };

        std::string getName() override {
            return "vertical_lemniscate";
        }

        private:
        Eigen::Vector3d         center;
        double                  a;
        double                  angularVelocity;
        double                  yaw;
        double                  time;

        double                  omega = 0;

        ros::Time               start_time;
        Sophus::SE3d            start_pose;
    };

}

#endif
