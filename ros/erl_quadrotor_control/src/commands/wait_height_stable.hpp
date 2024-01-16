#ifndef ERL_WAIT_HEIGHT_STABLE_HPP
#define ERL_WAIT_HEIGHT_STABLE_HPP

#include <ros/ros.h>
#include <string>
#include <memory.h>
#include "command.hpp"
#include "../PX4_quadcoptor.hpp"
#include <math.h>

namespace ERLControl{

    class WaitHeightStableCommand : public Command {
        public:
        WaitHeightStableCommand(const double& zVelTolerance, double stableDuration = 3.0) : 
            tolerance(zVelTolerance), stable_duration(stableDuration) {}

        void execute() override {
            if (abs(quad->getMavrosOdometry().twist.twist.linear.z) < tolerance){
                if (!stable_since_valid){
                    stable_since = ros::Time::now();
                    stable_since_valid = true;
                }
            }else{
                stable_since_valid = false;
            }
            
        }

        bool isFinished() override {
            return (stable_since_valid && ros::Time::now() - stable_since > stable_duration); 
        };

        std::string getName() override {
            return "wait_height_stable";
        }

        private:
        double          tolerance;
        ros::Duration   stable_duration;
        ros::Time       stable_since;
        bool            stable_since_valid = false;
    };

}

#endif