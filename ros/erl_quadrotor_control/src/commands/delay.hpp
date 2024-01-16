#ifndef ERL_CONTROL_DELAY_HPP
#define ERL_CONTROL_DELAY_HPP

#include <ros/ros.h>
#include "command.hpp"

namespace ERLControl{

    class DelayCommand : public Command {
        public:
        DelayCommand(const double& duration = 5.0) : delay_duration(duration) {}

        void initialize() override {
            delay_start_time = ros::Time::now();
        }

        bool isFinished() override {
            return (ros::Time::now() - delay_start_time) > delay_duration;
        };

        std::string getName() override {
            return "wait_height_stable";
        }

        private:
        ros::Duration   delay_duration;
        ros::Time       delay_start_time;
    };

}

#endif