#include <ros/ros.h>

#include "PX4_quadcoptor.hpp"
#include "commands/delay.hpp"
#include "commands/wait_height_stable.hpp"
#include "commands/circle.hpp"
#include "commands/vertical_circle.hpp"
#include "commands/lemniscate.hpp"
#include "commands/vertical_lemniscate.hpp"
#include "commands/waypoint.hpp"
#include "commands/piecewise_linear.hpp"
#include <vector>
#include <math.h>


int main(int argc, char **argv) {
    ERLControl::PX4Quadcoptor quad;

    quad.addCommand(std::make_unique<ERLControl::WayPoint>(Eigen::Vector3d(0.0,0.0,0.4), 0.0, 2, 0.05));

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::vector<double> x_des = {0, -2, 2};
    std::vector<double> y_des = {0, -2, 2};
    std::vector<double> z_des = {3};
    std::vector<double> yaw_des = {-2*M_PI/3, -M_PI/3, 0, M_PI/3, 2*M_PI/3};
    quad.setStartDataCommandIndex();
    for (double xd : x_des){
        for (double zd : z_des){
            quad.addCommand(std::make_unique<ERLControl::WayPoint>(Eigen::Vector3d(xd, 0, zd) /*waypoint*/, 0 /*des yaw*/, 0 /*total time*/, 0.05 /*threshold*/));
            quad.addCommand(std::make_unique<ERLControl::DelayCommand>(0.1));
            quad.addCommand(std::make_unique<ERLControl::WayPoint>(Eigen::Vector3d(0.0,0.0,1.0) /*waypoint*/, 0 /*des yaw*/, 0 /*total time*/, 0.05 /*threshold*/));
            quad.addCommand(std::make_unique<ERLControl::DelayCommand>(0.1));
        }
    }

      for (double yd : y_des){
        for (double zd : z_des){
            quad.addCommand(std::make_unique<ERLControl::WayPoint>(Eigen::Vector3d(0, yd, zd) /*waypoint*/, 0 /*des yaw*/, 0 /*total time*/, 0.05 /*threshold*/));
            quad.addCommand(std::make_unique<ERLControl::DelayCommand>(0.1));
            quad.addCommand(std::make_unique<ERLControl::WayPoint>(Eigen::Vector3d(0.0,0.0,1.0) /*waypoint*/, 0 /*des yaw*/, 0 /*total time*/, 0.05 /*threshold*/));
            quad.addCommand(std::make_unique<ERLControl::DelayCommand>(0.1));
        }
      }

    for (double zd : z_des){
      for (double yawd : yaw_des){
        quad.addCommand(std::make_unique<ERLControl::WayPoint>(Eigen::Vector3d(0, 0, zd) /*waypoint*/, yawd /*des yaw*/, 0 /*total time*/, 0.05 /*threshold*/));
        quad.addCommand(std::make_unique<ERLControl::DelayCommand>(0.1));
        quad.addCommand(std::make_unique<ERLControl::WayPoint>(Eigen::Vector3d(0.0,0.0,1.0) /*waypoint*/, 0 /*des yaw*/, 0 /*total time*/, 0.05 /*threshold*/));
        quad.addCommand(std::make_unique<ERLControl::DelayCommand>(0.1));
      }
    }
    quad.setEndDataCommandIndex();

    quad.addCommand(std::make_unique<ERLControl::WayPoint>(Eigen::Vector3d(0.0,0.0,0.4), 0.0, 2, 0.05));



    quad.initNode(argc, argv, "px4_quad_data_collection");

    ROS_WARN("======Node initialized======\n");

    ros::Rate loopRate(250);

    while (ros::ok() && !quad.isRemoteTakeover() && !quad.isShutdown()){
        quad.execute();
        loopRate.sleep();
    }
    
    return 0;
}
