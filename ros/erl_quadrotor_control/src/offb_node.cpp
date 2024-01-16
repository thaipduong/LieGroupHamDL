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

////    ////////////////////////////////////////////////////////////TRAJECTORY TRACKING///////////////////////////////////////////////////////////////
    /* Circular Trajectory */
    quad.addCommand(std::make_unique<ERLControl::WayPoint>(Eigen::Vector3d(0.0,0.0,1.0), 0.0, 5, 0.05, 10, true));
    quad.addCommand(std::make_unique<ERLControl::DelayCommand>(10.0));
    quad.addCommand(std::make_unique<ERLControl::CircleCommand>(Eigen::Vector3d(-2.0,0.0,1.0) /*center*/, 2.0 /*radius*/, 0.5 /*omega*/, 0.0 /*yaw*/, 180.0 /*total_time*/));
    quad.addCommand(std::make_unique<ERLControl::WayPoint>(Eigen::Vector3d(0.0,0.0,1.0), 0.0, 2, 0.05));
    quad.addCommand(std::make_unique<ERLControl::DelayCommand>(1.0));
    quad.addCommand(std::make_unique<ERLControl::WayPoint>(Eigen::Vector3d(0.0,0.0,0.4), 0.0, 2, 0.05));

    // /* Piecewise Linear Trajectory */
    // quad.addCommand(std::make_unique<ERLControl::WayPoint>(Eigen::Vector3d(0.0, 0.0, 2.0), 0.0 /*yaw*/, 10 /*duration*/, 0.05 /*tolerance*/));
    // quad.addCommand(std::make_unique<ERLControl::DelayCommand>(3.0));
    // quad.addCommand(std::make_unique<ERLControl::WayPoint>(Eigen::Vector3d(-4.0, 0.0, 2.0), 0.0 /*yaw*/, 10 /*duration*/, 0.05 /*tolerance*/));
    // quad.addCommand(std::make_unique<ERLControl::DelayCommand>(3.0));
    // quad.addCommand(std::make_unique<ERLControl::PiecewiseLinear>(Eigen::Vector3d(-2.0, 1.0, 2.0) /*Waypoint_1*/, 
    //                                                               Eigen::Vector3d(0.0, -1.0, 2.0) /*Waypoint_2*/,
    //                                                               Eigen::Vector3d(2.0, 1.0, 2.0) /*Waypoint_3*/,
    //                                                               Eigen::Vector3d(4.0, 0.0, 2.0) /*Waypoint_4*/,
    //                                                               0.0 /*yaw*/, 20 /*duration*/, 0.05 /*tolerance*/));
    // quad.addCommand(std::make_unique<ERLControl::WayPoint>(Eigen::Vector3d(0.0,0.0,1.0), 0.0 /*yaw*/, 10 /*duration*/, 0.05 /*tolerance*/));


//    // Lemniscate Trajectory
//     quad.addCommand(std::make_unique<ERLControl::WayPoint>(Eigen::Vector3d(0.0,2.0,1.0) /*target position*/, 0.0 /*target yaw*/, 2 /*time*/, 0.05 /*goal threshold*/));
//     quad.addCommand(std::make_unique<ERLControl::DelayCommand>(10.0));
//     quad.addCommand(std::make_unique<ERLControl::LemniscateCommand>(Eigen::Vector3d(0.0,0.0,1.0) /*center*/, 2.0 /*half width*/, 0.5 /*omega*/, 0.5*1.57079632679 /*yaw*/, 30.0 /*time*/));
//     quad.addCommand(std::make_unique<ERLControl::WayPoint>(Eigen::Vector3d(0.0,0.0,1.0), 0.0, 2, 0.05));
//     quad.addCommand(std::make_unique<ERLControl::DelayCommand>(1.0));
//     quad.addCommand(std::make_unique<ERLControl::WayPoint>(Eigen::Vector3d(0.0,0.0,0.4), 0.0, 2, 0.05));




    quad.initNode(argc, argv, "px4_quad");

    ROS_WARN("======Node initialized======\n");

    ros::Rate loopRate(250);

    while (ros::ok() && !quad.isRemoteTakeover() && !quad.isShutdown()){
        quad.execute();
        loopRate.sleep();
    }
    
    return 0;
}
