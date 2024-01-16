#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <gazebo_msgs/ModelStates.h>
#include "utils/SophusConversion.hpp"
#include "utils/EigenConversion.hpp"

ros::Publisher mocap_odometry_pub;
ros::Subscriber gazebo_odometry_sub;
ros::Subscriber mavros_odometry_sub;

int getIndex(std::vector<std::string> v, std::string value)
{
    for(int i = 0; i < v.size(); i++)
    {
        if(v[i].compare(value) == 0)
            return i;
    }
    return -1;
}

void model_states_callback(const gazebo_msgs::ModelStates::ConstPtr& model_states)
{
    int quad_index = getIndex(model_states->name, "iris");
    nav_msgs::Odometry quad_odom;

    ros::Time current_time = ros::Time::now();
    quad_odom.header.stamp = current_time;
    quad_odom.header.frame_id = "odom";
    quad_odom.pose.pose = model_states->pose[quad_index];
    Eigen::Quaterniond quat = ERLControl::convert(quad_odom.pose.pose.orientation);
    if (quat.norm() < 0.1)
    {
      ROS_WARN("Invalid quaternion!!! Waiting for valid odometry...");
      return;
    }
    Sophus::SE3d quad_pose_sophus = ERLControl::convert(quad_odom.pose.pose);
    Sophus::SO3d R_T(quad_pose_sophus.rotationMatrix().transpose());
    auto lin_vel_world = ERLControl::convert(model_states->twist[quad_index].linear);
    auto ang_vel_world = ERLControl::convert(model_states->twist[quad_index].angular);
    auto lin_vel_body = R_T * lin_vel_world;
    auto ang_vel_body = R_T * ang_vel_world;

    quad_odom.child_frame_id = "base_link";
    quad_odom.twist.twist.linear.x = lin_vel_body.x();
    quad_odom.twist.twist.linear.y = lin_vel_body.y();
    quad_odom.twist.twist.linear.z = lin_vel_body.z();
    quad_odom.twist.twist.angular.x = ang_vel_body.x();
    quad_odom.twist.twist.angular.y = ang_vel_body.y();
    quad_odom.twist.twist.angular.z = ang_vel_body.z();
    mocap_odometry_pub.publish(quad_odom);
}


int main(int argc, char** argv){
    
    ros::init(argc, argv, "mocap_sim_node");
    ros::NodeHandle nh;

    ros::Rate rate(250);
    mocap_odometry_pub = nh.advertise<nav_msgs::Odometry>("/mavros/odometry/out", 1);
    gazebo_odometry_sub = nh.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states",
                                                                 1,
                                                                 model_states_callback);

    while (ros::ok()){
      //ROS_WARN("mocap_sim_node running");
      ros::spinOnce();
        rate.sleep();
    }

}


