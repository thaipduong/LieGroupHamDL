#include <ros/ros.h>
#include <mavros_msgs/RCIn.h>


ros::Publisher rc_signal_pub;

int main(int argc, char** argv){
    
    ros::init(argc, argv, "dummy_RC_node");
    ros::NodeHandle nh;

    ros::Rate rate(50);
    rc_signal_pub = nh.advertise<mavros_msgs::RCIn>("mavros/rc/in", 10);

    mavros_msgs::RCIn rcInput;
    rcInput.channels.push_back(1600);
    rcInput.channels.push_back(1600);
    rcInput.channels.push_back(1600);
    rcInput.channels.push_back(1600);
    
    while (ros::ok()){
        rc_signal_pub.publish(rcInput);
        rate.sleep();
    }

}