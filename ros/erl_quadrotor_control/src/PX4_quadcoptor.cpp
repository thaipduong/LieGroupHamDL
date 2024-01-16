#include "PX4_quadcoptor.hpp"

#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/BatteryState.h>
#include <std_msgs/Bool.h>
#include <sstream>
#include <chrono>


#include "utils/SophusConversion.hpp"
#include <tf/tf.h>

namespace ERLControl{

    const double MINIMUM_THROTTLE = 1500; // ms
    const double CRITICAL_THROTTLE = 1000; // ms
    const int    THROTTLE_CHANNEL = 2;

    void PX4Quadcoptor::execute() {
        state_machine_update();
        publishTargetPose();
        ros::spinOnce();
    }


    void PX4Quadcoptor::setTargetState(const Eigen::Vector3d& target_position, const Eigen::Vector3d& target_velocity, const Eigen::Vector3d& target_acceleration, const Eigen::Vector3d& target_jerk, const double target_yaw,  const double target_yaw_dot) {
        target_position_world = target_position;
        target_velocity_world = target_velocity;
        target_acceleration_world = target_acceleration;
        target_jerk_world = target_jerk;
        target_yaw_world = target_yaw; 
        target_yaw_dot_world = target_yaw_dot;
    }

    void PX4Quadcoptor::setControlMethod(const ControlMethod& ctrlMethod){
        ctrl = ctrlMethod;
    }

    void PX4Quadcoptor::setStartDataCommandIndex(){
        idx_command_start_data = incomming_commands.size();
    }

    void PX4Quadcoptor::setEndDataCommandIndex(){
        idx_command_end_data = incomming_commands.size();
    }

    Sophus::SE3d PX4Quadcoptor::getLocalPose(){
      return convert(mavros_odometry.pose.pose);
    }

    bool PX4Quadcoptor::offboard_main() {
        is_offboard = true;
        if (exed_command_count == idx_command_start_data)
          is_collecting_data = true;
        if (exed_command_count == idx_command_end_data)
          is_collecting_data = false;
        //std::cout << "####" << is_collecting_data << "$$$$" << idx_command_start_data << "####" << idx_command_end_data << "####" << exed_command_count << std::endl;
        if (incomming_commands.size() >= 1){
            incomming_commands.front()->step();
            if (incomming_commands.front()->isProcessEnded()){
                ROS_WARN("[PX4 Quadcoptor] Command [%s] finished", incomming_commands.front()->getName().c_str());
                incomming_commands.pop();

                if (incomming_commands.size() >= 1){
                    ROS_WARN("[PX4 Quadcoptor] Running command [%s]", incomming_commands.front()->getName().c_str());
                }else{
                    ROS_WARN("[PX4 Quadcoptor] No more commands");
                }
                exed_command_count++;
            }
            return false;
        }else{
            return true; // signal to exit offboard state and land
        }
    }

    void PX4Quadcoptor::addCommand(std::unique_ptr<Command>&& command){
        command->setPX4(this);
        incomming_commands.push(std::move(command));

    }

    void PX4Quadcoptor::initNode(int argc, char** argv, std::string aName){
        
        // initialize node handle
        ros::init(argc, argv, aName);
        nh = new ros::NodeHandle();
        nh->getParam("/offb/controlMethod", ctrlString);

        // Control Method 
        if (ctrlString == "GEOMETRIC") {
            setControlMethod(ERLControl::ControlMethod::GEOMETRIC);
            ROS_WARN("[Control Method]: GEOMETRIC");
        }
        else if (ctrlString == "LEARNED_ENERGY_BASED") {
            setControlMethod(ERLControl::ControlMethod::LEARNED_ENERGY_BASED);
            ROS_WARN("[Control Method]: LEARNED_ENERGY_BASED");
        }
        else {
            setControlMethod(ERLControl::ControlMethod::PX4);
            ROS_WARN("[Control Method]: PX4");
        }

        /////////////////////////////////////////////// OBTAIN CONTROL GAINS //////////////////////////
        Eigen::Vector3d Kp, Kv, KR, Kw;
        std::string Gnet_path, Dvnet_path, Dwnet_path;
        nh->getParam("/offb/Kpx", Kp(0));
        nh->getParam("/offb/Kpy", Kp(1));
        nh->getParam("/offb/Kpz", Kp(2));
        nh->getParam("/offb/Kvx", Kv(0));
        nh->getParam("/offb/Kvy", Kv(1));
        nh->getParam("/offb/Kvz", Kv(2));
        nh->getParam("/offb/KRx", KR(0));
        nh->getParam("/offb/KRy", KR(1));
        nh->getParam("/offb/KRz", KR(2));
        nh->getParam("/offb/Kwx", Kw(0));
        nh->getParam("/offb/Kwy", Kw(1));
        nh->getParam("/offb/Kwz", Kw(2));
        nh->getParam("/offb/Gnet", Gnet_path);
        nh->getParam("/offb/Dvnet", Dvnet_path);
        nh->getParam("/offb/Dwnet", Dwnet_path);

        // Init EnergyBased Control
        energyBasedCtrl.init(Kp, Kv, KR, Kw, Gnet_path, Dvnet_path, Dwnet_path);

        // initialize subscribers
        mavros_state_sub = nh->subscribe<mavros_msgs::State>(
            "mavros/state", 
            10, 
            &PX4Quadcoptor::mavros_state_cb, 
            this
        );

        mavros_extended_state_sub = nh->subscribe<mavros_msgs::ExtendedState>(
            "mavros/extended_state", 
            10, 
            &PX4Quadcoptor::mavros_extended_state_cb, 
            this
        );

        mavros_battery_sub = nh->subscribe<sensor_msgs::BatteryState>(
            "mavros/battery",
            1,
            &PX4Quadcoptor::mavros_battery_state_cb,
            this
        );

        mavros_odometry_sub = nh->subscribe<nav_msgs::Odometry>(
            "mavros/odometry/in", 
            10, 
            &PX4Quadcoptor::mavros_odometry_cb, 
            this
        );

        mavros_rc_sub = nh->subscribe<mavros_msgs::RCIn>(
            "mavros/rc/in", 
            10, 
            &PX4Quadcoptor::mavros_rc_cb, 
            this
        );

        
        // initialize publishers
        local_pos_target_pub = nh->advertise<geometry_msgs::PoseStamped>("mavros/setpoint_position/local",10);
        kr_pos_cmd_pub = nh->advertise<kr_mav_msgs::PositionCommand>("position_cmd", 10, true);
        kr_motors_pub = nh->advertise<std_msgs::Bool>("/motors", 10, true);
        geometric_control_pub = nh->advertise<mavros_msgs::AttitudeTarget>("mavros/setpoint_raw/attitude",10);
        target_trajectory_pub = nh->advertise<nav_msgs::Odometry>("target_trajectory",10);
        data_collection_flag_pub = nh->advertise<std_msgs::Bool>("collect_data",1);

        // initialize service clients
        request_arming_cl = nh->serviceClient<mavros_msgs::CommandBool>("mavros/cmd/arming");
        switch_mode_cl = nh->serviceClient<mavros_msgs::SetMode>("mavros/set_mode");
        

        node_start_time = ros::Time::now();


        // Initial Targets 
        target_position_world = Eigen::Vector3d(0.0, 0.0, 1.0);
        target_velocity_world = Eigen::Vector3d(0.0, 0.0, 0.0);
        target_acceleration_world = Eigen::Vector3d(0.0, 0.0, 0.0);
        target_jerk_world = Eigen::Vector3d(0.0, 0.0, 0.0);
        target_yaw_world = 0.0; 
    }

    // callback functions for subscribers
    void PX4Quadcoptor::mavros_state_cb(const StateConstPtr& msg){
        mavros_state = *msg;
        mavros_state_count++;
    }

    void PX4Quadcoptor::mavros_battery_state_cb(const BatteryStateConstPtr& msg){
        if (msg->voltage > 0)
          voltage = msg->voltage;
        if (msg->percentage > 0)
          if (msg->percentage <= 1)
            percentage = msg->percentage;
    }

    void PX4Quadcoptor::mavros_extended_state_cb(const ExStateConstPtr& msg){
        mavros_extended_state = *msg;
        mavros_extended_state_count++;
    }
    
    void PX4Quadcoptor::mavros_odometry_cb(const OdometryConstPtr& msg){
        mavros_odometry = *msg;
        mavros_odometry_count++;
    }

    void PX4Quadcoptor::mavros_rc_cb(const RCConstPtr& msg){
        mavros_rc = *msg;
        mavros_rc_count++;
    }


    // state machine functions
    std::map<PX4States, StateTransitionFunctions> PX4Quadcoptor::state_machine_map = {
        {PX4States::DISCONNECTED,           &PX4Quadcoptor::state_machine_disconnected},
        {PX4States::CHECK_THROTTLE,         &PX4Quadcoptor::state_machine_check_throttle},
        {PX4States::RECEIVING_ORIENTATION,  &PX4Quadcoptor::state_machine_receiving_odometry},
        {PX4States::SWITCHING_MODE,         &PX4Quadcoptor::state_machine_switching_mode},
        {PX4States::ARMING,                 &PX4Quadcoptor::state_machine_arming},
        {PX4States::OFFBOARD_CONTROLLED,    &PX4Quadcoptor::state_machine_offboard_controlled},
        {PX4States::AUTO_LAND,              &PX4Quadcoptor::state_machine_auto_land},
        {PX4States::REMOTE_CONTROLLED,      &PX4Quadcoptor::state_machine_remote_controlled},
    };

    void PX4Quadcoptor::state_machine_disconnected(){
        if (mavros_state.connected == true){
            state = PX4States::RECEIVING_ORIENTATION;
            ROS_WARN("[PX4Quadcoptor] FCU connection detected, switching to [RECEIVING_ORIENTATION] state");
            return;
        }
    }

    void PX4Quadcoptor::state_machine_check_throttle(){
        
        // Check throttle 
        check_throttle_iter++;

        if (mavros_rc_count < 100){
            if (check_throttle_iter % 100 == 0){
                ROS_WARN("Remote signal pending");
            }
            return;
        }
        
        if (check_throttle_iter % 250 == 0){
            if (check_throttle_error() == false){ // no error
                state = PX4States::SWITCHING_MODE;
            }
        }
    }

    void PX4Quadcoptor::state_machine_receiving_odometry(){
        if (mavros_odometry_count >= 100 && initial_pose_world_valid == false){
            initial_pose_world_valid = true;

            Sophus::SE3d unProcessedInitialPoseWorld = convert(mavros_odometry.pose.pose);

            // ignore rotation on roll and pitch axis
            initial_pose_world.translation() = unProcessedInitialPoseWorld.translation();
            initial_pose_world.so3() = Sophus::SO3d().rotZ(unProcessedInitialPoseWorld.angleZ());


            state = PX4States::CHECK_THROTTLE;
            ROS_WARN("[PX4Quadcoptor] initial pose received, switching to [CHECK THROTTLE] state");
        }
    }

    void PX4Quadcoptor::state_machine_switching_mode(){
        
        // wait for 5 seconds since start, send initial takeoff pose to FCU
        if ((ros::Time::now() - node_start_time) < ros::Duration(5.0)){
            return;
        }

        // switch to offboard mode and try to arm
        if (mavros_state.mode == "OFFBOARD"){
            state = PX4States::ARMING; 
            ROS_WARN("[PX4Quadcoptor] quadcoptor in [OFFBOARD] mode! switching to [ARMING] state");
        }
        else{
            if (last_mode_retry_time == nullptr || (ros::Time::now() - *last_mode_retry_time) > ros::Duration(5.0)){
            ROS_WARN("[PX4Quadcoptor] time passed, trying to switch mode");
            
            mavros_msgs::SetMode changeModeRequest;
            changeModeRequest.request.custom_mode = "OFFBOARD";

            switch_mode_cl.call(changeModeRequest);

            last_mode_retry_time = std::make_unique<ros::Time>(ros::Time::now());
        }
        }
    }

    void PX4Quadcoptor::state_machine_arming(){
        if (!mavros_state.armed && mavros_state.mode == "OFFBOARD") {
            if (last_arming_retry_time == nullptr || (ros::Time::now() - *last_arming_retry_time) > ros::Duration(5.0)){
                ROS_WARN("[PX4Quadcoptor] time passed, trying to arm");
                
                mavros_msgs::CommandBool arming_request;
                arming_request.request.value = true;
                request_arming_cl.call(arming_request);
                last_arming_retry_time = std::make_unique<ros::Time>(ros::Time::now());

                if (arming_request.response.success == true){
                    state = PX4States::OFFBOARD_CONTROLLED;
                    kr_motor_arming.data = true; 
                    kr_motors_pub.publish(kr_motor_arming);
                    ROS_WARN("[PX4Quadcoptor] quadcoptor armed! switching to [OFFBOARD_CONTROLLED] state");
                } else {
                    kr_motor_arming.data = false; 
                    kr_motors_pub.publish(kr_motor_arming);
                    ROS_WARN("[PX4Quadcoptor] arming failed, retrying in 5.0 seconds");
                }
            }
        }
        else if (mavros_state.armed && mavros_state.mode == "OFFBOARD") {
            state = PX4States::OFFBOARD_CONTROLLED;
            ROS_WARN("[PX4Quadcoptor] quadcoptor armed! switching to [OFFBOARD_CONTROLLED] state");
        }
    }

    void PX4Quadcoptor::state_machine_offboard_controlled(){
        bool isOffboardFinished = offboard_main();
        if (isOffboardFinished){
            state = PX4States::AUTO_LAND;
            ROS_WARN("[PX4Quadcoptor] commands finished! switching to [AUTO_LAND] state");
        }

        check_remote_take_over();
    }

    void PX4Quadcoptor::state_machine_auto_land(){
        mavros_msgs::SetMode autolandSwitchCommand;
        autolandSwitchCommand.request.custom_mode = "AUTO.LAND";
        
        if (mavros_state.mode != "AUTO.LAND"){
            if (switch_mode_cl.call(autolandSwitchCommand) && autolandSwitchCommand.response.mode_sent){
                ROS_WARN("[PX4Quadcoptor] auto landing!");
                // FIXME: test only
                
            }
        }
        
        if(mavros_state.mode == "AUTO.LAND"){
            is_shut_down = true;
        }
    }
    
    void PX4Quadcoptor::state_machine_remote_controlled(){

    }

    void PX4Quadcoptor::publishTargetPose(){
      std_msgs::Bool data_flag_msg;
      data_flag_msg.data = is_collecting_data;
      data_collection_flag_pub.publish(data_flag_msg);
      kr_pos_cmd.position.x = target_position_world.x();
      kr_pos_cmd.position.y = target_position_world.y();
      kr_pos_cmd.position.z = target_position_world.z();
      kr_pos_cmd.velocity.x = target_velocity_world.x();
      kr_pos_cmd.velocity.y = target_velocity_world.y();
      kr_pos_cmd.velocity.z = target_velocity_world.z();
      kr_pos_cmd.acceleration.x = target_acceleration_world.x();
      kr_pos_cmd.acceleration.y = target_acceleration_world.y();
      kr_pos_cmd.acceleration.z = target_acceleration_world.z();
      kr_pos_cmd.jerk.x = 0.0;
      kr_pos_cmd.jerk.y = 0.0;
      kr_pos_cmd.jerk.z = 0.0;
      kr_pos_cmd.yaw = target_yaw_world;
      kr_pos_cmd.yaw_dot = target_yaw_dot_world;

      std::string frame_id = "map";
      ros::Time stamp = ros::Time::now();

      nav_msgs::Odometry target_odom;
      target_odom.header.stamp = stamp;
      target_odom.header.frame_id = frame_id;

      geometry_msgs::Pose pose_vis;
      pose_vis.position = kr_pos_cmd.position;
      auto q_vis = tf::createQuaternionMsgFromYaw(target_yaw_world);
      pose_vis.orientation = q_vis;

      geometry_msgs::Twist twist_vis;
      twist_vis.linear.x = target_velocity_world.x(); //world frame
      twist_vis.linear.y = target_velocity_world.y(); //world frame
      twist_vis.linear.z = target_velocity_world.z(); //world frame

      target_odom.pose.pose = pose_vis;
      target_odom.twist.twist = twist_vis;
      target_trajectory_pub.publish(target_odom);

      switch (ctrl){
          case ControlMethod::GEOMETRIC:
          {
            kr_pos_cmd_pub.publish(kr_pos_cmd);
            break;
          }
          case ControlMethod::LEARNED_ENERGY_BASED:
          {
            energyBasedCtrl.setTargetState(kr_pos_cmd);
            energyBasedCtrl.setCurrentState(mavros_odometry);
            mavros_msgs::AttitudeTarget attitudeMsg;
            if (is_offboard == true) {
                energyBasedCtrl.compute(is_offboard);
                attitudeMsg = energyBasedCtrl.getAttitudeTargetMsg(frame_id, stamp);
            }

            else {
                attitudeMsg.header.stamp = stamp;
                attitudeMsg.header.frame_id = frame_id;
                attitudeMsg.body_rate.x = 0.0;
                attitudeMsg.body_rate.y = 0.0;
                attitudeMsg.body_rate.z = 0.0;
                attitudeMsg.thrust = 0.0;
                attitudeMsg.type_mask = 128;  // Ignore orientation messages
                attitudeMsg.orientation.w = 1.0;
                attitudeMsg.orientation.x = 0.0;
                attitudeMsg.orientation.y = 0.0;
                attitudeMsg.orientation.z = 0.0;
            }

            bool scaled_by_battery = true;
            if (scaled_by_battery){
              //std::cout << "Battery voltage = " << voltage << ", percentage = " << percentage << std::endl;
            }

            geometric_control_pub.publish(attitudeMsg);
            auto send_time = std::chrono::high_resolution_clock::now();
            break;
          }
          case ControlMethod::PX4:
          {
            geometry_msgs::PoseStamped poseStamped;
            poseStamped.header.frame_id = frame_id;
            poseStamped.header.seq = pose_pub_seq++;
            poseStamped.header.stamp = stamp;
            poseStamped.pose.position = kr_pos_cmd.position;
            auto q = tf::createQuaternionMsgFromYaw(target_yaw_world);
            poseStamped.pose.orientation = q;
            local_pos_target_pub.publish(poseStamped);
            break;
          }
          default:
            break;
      }

   }



    void PX4Quadcoptor::check_remote_take_over(){
        check_remote_takeover_display_interval++;

        if (mavros_state.mode == "MANUAL" || mavros_state.mode == "STABLIZED"){
            remote_take_over = true;
            ROS_ERROR("remote take over detected, mode is %s", mavros_state.mode.c_str());
        }

        if (check_remote_takeover_display_interval % 25 == 0){
            check_throttle_error(); // spam the console for immediate notice 
        }
    }

    bool PX4Quadcoptor::check_throttle_error(){
        double throttleMs = mavros_rc.channels[THROTTLE_CHANNEL];

        if (throttleMs >= MINIMUM_THROTTLE){
            // throttle safe for takeover
            return false;
        }else if (throttleMs >= CRITICAL_THROTTLE){
            // throttle stick working, but is too low for a safe takeover
            ROS_WARN("[PX4Quadcoptor] Throttle is unsafe for manual take over.");
            ROS_WARN("[PX4Quadcoptor] Please increase throttle to abound 60 percent.");
        }else{
            // ARM control stick on the remote control is in wrong position
            // in this case, throttle RC output is locked at minimum.
            // If take over in this setting, quadcoptor will fall to ground!!
            ROS_ERROR("[PX4Quadcoptor] Throttle is very unsafe for manual take over.");
            ROS_ERROR("[PX4Quadcoptor] The armnig stick on the remote control is in wrong position");

        }

        return true;
    }

}
