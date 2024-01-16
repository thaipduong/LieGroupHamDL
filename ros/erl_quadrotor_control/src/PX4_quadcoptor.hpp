#ifndef PX4_QUADCOPTOR_HPP
#define PX4_QUADCOPTOR_HPP


#include <ros/ros.h>
#include <sophus/se3.hpp>

#include <string>
#include <memory>
#include <map>
#include <queue>

// messages
#include <std_msgs/Bool.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/ExtendedState.h>
#include <mavros_msgs/RCIn.h>
#include <sensor_msgs/BatteryState.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <kr_mav_msgs/PositionCommand.h>
#include <mavros_msgs/AttitudeTarget.h>

#include "command.hpp"
#include "energy_based_control.hpp"

namespace ERLControl{

    // definition of state machines
    enum class PX4States {
        DISCONNECTED = 0,
        CHECK_THROTTLE,
        RECEIVING_ORIENTATION,
        SWITCHING_MODE,
        ARMING,
        OFFBOARD_CONTROLLED,
        AUTO_LAND,
        REMOTE_CONTROLLED
    };

    // definition of control method
    enum class ControlMethod {
        PX4 = 0,
        GEOMETRIC = 1,
        LEARNED_ENERGY_BASED = 2
    };

    class PX4Quadcoptor;
    using StateConstPtr = mavros_msgs::State::ConstPtr;
    using BatteryStateConstPtr = sensor_msgs::BatteryState::ConstPtr;
    using ExStateConstPtr = mavros_msgs::ExtendedState::ConstPtr;
    using OdometryConstPtr = nav_msgs::Odometry::ConstPtr;
    using RCConstPtr = mavros_msgs::RCIn::ConstPtr;
    using PoseConstPtr = geometry_msgs::PoseStamped::ConstPtr;
    using TwistConstPtr = geometry_msgs::TwistStamped::ConstPtr;

    using StateTransitionFunctions = void (PX4Quadcoptor::*)(void);

    class PX4Quadcoptor{
        public:
        // initialize ros node, should be called at beginning
        void        initNode(int argc, char** argv, std::string aName);

        // main function of PX4Quadcoptor class. Should be called periodically
        // in the node.
        void        execute();

        // get current PX4 state
        PX4States   getState(){ return state; }

        // getters for remote takeover and shutdown
        bool        isRemoteTakeover() {return remote_take_over; }
        bool        isShutdown() {return is_shut_down; }

        // getters to received messages
        mavros_msgs::State          getMavrosState() { return mavros_state; }
        mavros_msgs::ExtendedState  getMavrosExtendedState() { return mavros_extended_state; }
        nav_msgs::Odometry          getMavrosOdometry() { return mavros_odometry; }
        mavros_msgs::RCIn           getMavrosRCIn() { return mavros_rc; }

        // return the local pose. Local pose is defined in 
        // the frame indicated by variable [initial_pose_world]
        
        // should be only used after receiving pose and valid
        // [initial_pose_world]. If not, the missing coordinate will be defaulted
        // to identity.
        Sophus::SE3d getLocalPose();
        
        // set the position target for PX4 in LOCAL POSE. Local pose is defined in 
        // the frame indicated by variable [initial_pose_world]
//        void setTargetPose(const geometry_msgs::Pose& targetPoseLocal);
//        void setTargetPose(const Sophus::SE3d& targetPoseLocal);
        void setTargetState(const Eigen::Vector3d& target_position,
                            const Eigen::Vector3d& target_velocity,
                            const Eigen::Vector3d& target_acceleration,
                            const Eigen::Vector3d& target_jerk,
                            const double target_yaw,
                            const double target_yaw_dot = 0.);


        // Energy Based Control
        EnergyBasedControl energyBasedCtrl;

        // set control method 
        void setControlMethod(const ControlMethod& ctrlMethod);
        
        // add a new command to the command queue for the quad to execute
        // when entered [OFFBOARD_CONTROLLED] mode.
        void addCommand(std::unique_ptr<Command>&& command);
        
        void setStartDataCommandIndex();
        void setEndDataCommandIndex();


        protected:
        // ===== Quadcoptor states =====
        
        // stores ROS time when [initNode()] is called 
        ros::Time                   node_start_time;

        // position command variables
        size_t                      pose_pub_seq = 0;
        //Sophus::SE3d                target_pose_world; // target pose in world frame
        Eigen::Vector3d             target_position_world;
        Eigen::Vector3d             target_velocity_world;
        Eigen::Vector3d             target_acceleration_world;
        Eigen::Vector3d             target_jerk_world;
        double                      target_yaw_world = 0.;
        double                      target_yaw_dot_world = 0.;

        Sophus::SE3d                initial_pose_world; // initial pose in world frame
        bool                        initial_pose_world_valid = false;

        // kr position command variable 
        kr_mav_msgs::PositionCommand kr_pos_cmd; // target position + yaw angle 

        // offboard mode variable
        bool                        is_offboard = false; 
        bool                        is_offboard_finished = false; // is the command queue empty
        bool                        is_shut_down = false; // is the node supposed to be shutdown
        bool                        is_collecting_data = false; // is the state and control data being collected?

        // remote takeover functionality
        bool                        remote_take_over = false; // is a remote takeover detected
        size_t                      check_remote_takeover_display_interval = 0; // interval to display throttle warning

        // state machine variables
        
        // [CHECK_THROTTLE]
        size_t                      check_throttle_iter = 0;
        
        // [ARMING]
        std::unique_ptr<ros::Time>  last_arming_retry_time = nullptr;
        
        // [SWITCHING_MODE]
        std::unique_ptr<ros::Time>  last_mode_retry_time = nullptr;

        // Battery
        double voltage = -1;
        double percentage = -1;


        // command pattern for offboard mode. This queue will store all the 
        // commands of the quad, and execute them sequentially.
        std::queue<std::unique_ptr<Command>> incomming_commands;

        // ------ ROS INTERFACES ------
        ros::NodeHandle*            nh = nullptr;

        // publishers
        ros::Publisher              local_pos_target_pub;
        ros::Publisher              kr_pos_cmd_pub;
        ros::Publisher              kr_motors_pub;
        ros::Publisher              geometric_control_pub;
        ros::Publisher              target_trajectory_pub;
        ros::Publisher              data_collection_flag_pub;

        // service clients
        ros::ServiceClient          request_arming_cl;
        ros::ServiceClient          switch_mode_cl;

        // subscribers
        ros::Subscriber             mavros_state_sub;
        ros::Subscriber             mavros_battery_sub;
        ros::Subscriber             mavros_extended_state_sub;
        ros::Subscriber             mavros_odometry_sub;
        ros::Subscriber             mavros_rc_sub;
//        ros::Subscriber             mavros_pose_sub;
//        ros::Subscriber             mavros_vel_sub;

        // received messages
        mavros_msgs::State          mavros_state;
        mavros_msgs::ExtendedState  mavros_extended_state;
        nav_msgs::Odometry          mavros_odometry;
        mavros_msgs::RCIn           mavros_rc;
//        geometry_msgs::PoseStamped  mavros_pose;
//        geometry_msgs::TwistStamped mavros_twist;
        std_msgs::Bool              kr_motor_arming;
        mavros_msgs::AttitudeTarget mavros_attitude; 

        // received message count
        size_t                      mavros_state_count = 0;
        size_t                      mavros_extended_state_count = 0;
        size_t                      mavros_odometry_count = 0; 
        size_t                      mavros_rc_count = 0;
        size_t                      exed_command_count = 0;
        int                         idx_command_start_data = -1;
        int                         idx_command_end_data = -1;

        // callback functions for subscribers
        void mavros_state_cb(const StateConstPtr& msg);
        void mavros_battery_state_cb(const BatteryStateConstPtr& msg);
        void mavros_extended_state_cb(const ExStateConstPtr& msg);
        void mavros_odometry_cb(const OdometryConstPtr& msg);
        void mavros_rc_cb(const RCConstPtr& msg);
        void mavros_pose_cb(const PoseConstPtr& msg);
        void mavros_vel_cb(const TwistConstPtr& msg);

        // called automatically at each loop to feed watchdog
        void publishTargetPose();

        private:
        // main function to be called in offboard mode.
        bool offboard_main();

        // check if the current state is switched to MANUAL or STABLIZED
        void check_remote_take_over();

        // check if the current throttle is too low for remote takeover
        bool check_throttle_error();

        // ------ STATE MACHINE ------
        PX4States                   state = PX4States::DISCONNECTED;

        // ------ CONTROL METHOD ------
        ControlMethod               ctrl = ControlMethod::PX4;
        std::string                 ctrlString; 
        
        void state_machine_update(){
            // call the corresponding function based on current state
            (this->*state_machine_map[state])();
        }

        // state machine will call each of these functions at corresponding states
        static std::map<PX4States, StateTransitionFunctions> state_machine_map;

        // state machine main functions. Will be called at the corresponding states
        void state_machine_disconnected();
        void state_machine_check_throttle();
        void state_machine_receiving_odometry();
        void state_machine_switching_mode();
        void state_machine_arming();
        void state_machine_offboard_controlled();
        void state_machine_auto_land();
        void state_machine_remote_controlled();        
    };

}

#endif
