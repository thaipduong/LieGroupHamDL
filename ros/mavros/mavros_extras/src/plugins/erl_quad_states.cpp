#include <mavros/mavros_plugin.h>
#include <mavros_msgs/ERLQuadStates.h>
#include <tf2_eigen/tf2_eigen.h>

#include <boost/algorithm/string.hpp>

#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TransformStamped.h>

namespace mavros {
namespace extra_plugins {
using Matrix6d = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>;




class ERLQuadStatesPlugin : public plugin::PluginBase {
public:
	ERLQuadStatesPlugin() : PluginBase(),
		erl_quad_states_nh("~erl"),
    fcu_odom_parent_id_des("map"),
    fcu_odom_child_id_des("base_link")
	{ }

	void initialize(UAS &uas_) override
	{
		PluginBase::initialize(uas_);

		erl_quad_states_pub = erl_quad_states_nh.advertise<mavros_msgs::ERLQuadStates>("erl_quad_states", 10);
		// publishers
		odom_pub = erl_quad_states_nh.advertise<nav_msgs::Odometry>("erl_quad_odom", 10);
	}

	Subscriptions get_subscriptions() override
	{
		return {
			make_handler(&ERLQuadStatesPlugin::handle_erl_quad_states)
		};
	}

private:
	ros::NodeHandle erl_quad_states_nh;

	ros::Publisher erl_quad_states_pub;
	ros::Publisher odom_pub;			//!< nav_msgs/Odometry publisher


  //!< desired orientation of the fcu odometry message's parent frame
  std::string fcu_odom_parent_id_des;
  //!< desired orientation of the fcu odometry message's child frame
  std::string fcu_odom_child_id_des;

  /**
	 * @brief Lookup static transform with error handling
	 * @param[in] &target The parent frame of the transformation you want to get
	 * @param[in] &source The child frame of the transformation you want to get
	 * @param[in,out] &tf_source2target The affine transform from the source to target
	 */
	void lookup_static_transform(const std::string &target, const std::string &source,
		Eigen::Affine3d &tf_source2target)
	{
		try {
			// transform lookup at current time.
			tf_source2target = tf2::transformToEigen(m_uas->tf2_buffer.lookupTransform(
				target, source, ros::Time(0)));
		} catch (tf2::TransformException &ex) {
			ROS_ERROR_THROTTLE_NAMED(1, "odom", "ODOM: Ex: %s", ex.what());
			return;
		}
	}



	void handle_erl_quad_states(const mavlink::mavlink_message_t *msg, mavlink::common::msg::ERL_QUAD_STATES &erl_quad_states_msg)
	{


    /**
     * Required rotations to transform the FCU's odometry msg tto desired parent and child frame
     */
    Eigen::Affine3d tf_parent2parent_des;
    Eigen::Affine3d tf_child2child_des;

    lookup_static_transform(fcu_odom_parent_id_des, "map_ned", tf_parent2parent_des);
    lookup_static_transform(fcu_odom_child_id_des, "base_link_frd", tf_child2child_des);



		auto odom = boost::make_shared<nav_msgs::Odometry>();

		odom->header = m_uas->synchronized_header(fcu_odom_parent_id_des, erl_quad_states_msg.timestamp);
		//odom->pose.covariance[2] = erl_quad_states_msg.timestamp % 1000;
		//odom->pose.covariance[1] = ((erl_quad_states_msg.timestamp - (erl_quad_states_msg.timestamp % 1000))/1000) % 1000;
		//odom->pose.covariance[0] = ((erl_quad_states_msg.timestamp - (erl_quad_states_msg.timestamp % 1000000))/1000000) % 1000;
		odom->child_frame_id = fcu_odom_child_id_des;

Eigen::Vector3d position {};         //!< Position vector. WRT frame_id
    Eigen::Quaterniond orientation {};   //!< Attitude quaternion. WRT frame_id
    Eigen::Vector3d lin_vel {};          //!< Linear velocity vector. WRT child_frame_id
    Eigen::Vector3d ang_vel {};          //!< Angular velocity vector. WRT child_frame_id
    Eigen::Vector3d torque {};          //!< Control input vector. WRT child_frame_id
    Eigen::Vector3d scaled_torque {};          //!< Scaled control input vector. WRT child_frame_id
	/**
     * Position parsing to desired parent
     */
    position =
      Eigen::Vector3d(
      tf_parent2parent_des.linear() *
      Eigen::Vector3d(erl_quad_states_msg.position[0], erl_quad_states_msg.position[1], erl_quad_states_msg.position[2]));
tf::pointEigenToMsg(position, odom->pose.pose.position);
    /**
     * Orientation parsing. Quaternion has to be the rotation from desired child frame to desired parent frame
     */
    Eigen::Quaterniond q_child2parent(ftf::mavlink_to_quaternion(erl_quad_states_msg.orientation));
    Eigen::Affine3d tf_childDes2parentDes = tf_parent2parent_des * q_child2parent *
      tf_child2child_des.inverse();
    orientation = Eigen::Quaterniond(tf_childDes2parentDes.linear());

		tf::quaternionEigenToMsg(orientation, odom->pose.pose.orientation);
    /**
     * Velocities parsing
     * Linear and angular velocities are transforned to the desired child_frame.
     */
    lin_vel =
      Eigen::Vector3d(
      tf_child2child_des.linear() *
      Eigen::Vector3d(erl_quad_states_msg.velocity[0], erl_quad_states_msg.velocity[1], erl_quad_states_msg.velocity[2]));
    ang_vel =
      Eigen::Vector3d(
      tf_child2child_des.linear() *
      Eigen::Vector3d(erl_quad_states_msg.angular_velocity[0], erl_quad_states_msg.angular_velocity[1], erl_quad_states_msg.angular_velocity[2]));
        tf::vectorEigenToMsg(lin_vel, odom->twist.twist.linear);
		tf::vectorEigenToMsg(ang_vel, odom->twist.twist.angular);
    /**
     * Control parsing
     * Force and torque are transforned to the desired child_frame.
     */
   torque =
     Eigen::Vector3d(
      tf_child2child_des.linear() *
      Eigen::Vector3d(erl_quad_states_msg.controls[1], erl_quad_states_msg.controls[2], erl_quad_states_msg.controls[3]));
    scaled_torque =
      Eigen::Vector3d(
      tf_child2child_des.linear() *
      Eigen::Vector3d(erl_quad_states_msg.controls_scaled[1], erl_quad_states_msg.controls_scaled[2], erl_quad_states_msg.controls_scaled[3]));



        auto erl_quad_states = boost::make_shared<mavros_msgs::ERLQuadStates>();
        erl_quad_states->timestamp = erl_quad_states_msg.timestamp; 
		erl_quad_states->position[0] = position(0);
		erl_quad_states->position[1] = position(1);
		erl_quad_states->position[2] = position(2);
		erl_quad_states->orientation[0] = orientation.w();
		erl_quad_states->orientation[1] = orientation.x();
		erl_quad_states->orientation[2] = orientation.y();
		erl_quad_states->orientation[3] = orientation.z();
		erl_quad_states->velocity[0] = lin_vel(0);
		erl_quad_states->velocity[1] = lin_vel(1);
		erl_quad_states->velocity[2] = lin_vel(2);
		erl_quad_states->angular_velocity[0] = ang_vel(0);
		erl_quad_states->angular_velocity[1] = ang_vel(1);
		erl_quad_states->angular_velocity[2] = ang_vel(2);
        erl_quad_states->controls[0] = erl_quad_states_msg.controls_scaled[0]; // scaled torque in frd frame
        erl_quad_states->controls[1] = erl_quad_states_msg.controls_scaled[1];
        erl_quad_states->controls[2] = erl_quad_states_msg.controls_scaled[2];
        erl_quad_states->controls[3] = erl_quad_states_msg.controls_scaled[3];
		erl_quad_states->controls_scaled[0] = erl_quad_states_msg.controls_scaled[0];
        erl_quad_states->controls_scaled[1] = scaled_torque[0];  // scaled torque in enu frame
        erl_quad_states->controls_scaled[2] = scaled_torque[1];
        erl_quad_states->controls_scaled[3] = scaled_torque[2];

        //! Publish the data
        erl_quad_states_pub.publish(erl_quad_states);
		odom_pub.publish(odom);
	}
};
}	// namespace extra_plugins
}	// namespace mavros

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(mavros::extra_plugins::ERLQuadStatesPlugin, mavros::plugin::PluginBase)
