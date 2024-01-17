
#include "energy_based_control.hpp"
#include <chrono>


namespace ERLControl{

    void EnergyBasedControl::init(Eigen::Vector3d Kp, Eigen::Vector3d Kv, Eigen::Vector3d KR, Eigen::Vector3d Kw,
                                  std::string Gnet_path, std::string Dvnet_path, std::string Dwnet_path) {
        Kp_ = Kp;
        Kv_ = Kv;
        KR_ = KR;
        Kw_ = Kw;

        if (!Gnet_path.empty()){
          std::cout << "Loading G(q) from " << Gnet_path << std::endl;
          module_gnet = torch::jit::load(Gnet_path);
        }
        if (!Dvnet_path.empty()){
          std::cout << "Loading Dv(q) from " << Dvnet_path<< std::endl;
          module_Dvnet = torch::jit::load(Dvnet_path);
        }
        if (!Dwnet_path.empty()){
          std::cout << "Loading Dw(q) from " << Dwnet_path << std::endl;
          module_Dwnet = torch::jit::load(Dwnet_path);
        }


    }



    void EnergyBasedControl::setCurrentState(nav_msgs::Odometry &odom) {
        position_ << odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z;
        Eigen::Vector4d quat;
        quat << odom.pose.pose.orientation.w, odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z;
        velocity_ << odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z; // body frame
        angular_velocity_ << odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z; // body frame
        rotmat_ = EnergyBasedControl::quat2RotMatrix(quat);

        auto start = std::chrono::high_resolution_clock::now();

        torch::Tensor pose = torch::tensor({0., 0., 0.,
          1., 0., 0.,
          0., 1., 0.,
          0., 0., 1.}, {torch::kFloat32});
        torch::Tensor x = torch::tensor({{0., 0., 0.}}, {torch::kFloat32});
        torch::Tensor R = torch::tensor({{rotmat_(0,0), rotmat_(0,1), rotmat_(0,2),
                                          rotmat_(1,0), rotmat_(1,1), rotmat_(1,2),
                                          rotmat_(2,0), rotmat_(2,1), rotmat_(2,2)}}, {torch::kFloat32});
        std::vector<torch::jit::IValue> poses;
        poses.push_back(pose);
        atOutputg = module_gnet.forward(poses).toTensor();

        auto atOutputg_float = atOutputg.accessor<float,3>();
        for (int i = 0; i < 6; i++)
          for (int j = 0; j < 4; j++)
              outputg(i,j) = atOutputg_float[0][i][j];

        // Since the dynamics don't change if we scale the mass, potential energy, and input matrix, we fixed the mass
        // and learn the scaled version of the input matrix G and the dissipation, without loss of generality.
        double m = 1.3;
        for (int i = 0; i < 3; i++)
        {
          ouputdVdp(i,0) = 0.0;
          ouputdVdR1(i,0) = 0.0;
          ouputdVdR2(i,0) = 0.0;
          ouputdVdR3(i,0) = 0.0;
        }
        ouputdVdp(2,0) = m*9.81;


        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
              outputM1(i,j) = 0.0;
        outputM1(0,0) = 1/m;
        outputM1(1,1) = 1/m;
        outputM1(2,2) = 1/m;
        outputM1 = outputM1.inverse().eval();

        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
              outputM2(i,j) = 0.0;
        outputM2(0,0) = 1/0.012;
        outputM2(1,1) = 1/0.012;
        outputM2(2,2) = 1/0.02;
        outputM2 = outputM2.inverse().eval();


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
        std::vector<torch::jit::IValue> xs;
        xs.push_back(x);
        atoutputDv = module_Dvnet.forward(xs).toTensor();
        auto atoutputDv_float = atoutputDv.accessor<float, 3>();
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
              outputDv(i,j) = atoutputDv_float[0][i][j];

        std::vector<torch::jit::IValue> Rs;
        Rs.push_back(R);
        atoutputDw = module_Dwnet.forward(Rs).toTensor();

        auto atoutputDw_float = atoutputDw.accessor<float, 3>();
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
              outputDw(i,j) = atoutputDw_float[0][i][j];

        auto finish = std::chrono::high_resolution_clock::now();
//        std::cout << "Neural net queries took "
//                   << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
//                   << " microseconds\n";
    }


    void EnergyBasedControl::setTargetState(const kr_mav_msgs::PositionCommand kr_pos_cmd) {
        target_position_ << kr_pos_cmd.position.x, kr_pos_cmd.position.y, kr_pos_cmd.position.z;
        target_velocity_ << kr_pos_cmd.velocity.x, kr_pos_cmd.velocity.y, kr_pos_cmd.velocity.z; // velocity in the world frame
        target_acceleration_ << kr_pos_cmd.acceleration.x, kr_pos_cmd.acceleration.y, kr_pos_cmd.acceleration.z;
        target_yaw_ = kr_pos_cmd.yaw;
        target_yaw_dot = kr_pos_cmd.yaw_dot;
    }


    void EnergyBasedControl::compute(const bool &start) {
      auto start_time = std::chrono::high_resolution_clock::now();
      double yaw = atan2(rotmat_(1,0), rotmat_(0,0));
      auto RT = rotmat_.transpose();
      auto w_hat = EnergyBasedControl::hat_map(angular_velocity_);
      Eigen::Vector3d pv = outputM1*velocity_;
      Eigen::Vector3d pw = outputM2*angular_velocity_;

      // Calculate thrust
      auto RTdV = RT*ouputdVdp;
      auto pvxw = pv.cross(angular_velocity_);
      Eigen::Vector3d Dvv = outputDv*velocity_;
      auto Kpep = Kp_.cwiseProduct(position_ - target_position_);
      auto RTKp = outputM1*RT*Kpep;
      auto Kdvv_ref = outputM1*(Kv_.cwiseProduct(velocity_ - RT*target_velocity_));
      auto pvdot_ref = outputM1*(RT*target_acceleration_ - w_hat*(RT*target_velocity_));
      Eigen::Vector3d b_p_B = RTdV  - RTKp - Kdvv_ref + pvdot_ref - pvxw + Dvv;
      Eigen::Vector3d b_p = rotmat_*b_p_B;

      // Limite max tilt angle
      auto tiltangle = acos(b_p(2)/b_p.norm());
      double scale_acc = 1;
      if (tiltangle > max_tilt_angle){
        double xy_mag = sqrt(b_p(0)*b_p(0) + b_p(1)*b_p(1));
        double xy_mag_max = b_p(2) * tan(max_tilt_angle);
        scale_acc = xy_mag_max/xy_mag;
        b_p(0) = b_p(0) * scale_acc;
        b_p(1) = b_p(1) * scale_acc;
      }
      b_p_B = RT*b_p;

      //////////////////////////////////////////////////////////////////////////////
      Eigen::Vector3d b1c, b2c, b3c;
      const Eigen::Vector3d b2d(-std::sin(target_yaw_), std::cos(target_yaw_), 0);

      if(b_p.norm() > 1e-6f)
        b3c.noalias() = b_p.normalized();
      else
        b3c.noalias() = Eigen::Vector3d::UnitZ();

      b1c.noalias() = b2d.cross(b3c).normalized();
      b2c.noalias() = b3c.cross(b1c).normalized();

      const Eigen::Vector3d b_p_Bdot = scale_acc*outputM1*(-Kp_.cwiseProduct(velocity_ - RT*target_velocity_));
      const Eigen::Vector3d b_p_dot = rotmat_*b_p_Bdot;
      const Eigen::Vector3d b3c_dot = b3c.cross(b_p_dot / b_p.norm()).cross(b3c);
      const Eigen::Vector3d b2d_dot(-std::cos(target_yaw_) * target_yaw_dot, -std::sin(target_yaw_) * target_yaw_dot, 0);
      const Eigen::Vector3d b1c_dot =
          b1c.cross(((b2d_dot.cross(b3c) + b2d.cross(b3c_dot)) / (b2d.cross(b3c)).norm()).cross(b1c));
      const Eigen::Vector3d b2c_dot = b3c_dot.cross(b1c) + b3c.cross(b1c_dot);

      Eigen::Matrix3d Rc;
      Rc << b1c, b2c, b3c;
      q_des_ = Eigen::Quaterniond(Rc);

      Eigen::Matrix3d Rc_dot;
      Rc_dot << b1c_dot, b2c_dot, b3c_dot;
      //////////////////////////////////////////////////////////////////////////////
      Eigen::Matrix3d wc_hat = Rc.transpose()*Rc_dot;
      Eigen::Vector3d wc = EnergyBasedControl::vee_map(wc_hat);
      // Calculate b_R
      Eigen::Vector3d rotmat_r1, rotmat_r2, rotmat_r3;
      rotmat_r1 << rotmat_(0,0), rotmat_(0,1), rotmat_(0,2);
      rotmat_r2 << rotmat_(1,0), rotmat_(1,1), rotmat_(1,2);
      rotmat_r3 << rotmat_(2,0), rotmat_(2,1), rotmat_(2,2);
      Eigen::Vector3d rxdV = rotmat_r1.cross(ouputdVdR1) + rotmat_r2.cross(ouputdVdR2) + rotmat_r3.cross(ouputdVdR3);
      Eigen::Vector3d pwxw = pw.cross(angular_velocity_);
      Eigen::Vector3d pvxv = pv.cross(velocity_);
      Eigen::Vector3d Dww = outputDw*angular_velocity_;
      Eigen::Vector3d e_euler = 0.5*KR_.cwiseProduct(EnergyBasedControl::vee_map(Rc.transpose()*rotmat_ - RT*Rc));
      Eigen::Vector3d kdwwc = Kw_.cwiseProduct(angular_velocity_ - RT*(Rc*wc));
      Eigen::Vector3d bR = outputM2*(-e_euler - kdwwc) - pwxw - pvxv + Dww + rxdV;

      // Calculate the control
      auto outputgT = outputg.transpose();
      auto ggT_inv = (outputgT*outputg).inverse().eval();
      auto g_dagger = ggT_inv*outputgT;
      Eigen::Matrix<double, 6, 1> wrench;

      Eigen::Vector3d max_threshold;
      max_threshold << 1., 1., 1.; // for safety of the real drone, we limit the raw control input here just in case it goes too high.

      if (b_p_B(0) < -1){
        b_p_B(0) = -1;
      }
      if (b_p_B(0) > 1){
        b_p_B(0) = 1;
      }
      if (b_p_B(1) < -1){
        b_p_B(1) = -1;
      }
      if (b_p_B(1) > 1){
        b_p_B(1) = 1;
      }

      bR = bR.cwiseMin(max_threshold).cwiseMax(-max_threshold);

      wrench << b_p_B(0), b_p_B(1), b_p_B(2),bR(0), bR(1), bR(2);
      control = g_dagger*wrench;
      if (control(0) < 0){
          control(0) = 0;
      }
      if (control(0) > 1){
          control(0) = 1;
      }

      control(1) = wc(0);
      control(2) = wc(1);
      control(3) = wc(2);

      auto finish_time = std::chrono::high_resolution_clock::now();
//              std::cout << "##################### computing control took "
//                         << std::chrono::duration_cast<std::chrono::microseconds>(finish_time - start_time).count()
//                         << " microseconds\n";
    }


    mavros_msgs::AttitudeTarget EnergyBasedControl::getAttitudeTargetMsg(std::string frame_id, ros::Time stamp) {
        mavros_msgs::AttitudeTarget msg;

        msg.header.stamp = stamp;
        msg.header.frame_id = frame_id;
        msg.body_rate.x = control(1);
        msg.body_rate.y = control(2);
        msg.body_rate.z = control(3);
        msg.thrust = control(0);
        msg.type_mask = 0;//128;  // Ignore orientation messages
        msg.orientation.w = q_des_.w();
        msg.orientation.x = q_des_.x();
        msg.orientation.y = q_des_.y();
        msg.orientation.z = q_des_.z();

        return msg; 
    }

    double EnergyBasedControl::getYawAngleFromQuat(Eigen::Vector4d &quat) {
        const Eigen::Matrix3d R = EnergyBasedControl::quat2RotMatrix(quat);
        double yaw =  atan2(R(1,0), R(0,0));
        return yaw;
    }

    Eigen::Vector4d EnergyBasedControl::rot2Quaternion(const Eigen::Matrix3d &R) {
        Eigen::Vector4d quat;
        double tr = R.trace();
        if (tr > 0.0) {
            double S = sqrt(tr + 1.0) * 2.0;  // S=4*qw
            quat(0) = 0.25 * S;
            quat(1) = (R(2, 1) - R(1, 2)) / S;
            quat(2) = (R(0, 2) - R(2, 0)) / S;
            quat(3) = (R(1, 0) - R(0, 1)) / S;
        } else if ((R(0, 0) > R(1, 1)) & (R(0, 0) > R(2, 2))) {
            double S = sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2)) * 2.0;  // S=4*qx
            quat(0) = (R(2, 1) - R(1, 2)) / S;
            quat(1) = 0.25 * S;
            quat(2) = (R(0, 1) + R(1, 0)) / S;
            quat(3) = (R(0, 2) + R(2, 0)) / S;
        } else if (R(1, 1) > R(2, 2)) {
            double S = sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2)) * 2.0;  // S=4*qy
            quat(0) = (R(0, 2) - R(2, 0)) / S;
            quat(1) = (R(0, 1) + R(1, 0)) / S;
            quat(2) = 0.25 * S;
            quat(3) = (R(1, 2) + R(2, 1)) / S;
        } else {
            double S = sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1)) * 2.0;  // S=4*qz
            quat(0) = (R(1, 0) - R(0, 1)) / S;
            quat(1) = (R(0, 2) + R(2, 0)) / S;
            quat(2) = (R(1, 2) + R(2, 1)) / S;
            quat(3) = 0.25 * S;
        }
        return quat;
    }

    Eigen::Matrix3d EnergyBasedControl::quat2RotMatrix(const Eigen::Vector4d &q) {
        Eigen::Matrix3d rotmat;
        rotmat << q(0) * q(0) + q(1) * q(1) - q(2) * q(2) - q(3) * q(3), 2 * q(1) * q(2) - 2 * q(0) * q(3),
            2 * q(0) * q(2) + 2 * q(1) * q(3),

            2 * q(0) * q(3) + 2 * q(1) * q(2), q(0) * q(0) - q(1) * q(1) + q(2) * q(2) - q(3) * q(3),
            2 * q(2) * q(3) - 2 * q(0) * q(1),

            2 * q(1) * q(3) - 2 * q(0) * q(2), 2 * q(0) * q(1) + 2 * q(2) * q(3),
            q(0) * q(0) - q(1) * q(1) - q(2) * q(2) + q(3) * q(3);
        return rotmat;
    }

    Eigen::Vector3d EnergyBasedControl::vee_map(const Eigen::Matrix3d &m) {
        Eigen::Vector3d v;
        v << m(2,1), m(0,2), m(1,0);
        return v;
    }

    Eigen::Matrix3d EnergyBasedControl::hat_map(const Eigen::Vector3d &m) {
        Eigen::Matrix3d m_hat;
        m_hat << 0., -m(2), m(1), m(2), 0., -m(0), -m(1), m(0), 0.;
        return m_hat;
    }


}
