/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RL_REAL_MYROBOT_HPP
#define RL_REAL_MYROBOT_HPP

// #define PLOT
#define CSV_LOGGER
// #define USE_ROS

#include "rl_sdk.hpp"
#include "observation_buffer.hpp"
#include "inference_runtime.hpp"
#include "loop.hpp"
#include "fsm_myrobot.hpp"

#include <zmq.h>
#include <csignal>
#include <cmath>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <cstring>

#if defined(USE_ROS1) && defined(USE_ROS)
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#elif defined(USE_ROS2) && defined(USE_ROS)
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#endif

#ifdef PLOT
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

// ZMQ communication protocol constants (must match laptop_zmqtest.py)
static constexpr int ZMQ_STATE_DIM = 49;       // q(21) + qd(21) + ang_vel(3) + quat(4)
static constexpr int ZMQ_CMD_DIM = 21;         // q_cmd(21)
static constexpr int ZMQ_TS_BYTES = 8;         // timestamp: float64
static constexpr int ZMQ_STATE_PAYLOAD_BYTES = ZMQ_TS_BYTES + ZMQ_STATE_DIM * 8;
static constexpr int ZMQ_CMD_PAYLOAD_BYTES = ZMQ_TS_BYTES + ZMQ_CMD_DIM * 8;

class RL_Real : public RL
{
public:
    RL_Real(int argc, char **argv);
    ~RL_Real();

#if defined(USE_ROS2) && defined(USE_ROS)
    std::shared_ptr<rclcpp::Node> ros2_node;
#endif

private:
    // rl functions
    std::vector<float> Forward() override;
    void GetState(RobotState<float> *state) override;
    void SetCommand(const RobotCommand<float> *command) override;
    void RunModel();
    void RobotControl();

    // loop
    std::shared_ptr<LoopFunc> loop_keyboard;
    std::shared_ptr<LoopFunc> loop_control;
    std::shared_ptr<LoopFunc> loop_rl;

#ifdef CSV_LOGGER
    // Dedicated full-trajectory logger: runs from Passive to program kill,
    // independent of whether the RL policy is active.
    std::shared_ptr<LoopFunc> loop_log;
    std::chrono::steady_clock::time_point log_t0;
    void LogTick();
#endif

#ifdef PLOT
    std::shared_ptr<LoopFunc> loop_plot;
    const int plot_size = 100;
    std::vector<int> plot_t;
    std::vector<std::vector<float>> plot_real_joint_pos, plot_target_joint_pos;
    void Plot();
#endif

    // ZMQ communication
    void *zmq_context = nullptr;
    void *zmq_sub = nullptr;      // SUB socket: receive low_state from Pi
    void *zmq_pub = nullptr;      // PUB socket: send low_cmd to Pi
    std::string zmq_pi_ip = "192.168.1.133";
    int zmq_state_port = 5555;
    int zmq_cmd_port = 5556;
    std::string zmq_state_topic = "low_state";
    std::string zmq_cmd_topic = "low_cmd";

    bool InitZMQ();
    void CloseZMQ();
    bool ReceiveState();          // Non-blocking receive from Pi
    void SendCommand();           // Send q_cmd to Pi

    // Latest received state (protected by mutex for thread safety)
    std::mutex state_mutex;
    double last_state_ts = 0.0;
    float state_q[21] = {0};
    float state_qd[21] = {0};
    float state_ang_vel[3] = {0};
    float state_quat[4] = {1, 0, 0, 0};  // w, x, y, z
    std::atomic<bool> state_received{false};

    // Command to send
    float cmd_q[21] = {0};

    // Yaw hold PD controller (from rl_sim_mujoco.cpp)
    bool yaw_hold_target_initialized = false;
    float yaw_hold_target = 0.0f;

#if defined(USE_ROS1) && defined(USE_ROS)
    geometry_msgs::Twist cmd_vel;
    ros::Subscriber cmd_vel_subscriber;
    void CmdvelCallback(const geometry_msgs::Twist::ConstPtr &msg);
#elif defined(USE_ROS2) && defined(USE_ROS)
    geometry_msgs::msg::Twist cmd_vel;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_subscriber;
    void CmdvelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
#endif
};

#endif // RL_REAL_MYROBOT_HPP
