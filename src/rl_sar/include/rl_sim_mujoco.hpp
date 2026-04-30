/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RL_SIM_HPP
#define RL_SIM_HPP

// #define PLOT
#define CSV_LOGGER

#include "rl_sdk.hpp"
#include "observation_buffer.hpp"
#include "inference_runtime.hpp"
#include "loop.hpp"
#include "fsm_all.hpp"

#include <csignal>
#include <vector>
#include <string>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <memory>
#include <deque>
#include <mutex>
#include <random>
#include <chrono>

#include <mujoco/mujoco.h>
#include "joystick.hh"
#include "mujoco_utils.hpp"

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

class Button
{
public:
    Button() {}

    void update(bool state)
    {
        on_press = state ? state != pressed : false;
        on_release = state ? false : state != pressed;
        pressed = state;
    }

    bool pressed = false;
    bool on_press = false;
    bool on_release = false;
};

class RL_Sim : public RL
{
public:
    RL_Sim(int argc, char **argv);
    ~RL_Sim();

    std::unique_ptr<mj::Simulate> sim;
    static RL_Sim* instance;

private:
    // rl functions
    std::vector<float> Forward() override;
    void GetState(RobotState<float> *state) override;
    void SetCommand(const RobotCommand<float> *command) override;
    void RunModel();
    void RobotControl();

    // loop
    std::shared_ptr<LoopFunc> loop_keyboard;
    std::shared_ptr<LoopFunc> loop_joystick;
    std::shared_ptr<LoopFunc> loop_control;
    std::shared_ptr<LoopFunc> loop_rl;
    std::shared_ptr<LoopFunc> loop_plot;

    // plot
    const int plot_size = 100;
    std::vector<int> plot_t;
    std::vector<std::vector<float>> plot_real_joint_pos, plot_target_joint_pos;
    void Plot();

    // mujoco
    mjData *mj_data;
    mjModel *mj_model;
    std::string scene_name;

    // joystick
    std::unique_ptr<Joystick> sys_js;
    JoystickEvent sys_js_event;

    Button sys_js_button[20];
    int sys_js_axis[10] = {0};
    bool sys_js_active = false;
    float axis_deadzone = 0.05f;
    int sys_js_max_value = (1 << (16 - 1));
    void SetupSysJoystick(const std::string& device, int bits);
    void GetSysJoystick();

    // others
    std::string gazebo_model_name;
    std::map<std::string, float> joint_positions;
    std::map<std::string, float> joint_velocities;
    std::map<std::string, float> joint_efforts;
    void StartJointController(const std::string& ros_namespace, const std::vector<std::string>& names);

    // heading hold outer-loop states
    bool yaw_hold_target_initialized = false;
    float yaw_hold_target = 0.0f;

    // ============ Sim-to-Real: Actuator Action Delay ============
    // 设计：policy 在 50Hz 计算出的动作先进入时间戳 FIFO (action_delay_buffer_)，
    //       RobotControl (200Hz) 每轮检查一次，若 steady_clock::now() ≥ release_time
    //       则把该动作出队并推进原有的 output_dof_*_queue。这保证 10~20ms 的延时
    //       以 5ms (control loop dt) 的粒度离散化 → K ∈ {2,3,4}，per-episode 采样
    //       后在该 episode 内固定，完全复现 IsaacLab training 时的延时分布。
    struct DelayedAction {
        std::chrono::steady_clock::time_point release_time;
        std::vector<float> dof_pos, dof_vel, dof_tau;
    };
    std::deque<DelayedAction> action_delay_buffer_;
    std::mutex action_delay_mutex_;
    std::mt19937 delay_rng_{std::random_device{}()};
    float current_delay_ms_ = 0.0f;  // 本 episode 固定的延时，0 表示禁用
    bool was_rl_init_done_ = false;  // 检测 RL 状态首次进入，用于首次延时采样

    // 从 params 读 [min,max] 范围重新采样本 episode 的延时（R 键复位时调用）
    void ResampleActionDelay();
    // 清空滞留动作（R 键复位、FSM 重入 RL 状态时调用，防止上一 episode 脏数据）
    void ClearActionDelayBuffer();
    // 由 RobotControl (200Hz) 调用：把已到期的动作 flush 到 output_dof_*_queue
    void DrainActionDelayBuffer();
    #ifdef CSV_LOGGER
    std::shared_ptr<LoopFunc> loop_log;
    std::chrono::steady_clock::time_point log_t0;
    void LogTick();
#endif
#ifdef RL_MUJOCO_TEST_CSV
    // Test-only CSV: ang_vel(3), base quat wxyz(4), command(3).
    bool csv_initialized = false;
    std::ofstream test_csv_file;
#endif
};

#endif // RL_SIM_HPP
