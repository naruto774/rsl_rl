/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MYROBOT_FSM_HPP
#define MYROBOT_FSM_HPP

#include "fsm.hpp"
#include "rl_sdk.hpp"
#include <cmath>

namespace myrobot_fsm
{

class RLFSMStatePassive : public RLFSMState
{
public:
    RLFSMStatePassive(RL *rl) : RLFSMState(*rl, "RLFSMStatePassive") {}

    void Enter() override
    {
        std::cout << LOGGER::NOTE << "Entered passive mode. Press '0' (Keyboard) or 'A' (Gamepad) to switch to RLFSMStateGetUp." << std::endl;
    }

    void Run() override
        {
            // 读取 YAML 里的初始姿态和 Fixed PD 参数
            auto default_pos = rl.params.Get<std::vector<float>>("default_dof_pos");
            auto fixed_kp = rl.params.Get<std::vector<float>>("fixed_kp");
            auto fixed_kd = rl.params.Get<std::vector<float>>("fixed_kd");

            for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
            {
                
                fsm_command->motor_command.q[i] = default_pos[i]; 
                fsm_command->motor_command.dq[i] = 0;
                fsm_command->motor_command.kp[i] = fixed_kp[i]; // 不再是 0！
                fsm_command->motor_command.kd[i] = fixed_kd[i]; // 使用 YAML 设定的阻尼
                fsm_command->motor_command.tau[i] = 0;
            }
        }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        return state_name_;
    }
};

class RLFSMStateGetUp : public RLFSMState
{
public:
    RLFSMStateGetUp(RL *rl) : RLFSMState(*rl, "RLFSMStateGetUp") {}

    float percent_getup = 0.0f;

    void Enter() override
    {
        percent_getup = 0.0f;
        rl.now_state = *fsm_state;
        rl.start_state = rl.now_state;
    }

    void Run() override
    {
        const std::vector<float> getup_target =
            rl.params.Has("getup_dof_pos")
                ? rl.params.Get<std::vector<float>>("getup_dof_pos")
                : rl.params.Get<std::vector<float>>("default_dof_pos");
        Interpolate(percent_getup, rl.now_state.motor_state.q, getup_target, 0.05f, "Getting up", true);
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        if (percent_getup >= 1.0f)
        {
            if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
            {
                return "RLFSMStateRLLocomotion";
            }
            else if (rl.control.current_keyboard == Input::Keyboard::Num2 ||
                     rl.control.current_keyboard == Input::Keyboard::Down ||
                     rl.control.current_gamepad == Input::Gamepad::RB_DPadDown)
            {
                std::cout << LOGGER::INFO << "[FSMDebug] GetUp -> Dance trigger detected." << std::endl;
                return "RLFSMStateRLWholeBodyTrackingDance";
            }
            else if (rl.control.current_keyboard == Input::Keyboard::Num3 ||
                     rl.control.current_gamepad == Input::Gamepad::RB_DPadLeft)
            {
                std::cout << LOGGER::INFO << "[FSMDebug] GetUp -> mjlab tracking trigger detected." << std::endl;
                return "RLFSMStateRLMjlabTracking";
            }
            else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
            {
                return "RLFSMStateGetDown";
            }
        }
        return state_name_;
    }
};

class RLFSMStateGetDown : public RLFSMState
{
public:
    RLFSMStateGetDown(RL *rl) : RLFSMState(*rl, "RLFSMStateGetDown") {}

    float percent_getdown = 0.0f;

    void Enter() override
    {
        percent_getdown = 0.0f;
        rl.now_state = *fsm_state;
    }

    void Run() override
    {
        Interpolate(percent_getdown, rl.now_state.motor_state.q, rl.start_state.motor_state.q, 2.0f, "Getting down", true);
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X || percent_getdown >= 1.0f)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        return state_name_;
    }
};

class RLFSMStateRLLocomotion : public RLFSMState
{
public:
    RLFSMStateRLLocomotion(RL *rl) : RLFSMState(*rl, "RLFSMStateRLLocomotion") {}

    float percent_transition = 0.0f;

    void Enter() override
    {
        percent_transition = 0.0f;
        rl.episode_length_buf = 0;
        rl.rl_wait_first_frame = true;
        rl.rl_blend_active = false;
        rl.rl_blend_time = 0.0f;
        rl.rl_blend_duration = 0.2f;
        std::vector<float> hold_q_old_order = fsm_state->motor_state.q;
        std::vector<int> old_joint_mapping = rl.params.Get<std::vector<int>>("joint_mapping");
        rl.hold_q_on_enter = hold_q_old_order;
        rl.rl_blend_q_target = hold_q_old_order;
        rl.rl_blend_dq_target.assign(rl.params.Get<int>("num_of_dofs"), 0.0f);

        std::vector<float> stale_output;
        while (rl.output_dof_pos_queue.try_pop(stale_output)) {}
        while (rl.output_dof_vel_queue.try_pop(stale_output)) {}
        while (rl.output_dof_tau_queue.try_pop(stale_output)) {}

        // read params from yaml
        // config_name is set by rl_real_myrobot.cpp or rl_sim_mujoco.cpp before FSM init
        // default to "robot_lab" for MuJoCo sim, "robot_lab_real" for real robot
        if (rl.config_name.empty() || rl.config_name == "whole_body_tracking")
        {
            rl.config_name = "robot_lab";
        }
        std::string robot_config_path = rl.robot_name + "/" + rl.config_name;
        try
        {
            rl.InitRL(robot_config_path);
            rl.rl_blend_duration = std::max(0.0f, rl.params.Get<float>("rl_blend_duration", 0.2f));
            auto new_joint_mapping = rl.params.Get<std::vector<int>>("joint_mapping");
            const int num_dofs = rl.params.Get<int>("num_of_dofs");

            bool remap_ok =
                hold_q_old_order.size() == static_cast<size_t>(num_dofs) &&
                old_joint_mapping.size() == static_cast<size_t>(num_dofs) &&
                new_joint_mapping.size() == static_cast<size_t>(num_dofs);

            if (remap_ok)
            {
                std::vector<float> hold_q_new_order(num_dofs, 0.0f);
                for (int old_idx = 0; old_idx < num_dofs; ++old_idx)
                {
                    int actuator_id = old_joint_mapping[old_idx];
                    int new_idx = -1;
                    for (int idx = 0; idx < num_dofs; ++idx)
                    {
                        if (new_joint_mapping[idx] == actuator_id)
                        {
                            new_idx = idx;
                            break;
                        }
                    }
                    if (new_idx < 0)
                    {
                        remap_ok = false;
                        break;
                    }
                    hold_q_new_order[new_idx] = hold_q_old_order[old_idx];
                }
                if (remap_ok)
                {
                    rl.hold_q_on_enter = hold_q_new_order;
                }
            }

            if (!remap_ok)
            {
                rl.hold_q_on_enter = hold_q_old_order;
                std::cout << LOGGER::WARNING << "[RLLocomotion] hold_q remap skipped, fallback to previous joint order." << std::endl;
            }

            rl.rl_blend_q_target = rl.hold_q_on_enter;
            rl.rl_blend_dq_target.assign(rl.params.Get<int>("num_of_dofs"), 0.0f);
            rl.now_state = *fsm_state;
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.fsm.RequestStateChange("RLFSMStatePassive");
        }
    }

    void Run() override
    {
        // position transition from last default_dof_pos to current default_dof_pos
        // if (Interpolate(percent_transition, rl.now_state.motor_state.q, rl.params.Get<std::vector<float>>("default_dof_pos"), 0.5f, "Policy transition", true)) return;

        // Handle skill switch even on the first cycle after state transition.
        if (rl.control.current_keyboard == Input::Keyboard::Num2 ||
            rl.control.current_keyboard == Input::Keyboard::Down ||
            rl.control.current_gamepad == Input::Gamepad::RB_DPadDown)
        {
            std::cout << LOGGER::INFO << "[FSMDebug] Locomotion -> Dance trigger detected, requesting switch." << std::endl;
            rl.fsm.RequestStateChange("RLFSMStateRLWholeBodyTrackingDance");
            return;
        }

        if (!rl.rl_init_done) rl.rl_init_done = true;

        std::cout << "\r\033[K" << std::flush << LOGGER::INFO << "RL Controller [" << rl.config_name << "] x:" << rl.control.x << " y:" << rl.control.y << " yaw:" << rl.control.yaw << std::flush;
        RLControl();
    }

    void Exit() override
    {
        rl.rl_init_done = false;
    }

    std::string CheckChange() override
    {
        // Keep dance state even when robot falls, so we can observe policy outputs
        // in out-of-distribution postures. Do not auto-exit on P/LB_X here.
        if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
        {
            return "RLFSMStateGetDown";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
        {
            return "RLFSMStateRLLocomotion";
        } 
        else if (rl.control.current_keyboard == Input::Keyboard::Num2 ||
                 rl.control.current_keyboard == Input::Keyboard::Down ||
                 rl.control.current_gamepad == Input::Gamepad::RB_DPadDown)
        {
            return "RLFSMStateRLWholeBodyTrackingDance";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num3 ||
                 rl.control.current_gamepad == Input::Gamepad::RB_DPadLeft)
        {
            return "RLFSMStateRLMjlabTracking";
        }
        return state_name_;
    }
};
class RLFSMStateRLWholeBodyTrackingDance : public RLFSMState
{
public:
    RLFSMStateRLWholeBodyTrackingDance(RL *rl) : RLFSMState(*rl, "RLFSMStateRLWholeBodyTrackingDance") {}

    void Enter() override
    {
        rl.episode_length_buf = 0;
        rl.rl_wait_first_frame = true;
        rl.rl_blend_active = false;
        rl.rl_blend_time = 0.0f;
        rl.rl_blend_duration = 0.2f;
        std::vector<float> hold_q_old_order = fsm_state->motor_state.q;
        std::vector<int> old_joint_mapping = rl.params.Get<std::vector<int>>("joint_mapping");
        rl.hold_q_on_enter = hold_q_old_order;
        rl.rl_blend_q_target = hold_q_old_order;
        rl.rl_blend_dq_target.assign(rl.params.Get<int>("num_of_dofs"), 0.0f);

        std::vector<float> stale_output;
        while (rl.output_dof_pos_queue.try_pop(stale_output)) {}
        while (rl.output_dof_vel_queue.try_pop(stale_output)) {}
        while (rl.output_dof_tau_queue.try_pop(stale_output)) {}

        // read params from yaml
        rl.config_name = "whole_body_tracking";
        std::string robot_config_path = rl.robot_name + "/" + rl.config_name;
        try
        {
            rl.InitRL(robot_config_path);
            // Dance smooth-start guard:
            // even when generic rl_blend_duration is 0 for ablation, keep a
            // minimum startup blend window to avoid first-frame target jump.
            const float rl_blend_duration_cfg =
                std::max(0.0f, rl.params.Get<float>("rl_blend_duration", 0.2f));
            const float dance_min_startup_blend_duration =
                std::max(0.0f, rl.params.Get<float>("dance_min_startup_blend_duration", 0.25f));
            rl.rl_blend_duration = std::max(rl_blend_duration_cfg, dance_min_startup_blend_duration);
            auto new_joint_mapping = rl.params.Get<std::vector<int>>("joint_mapping");
            const int num_dofs = rl.params.Get<int>("num_of_dofs");

            // Pure sim2sim deployment mode: no reference motion file dependency.
            rl.motion_loader.reset();
            rl.motion_length = 0.0f;
            if (rl.params.Get<int>("max_episode_length", -1) <= 1)
            {
                const float step_time = rl.GetPolicyStepTime();
                const float episode_duration_s = rl.params.Get<float>("episode_duration_seconds", 22.0f);
                const int max_episode_length = std::max(2, static_cast<int>(std::round(episode_duration_s / std::max(step_time, 1e-6f))));
                rl.params.config_node["max_episode_length"] = max_episode_length;
            }
            std::cout << LOGGER::INFO << "Dance episode horizon: "
                      << rl.params.Get<int>("max_episode_length", 0) << " steps @ "
                      << (1.0f / std::max(rl.GetPolicyStepTime(), 1e-6f)) << " Hz" << std::endl;

            const bool pose_from_amp = rl.ApplyWholeBodyTrackingInitPoseFromAmpObs();
            if (!pose_from_amp)
            {
                bool remap_ok =
                    hold_q_old_order.size() == static_cast<size_t>(num_dofs) &&
                    old_joint_mapping.size() == static_cast<size_t>(num_dofs) &&
                    new_joint_mapping.size() == static_cast<size_t>(num_dofs);

                if (remap_ok)
                {
                    std::vector<float> hold_q_new_order(num_dofs, 0.0f);
                    for (int old_idx = 0; old_idx < num_dofs; ++old_idx)
                    {
                        int actuator_id = old_joint_mapping[old_idx];
                        int new_idx = -1;
                        for (int idx = 0; idx < num_dofs; ++idx)
                        {
                            if (new_joint_mapping[idx] == actuator_id)
                            {
                                new_idx = idx;
                                break;
                            }
                        }
                        if (new_idx < 0)
                        {
                            remap_ok = false;
                            break;
                        }
                        hold_q_new_order[new_idx] = hold_q_old_order[old_idx];
                    }
                    if (remap_ok)
                    {
                        rl.hold_q_on_enter = hold_q_new_order;
                    }
                }

                if (!remap_ok)
                {
                    rl.hold_q_on_enter = hold_q_old_order;
                    std::cout << LOGGER::WARNING << "[WholeBodyTrackingDance] hold_q remap skipped, fallback to previous joint order." << std::endl;
                }
                rl.rl_blend_q_target = rl.hold_q_on_enter;
                rl.rl_blend_dq_target.assign(rl.params.Get<int>("num_of_dofs"), 0.0f);
            }

            rl.now_state = *fsm_state;
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.fsm.RequestStateChange("RLFSMStatePassive");
        }
    }

    void Run() override
    {
        // position transition from last default_dof_pos to current default_dof_pos
        // if (Interpolate(percent_transition, rl.now_state.motor_state.q, rl.params.Get<std::vector<float>>("default_dof_pos"), 0.5f, "Policy transition", true)) return;

        if (!rl.rl_init_done) rl.rl_init_done = true;

        // Print episode progress (sim2sim deployment), independent from any motion file.
        float percent = 0.0f;
        const int max_episode_length = rl.params.Get<int>("max_episode_length", -1);
        if (max_episode_length > 1)
        {
            percent = std::clamp(static_cast<float>(rl.episode_length_buf) /
                                 static_cast<float>(max_episode_length - 1), 0.0f, 1.0f);
        }
        LOGGER::PrintProgress(percent, rl.config_name);

        RLControl();

        // Auto-exit on playback completion is handled in CheckChange() below.
        // Manual keyboard/gamepad transitions still take precedence.
    }

    void Exit() override
    {
        rl.rl_init_done = false;
    }

    // -----------------------------------------------------------------------
    // CheckChange policy:
    //   1) Manual transitions (P / B / A / DPad) keep highest priority so that
    //      the operator can always interrupt the dance.
    //   2) If no manual input arrives and the episode buffer reaches
    //      max_episode_length (i.e. policy has played the full motion length
    //      it was trained on), auto-exit to GetDown.
    //      Rationale: keeping the FSM in Dance after the trained horizon
    //      pushes `progress` obs to a clamp(1.0) value that lies outside the
    //      training distribution. On hardware this OOD input drives the
    //      policy to emit saturating commands (sh_roll_L jumping ±soft_limit
    //      every frame). GetDown then performs a 2s interpolation back to
    //      start_state.motor_state.q and auto-switches to Passive, so the
    //      handover is mechanically smooth.
    // -----------------------------------------------------------------------
    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
        {
            return "RLFSMStateGetDown";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
        {
            return "RLFSMStateRLLocomotion";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num2 ||
                 rl.control.current_keyboard == Input::Keyboard::Down ||
                 rl.control.current_gamepad == Input::Gamepad::RB_DPadDown)
        {
            return "RLFSMStateRLWholeBodyTrackingDance";
        }

        const int max_episode_length = rl.params.Get<int>("max_episode_length", -1);
        if (max_episode_length > 1 &&
            static_cast<long long>(rl.episode_length_buf) >= static_cast<long long>(max_episode_length))
        {
            std::cout << "\n" << LOGGER::INFO
                      << "[Dance] episode completed (ep_step=" << rl.episode_length_buf
                      << ", max=" << max_episode_length << "), auto-exiting to GetDown."
                      << std::endl;
            return "RLFSMStateGetDown";
        }
        return state_name_;
    }
};

// ---------------------------------------------------------------------------
// mjlab whole-body tracking 部署状态（Num3 / 手柄 RB_DPadLeft 进入）。
//
// 主要功能：加载 mjlab tracking ONNX（config_name="mjlab"），抽取 baked motion 表，
//           以 50Hz 按相位跟踪整段参考动作，播满 852 帧自动退出到 GetDown。
// 运行逻辑：骨架与 WholeBodyTrackingDance 类似（hold->blend->策略接管），但：
//   - 不依赖 MotionLoader / ampobs.csv / sim2sim init pose（mjlab 直接在 MuJoCo 训练）；
//   - 观测/动作/频率全部由 mjlab/config.yaml 决定（114 维 / default_position / 50Hz）；
//   - 参考动作来自 RL::BuildMjlabMotionTable() 缓存表，由 ComputeObservation 按 t 查表。
// 使用方法：GetUp 到 getup_dof_pos（base.yaml，tabu motion 帧 0）后按 Num3 进入。
// ---------------------------------------------------------------------------
class RLFSMStateRLMjlabTracking : public RLFSMState
{
public:
    RLFSMStateRLMjlabTracking(RL *rl) : RLFSMState(*rl, "RLFSMStateRLMjlabTracking") {}

    void Enter() override
    {
        rl.episode_length_buf = 0;
        rl.rl_wait_first_frame = true;
        rl.rl_blend_active = false;
        rl.rl_blend_time = 0.0f;
        rl.rl_blend_duration = 0.2f;
        std::vector<float> hold_q_old_order = fsm_state->motor_state.q;
        std::vector<int> old_joint_mapping = rl.params.Get<std::vector<int>>("joint_mapping");
        rl.hold_q_on_enter = hold_q_old_order;
        rl.rl_blend_q_target = hold_q_old_order;
        rl.rl_blend_dq_target.assign(rl.params.Get<int>("num_of_dofs"), 0.0f);

        std::vector<float> stale_output;
        while (rl.output_dof_pos_queue.try_pop(stale_output)) {}
        while (rl.output_dof_vel_queue.try_pop(stale_output)) {}
        while (rl.output_dof_tau_queue.try_pop(stale_output)) {}

        rl.config_name = "mjlab";
        std::string robot_config_path = rl.robot_name + "/" + rl.config_name;
        try
        {
            rl.InitRL(robot_config_path);

            // 启动平滑窗口：即使 rl_blend_duration 为 0，也保留一个最小 blend 窗口，
            // 避免策略接管首帧 q_target 跳变。
            const float rl_blend_duration_cfg =
                std::max(0.0f, rl.params.Get<float>("rl_blend_duration", 0.2f));
            const float min_startup_blend_duration =
                std::max(0.0f, rl.params.Get<float>("dance_min_startup_blend_duration", 0.25f));
            rl.rl_blend_duration = std::max(rl_blend_duration_cfg, min_startup_blend_duration);

            auto new_joint_mapping = rl.params.Get<std::vector<int>>("joint_mapping");
            const int num_dofs = rl.params.Get<int>("num_of_dofs");

            // hold_q 关节顺序重映射（mjlab 与 base 均为恒等映射时为 no-op，仍保留以防 config 切换）。
            bool remap_ok =
                hold_q_old_order.size() == static_cast<size_t>(num_dofs) &&
                old_joint_mapping.size() == static_cast<size_t>(num_dofs) &&
                new_joint_mapping.size() == static_cast<size_t>(num_dofs);
            if (remap_ok)
            {
                std::vector<float> hold_q_new_order(num_dofs, 0.0f);
                for (int old_idx = 0; old_idx < num_dofs; ++old_idx)
                {
                    int actuator_id = old_joint_mapping[old_idx];
                    int new_idx = -1;
                    for (int idx = 0; idx < num_dofs; ++idx)
                    {
                        if (new_joint_mapping[idx] == actuator_id) { new_idx = idx; break; }
                    }
                    if (new_idx < 0) { remap_ok = false; break; }
                    hold_q_new_order[new_idx] = hold_q_old_order[old_idx];
                }
                if (remap_ok) rl.hold_q_on_enter = hold_q_new_order;
            }
            if (!remap_ok)
            {
                rl.hold_q_on_enter = hold_q_old_order;
                std::cout << LOGGER::WARNING << "[MjlabTracking] hold_q remap skipped, fallback to previous joint order." << std::endl;
            }
            rl.rl_blend_q_target = rl.hold_q_on_enter;
            rl.rl_blend_dq_target.assign(num_dofs, 0.0f);

            std::cout << LOGGER::INFO << "mjlab tracking horizon: "
                      << rl.params.Get<int>("max_episode_length", 0) << " steps @ "
                      << (1.0f / std::max(rl.GetPolicyStepTime(), 1e-6f)) << " Hz" << std::endl;

            rl.now_state = *fsm_state;
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.fsm.RequestStateChange("RLFSMStatePassive");
        }
    }

    void Run() override
    {
        if (!rl.rl_init_done) rl.rl_init_done = true;

        float percent = 0.0f;
        const int max_episode_length = rl.params.Get<int>("max_episode_length", -1);
        if (max_episode_length > 1)
        {
            percent = std::clamp(static_cast<float>(rl.episode_length_buf) /
                                 static_cast<float>(max_episode_length - 1), 0.0f, 1.0f);
        }
        LOGGER::PrintProgress(percent, rl.config_name);

        RLControl();
    }

    void Exit() override
    {
        rl.rl_init_done = false;
    }

    std::string CheckChange() override
    {
        // 手动切换优先级最高，操作者随时可中断跟踪。
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
        {
            return "RLFSMStateGetDown";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
        {
            return "RLFSMStateRLLocomotion";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num2 ||
                 rl.control.current_keyboard == Input::Keyboard::Down ||
                 rl.control.current_gamepad == Input::Gamepad::RB_DPadDown)
        {
            return "RLFSMStateRLWholeBodyTrackingDance";
        }

        // 播满整段 motion（episode_length_buf >= max_episode_length，即已用完 t=0..N-1）
        // 后自动退出到 GetDown：继续停留会把相位 clamp 在末帧，属训练分布外。
        const int max_episode_length = rl.params.Get<int>("max_episode_length", -1);
        if (max_episode_length > 1 &&
            static_cast<long long>(rl.episode_length_buf) >= static_cast<long long>(max_episode_length))
        {
            std::cout << "\n" << LOGGER::INFO
                      << "[MjlabTracking] motion completed (ep_step=" << rl.episode_length_buf
                      << ", max=" << max_episode_length << "), auto-exiting to GetDown." << std::endl;
            return "RLFSMStateGetDown";
        }
        return state_name_;
    }
};

} // namespace myrobot_fsm

class MYROBOTFSMFactory : public FSMFactory
{
public:
    MYROBOTFSMFactory(const std::string& initial) : initial_state_(initial) {}
    std::shared_ptr<FSMState> CreateState(void *context, const std::string &state_name) override
    {
        RL *rl = static_cast<RL *>(context);
        if (state_name == "RLFSMStatePassive")
            return std::make_shared<myrobot_fsm::RLFSMStatePassive>(rl);
        else if (state_name == "RLFSMStateGetUp")
            return std::make_shared<myrobot_fsm::RLFSMStateGetUp>(rl);
        else if (state_name == "RLFSMStateGetDown")
            return std::make_shared<myrobot_fsm::RLFSMStateGetDown>(rl);
        else if (state_name == "RLFSMStateRLLocomotion")
            return std::make_shared<myrobot_fsm::RLFSMStateRLLocomotion>(rl);
        else if (state_name == "RLFSMStateRLWholeBodyTrackingDance")
            return std::make_shared<myrobot_fsm::RLFSMStateRLWholeBodyTrackingDance>(rl);
        else if (state_name == "RLFSMStateRLMjlabTracking")
            return std::make_shared<myrobot_fsm::RLFSMStateRLMjlabTracking>(rl);
        return nullptr;
    }
    std::string GetType() const override { return "myrobot"; }
    std::vector<std::string> GetSupportedStates() const override
    {
        return {
            "RLFSMStatePassive",
            "RLFSMStateGetUp",
            "RLFSMStateGetDown",
            "RLFSMStateRLLocomotion",
            "RLFSMStateRLWholeBodyTrackingDance",
            "RLFSMStateRLMjlabTracking"
        };
    }
    std::string GetInitialState() const override { return initial_state_; }
private:
    std::string initial_state_;
};

REGISTER_FSM_FACTORY(MYROBOTFSMFactory, "RLFSMStatePassive")

#endif // GR1T1_FSM_HPP
