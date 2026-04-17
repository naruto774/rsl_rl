/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MYROBOT_FSM_HPP
#define MYROBOT_FSM_HPP

#include "fsm.hpp"
#include "rl_sdk.hpp"

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
        Interpolate(percent_getup, rl.now_state.motor_state.q, rl.params.Get<std::vector<float>>("default_dof_pos"), 0.05f, "Getting up", true);
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
        if (rl.config_name.empty())
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
        return nullptr;
    }
    std::string GetType() const override { return "myrobot"; }
    std::vector<std::string> GetSupportedStates() const override
    {
        return {
            "RLFSMStatePassive",
            "RLFSMStateGetUp",
            "RLFSMStateGetDown",
            "RLFSMStateRLLocomotion"
        };
    }
    std::string GetInitialState() const override { return initial_state_; }
private:
    std::string initial_state_;
};

REGISTER_FSM_FACTORY(MYROBOTFSMFactory, "RLFSMStatePassive")

#endif // GR1T1_FSM_HPP
