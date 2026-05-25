/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sdk.hpp"
#include "kinematics_fk.hpp"
#include <iomanip>
#include <filesystem>

// Out-of-line ctor/dtor 让 unique_ptr<KinematicsFK> 能在这里看到完整类型，
// 否则 hpp 里的 forward-decl + inline dtor 会触发 sizeof incomplete type 报错。
RL::RL() = default;
RL::~RL() = default;

bool RL::LoadFkModel(const std::string& mjcf_path,
                    const std::vector<int>& joint_mapping,
                    const std::string& root_body_name)
{
    if (!this->fk_helper_)
    {
        this->fk_helper_ = std::make_unique<KinematicsFK>();
    }
    return this->fk_helper_->Load(mjcf_path, joint_mapping, root_body_name);
}

bool RL::IsFkAvailable() const
{
    return this->fk_helper_ != nullptr && this->fk_helper_->IsLoaded();
}

bool RL::ComputeKeyBodyPosFK()
{
    if (!IsFkAvailable()) return false;
    // 复用 obs.dof_pos / obs.base_quat（RunModel 在调用本函数前已经填好）
    return this->fk_helper_->ComputeKeyBodyPosRel(
        this->obs.dof_pos, this->obs.base_quat, this->obs.key_body_pos_rel);
}
void RL::StateController(const RobotState<float>* state, RobotCommand<float>* command)
{
    auto updateState = [&](std::shared_ptr<FSMState> statePtr)
    {
        if (auto rl_fsm_state = std::dynamic_pointer_cast<RLFSMState>(statePtr))
        {
            rl_fsm_state->fsm_state = state;
            rl_fsm_state->fsm_command = command;
        }
    };
    for (auto& pair : fsm.states_)
    {
        updateState(pair.second);
    }

    fsm.Run();

    this->motiontime++;

    if (this->control.current_keyboard == Input::Keyboard::W)
    {
        this->control.x += 0.01f;
    }
    if (this->control.current_keyboard == Input::Keyboard::S)
    {
        this->control.x -= 0.01f;
    }
    if (this->control.current_keyboard == Input::Keyboard::A)
    {
        this->control.y += 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::D)
    {
        this->control.y -= 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::Q)
    {
        this->control.yaw += 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::E)
    {
        this->control.yaw -= 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::Space)
    {
        this->control.x = 0.0f;
        this->control.y = 0.0f;
        this->control.yaw = 0.0f;
    }
    if (this->control.current_keyboard == Input::Keyboard::N || this->control.current_gamepad == Input::Gamepad::X)
    {
        this->control.navigation_mode = !this->control.navigation_mode;
        std::cout << std::endl << LOGGER::INFO << "Navigation mode: " << (this->control.navigation_mode ? "ON" : "OFF") << std::endl;
    }
}

std::vector<float> RL::ComputeObservation()
{
    std::vector<std::vector<float>> obs_list;

    for (const std::string &observation : this->params.Get<std::vector<std::string>>("observations"))
    {
        // ============= Base Observations =============
        if (observation == "lin_vel")
        {
            obs_list.push_back(this->obs.lin_vel * this->params.Get<float>("lin_vel_scale"));
        }
        else if (observation == "ang_vel")
        {
            // In ROS1 Gazebo, the coordinate system for angular velocity is in the world coordinate system.
            // In ROS2 Gazebo, mujoco and real robot, the coordinate system for angular velocity is in the body coordinate system.
            if (this->ang_vel_axis == "body")
            {
                obs_list.push_back(this->obs.ang_vel * this->params.Get<float>("ang_vel_scale"));
            }
            else if (this->ang_vel_axis == "world")
            {
                obs_list.push_back(QuatRotateInverse(this->obs.base_quat, this->obs.ang_vel) * this->params.Get<float>("ang_vel_scale"));
            }
        }
        else if (observation == "gravity_vec")
        {
            // Sim-to-Real: 投影后再加均匀噪声，保证与 IsaacLab 中 projected_gravity 的
            // 噪声模型一致（U(-a,a)，a≈0.04）。对 base_quat 直接加噪会破坏单位模长，
            // 并且 0.04 描述的是“投影后分量”的不确定度，不是四元数分量的不确定度。
            // 实物端 yaml 未配置该字段 → Get(..,0.0f) 默认 0 → 自动禁用。
            auto proj_g = QuatRotateInverse(this->obs.base_quat, this->obs.gravity_vec);
            AddUniformNoiseInPlace(proj_g, this->params.Get<float>("noise_projected_gravity", 0.0f));
            obs_list.push_back(proj_g);
        }
        else if (observation == "commands")
        {
            obs_list.push_back(this->obs.commands * this->params.Get<std::vector<float>>("commands_scale"));
        }
        else if (observation == "dof_pos")
        {
            std::vector<float> dof_pos_rel = this->obs.dof_pos - this->params.Get<std::vector<float>>("default_dof_pos");
            for (int i : this->params.Get<std::vector<int>>("wheel_indices"))
            {
                dof_pos_rel[i] = 0.0f;
            }
            obs_list.push_back(dof_pos_rel * this->params.Get<float>("dof_pos_scale"));
        }
        else if (observation == "dof_vel")
        {
            obs_list.push_back(this->obs.dof_vel * this->params.Get<float>("dof_vel_scale"));
        }
        else if (observation == "actions")
        {
            obs_list.push_back(this->obs.actions);
        }
        else if (observation == "actions_history")
        {
            // Sim-to-Real: 动作历史堆叠（actuator delay robustness）
            // 数学：RobotLab / IsaacLab ObservationManager flatten_history_dim=True
            // 按 oldest -> newest 输出历史：
            //   o_{actions_history}_t = [a_{t-K}, ..., a_{t-2}, a_{t-1}]
            // 内部 deque 仍用 push_front 维护 newest-first，因此这里反向读出以对齐训练观测空间。
            const int K = this->params.Get<int>("actions_history_length", 1);
            const int n = this->params.Get<int>("num_of_dofs");
            std::vector<float> stacked;
            stacked.reserve(static_cast<size_t>(K) * static_cast<size_t>(n));
            for (int k = K - 1; k >= 0; --k)
            {
                if (k < static_cast<int>(this->obs.actions_history.size()))
                {
                    const auto& a = this->obs.actions_history[k];
                    // 防御性：若历史帧维度异常，按 num_of_dofs 截断/补零，保证 obs 维度恒定
                    if (static_cast<int>(a.size()) == n)
                    {
                        stacked.insert(stacked.end(), a.begin(), a.end());
                    }
                    else
                    {
                        std::vector<float> padded(n, 0.0f);
                        const int copy_n = std::min(static_cast<int>(a.size()), n);
                        for (int i = 0; i < copy_n; ++i) padded[i] = a[i];
                        stacked.insert(stacked.end(), padded.begin(), padded.end());
                    }
                }
                else
                {
                    // 历史不足 K 帧时按 0 padding（episode 起始 K 步内的暖启动）
                    stacked.insert(stacked.end(), n, 0.0f);
                }
            }
            obs_list.push_back(stacked);
        }
        // ============= Other Observations =============
        else if (observation == "whole_body_tracking/joint_pos")
        {
            obs_list.push_back(this->obs.dof_pos);
        }
        else if (observation == "whole_body_tracking/joint_vel")
        {
            obs_list.push_back(this->obs.dof_vel);
        }
        else if (observation == "whole_body_tracking/root_pos_relative")
        {
            // Align with training: use current robot root height (z).
            // If backend does not provide base_pos, keep a stable zero fallback.
            float root_z = 0.0f;
            if (this->obs.base_pos.size() >= 3)
            {
                root_z = this->obs.base_pos[2];
            }
            obs_list.push_back({root_z});
        }
        else if (observation == "whole_body_tracking/ref_body_quat_tan_norm")
        {
            // Isaac humanoid_amp: quaternion_to_tangent_and_normal
            // tangent = quat_apply(q, [1,0,0]), normal = quat_apply(q, [0,0,1])
            std::vector<float> root_rot_6d(6, 0.0f);
            if (this->obs.base_quat.size() == 4)
            {
                const std::vector<float> tangent = QuatApply(this->obs.base_quat, {1.0f, 0.0f, 0.0f});
                const std::vector<float> normal = QuatApply(this->obs.base_quat, {0.0f, 0.0f, 1.0f});
                root_rot_6d = {
                    tangent[0], tangent[1], tangent[2],
                    normal[0], normal[1], normal[2]
                };

                if (this->params.Get<bool>("root_rot_6d_swap_xy", false))
                {
                    std::swap(root_rot_6d[0], root_rot_6d[1]);
                    std::swap(root_rot_6d[3], root_rot_6d[4]);
                }
            }
            obs_list.push_back(root_rot_6d);
        }
        else if (observation == "whole_body_tracking/key_body_pos_relative")
        {
            const std::vector<int> default_key_body_joint_indices = {
                9, 10, 13, 14, 17, 18, 20, 19, 5, 7, 6, 12, 11
            };
            auto key_body_joint_indices = this->params.Get<std::vector<int>>(
                "key_body_joint_indices", default_key_body_joint_indices
            );
            std::vector<float> key_body_pos_relative;
            key_body_pos_relative.reserve(key_body_joint_indices.size() * 3);

            const bool has_key_body_cache =
                !this->obs.key_body_pos_rel.empty() &&
                this->obs.key_body_pos_rel.size() >= static_cast<size_t>(this->params.Get<int>("num_of_dofs")) * 3;
            if (has_key_body_cache)
            {
                for (int joint_idx : key_body_joint_indices)
                {
                    const int base = joint_idx * 3;
                    if (joint_idx >= 0 &&
                        base + 2 < static_cast<int>(this->obs.key_body_pos_rel.size()))
                    {
                        key_body_pos_relative.push_back(this->obs.key_body_pos_rel[base + 0]);
                        key_body_pos_relative.push_back(this->obs.key_body_pos_rel[base + 1]);
                        key_body_pos_relative.push_back(this->obs.key_body_pos_rel[base + 2]);
                    }
                    else
                    {
                        key_body_pos_relative.insert(key_body_pos_relative.end(), {0.0f, 0.0f, 0.0f});
                    }
                }
            }
            else
            {
                key_body_pos_relative.assign(key_body_joint_indices.size() * 3, 0.0f);
            }

            obs_list.push_back(key_body_pos_relative);
        }
        else if (observation == "whole_body_tracking/progress")
        {
            // H-1: training step k (1-indexed in episode_length_buf after `+=1` at RunModel
            // entry) computes obs BEFORE physics step. The very first policy.forward should see
            // progress=0 paired with the *initial* physical state, just like Isaac at reset.
            // Previous formula `episode_length_buf / (max-1)` was off-by-one (1/(max-1) at step 1).
            float progress = 0.0f;
            const int max_episode_length = this->params.Get<int>("max_episode_length", -1);
            if (max_episode_length > 1 && this->episode_length_buf >= 1)
            {
                progress = static_cast<float>(this->episode_length_buf - 1) /
                           static_cast<float>(max_episode_length - 1);
            }
            progress = std::clamp(progress, 0.0f, 1.0f);
            obs_list.push_back({progress});
        }
        else if (observation == "whole_body_tracking/motion_command")
        {
            std::vector<float> motion_cmd;
            if (this->motion_loader)
            {
                auto joint_pos_sdk = this->motion_loader->GetJointPos();
                auto joint_vel_sdk = this->motion_loader->GetJointVel();
                auto joint_mapping = this->params.Get<std::vector<int>>("joint_mapping");
                std::vector<float> joint_pos_training(joint_mapping.size());
                std::vector<float> joint_vel_training(joint_mapping.size());
                for (size_t i = 0; i < joint_mapping.size(); ++i)
                {
                    joint_pos_training[i] = joint_pos_sdk[joint_mapping[i]];
                    joint_vel_training[i] = joint_vel_sdk[joint_mapping[i]];
                }
                motion_cmd.insert(motion_cmd.end(), joint_pos_training.begin(), joint_pos_training.end());
                motion_cmd.insert(motion_cmd.end(), joint_vel_training.begin(), joint_vel_training.end());
            }
            else
            {
                motion_cmd.resize(this->params.Get<int>("num_of_dofs") * 2, 0.0f);
            }
            obs_list.push_back(motion_cmd);
        }
        else if (observation == "whole_body_tracking/motion_anchor_ori_b")
        {
            std::vector<float> anchor_ori(6, 0.0f);
            if (this->motion_loader)
            {
                auto waist_sdk_indices = this->params.Get<std::vector<int>>("waist_joint_indices");
                // MotionLoader::ComputeTorsoQuat expects [yaw, roll, pitch].
                // Some robots only provide 2 waist DoFs in config; in that case we pad the missing axis with 0.
                std::vector<float> waist_angles(3, 0.0f);
                for (size_t i = 0; i < waist_sdk_indices.size() && i < 3; ++i)
                {
                    int idx = InverseJointMapping(waist_sdk_indices[i]);
                    if (idx >= 0 && idx < static_cast<int>(this->obs.dof_pos.size()))
                    {
                        waist_angles[i] = this->obs.dof_pos[idx];
                    }
                }
                std::vector<float> robot_torso_quat_w = MotionLoader::ComputeTorsoQuat(this->obs.base_quat, waist_angles);
                std::vector<float> ref_torso_quat_w = this->motion_loader->GetAnchorQuat();
                std::vector<float> init_quat = this->motion_loader->GetInitQuat();
                std::vector<float> motion_anchor_quat_w = QuaternionMultiply(init_quat, ref_torso_quat_w);
                std::vector<float> robot_quat_inv = QuaternionConjugate(robot_torso_quat_w);
                std::vector<float> relative_quat = QuaternionMultiply(robot_quat_inv, motion_anchor_quat_w);
                std::vector<float> rot_matrix = QuaternionToRotationMatrix(relative_quat);
                anchor_ori = MatrixFirstTwoColumns(rot_matrix);
            }
            obs_list.push_back(anchor_ori);
        }
        else if (observation == "RoboMimic_Deploy/phase")
        {
            float motion_time = this->episode_length_buf * this->params.Get<float>("dt") * this->params.Get<int>("decimation");
            float count = motion_time;
            float phase = count / this->motion_length;
            std::vector<float> phase_vec = {phase};
            obs_list.push_back(phase_vec);
        }
    }

    this->obs_dims.clear();
    for (const auto& obs : obs_list)
    {
       this->obs_dims.push_back(obs.size());
    }

    std::vector<float> obs;
    for (const auto& obs_vec : obs_list)
    {
        obs.insert(obs.end(), obs_vec.begin(), obs_vec.end());
    }
    std::vector<float> clamped_obs = clamp(obs, -this->params.Get<float>("clip_obs"), this->params.Get<float>("clip_obs"));
    return clamped_obs;
}

void RL::InitObservations()
{
    this->obs.lin_vel = {0.0f, 0.0f, 0.0f};
    this->obs.ang_vel = {0.0f, 0.0f, 0.0f};
    this->obs.gravity_vec = {0.0f, 0.0f, -1.0f};
    this->obs.commands = {0.0f, 0.0f, 0.0f};
    this->obs.base_quat = {0.0f, 0.0f, 0.0f, 1.0f};
    this->obs.base_pos = {0.0f, 0.0f, 0.0f};
    this->obs.dof_pos = this->params.Get<std::vector<float>>("default_dof_pos");
    this->obs.dof_vel.clear();
    this->obs.dof_vel.resize(this->params.Get<int>("num_of_dofs"), 0.0f);
    this->obs.key_body_pos_rel.clear();
    this->obs.key_body_pos_rel.resize(this->params.Get<int>("num_of_dofs") * 3, 0.0f);
    this->obs.actions.clear();
    this->obs.actions.resize(this->params.Get<int>("num_of_dofs"), 0.0f);
    // 动作历史按 K 帧零向量初始化，保证首次 ComputeObservation 维度立即就位
    this->ClearActionsHistory();
    this->ComputeObservation();
}

void RL::ClearActionsHistory()
{
    // K 缺省为 1：旧 policy（observations 含 "actions" 而非 "actions_history"）下退化为单帧，行为不变
    const int K = this->params.Get<int>("actions_history_length", 1);
    const int n = this->params.Get<int>("num_of_dofs");
    this->obs.actions_history.clear();
    for (int k = 0; k < K; ++k)
    {
        this->obs.actions_history.emplace_back(n, 0.0f);
    }
}

void RL::InitOutputs()
{
    int num_of_dofs = this->params.Get<int>("num_of_dofs");
    this->output_dof_tau.clear();
    this->output_dof_tau.resize(num_of_dofs, 0.0f);
    this->output_dof_pos = this->params.Get<std::vector<float>>("default_dof_pos");
    this->output_dof_vel.clear();
    this->output_dof_vel.resize(num_of_dofs, 0.0f);
    this->output_dof_delta.clear();
    this->output_dof_delta.resize(num_of_dofs, 0.0f);
}

void RL::InitControl()
{
    this->control.x = 0.0f;
    this->control.y = 0.0f;
    this->control.yaw = 0.0f;
}

void RL::InitJointNum(size_t num_joints)
{
    this->robot_state.motor_state.resize(num_joints);
    this->start_state.motor_state.resize(num_joints);
    this->now_state.motor_state.resize(num_joints);
    this->robot_command.motor_command.resize(num_joints);
}

void RL::InitRL(std::string robot_config_path)
{
    std::lock_guard<std::mutex> lock(this->model_mutex);

    this->ReadYaml(robot_config_path, "config.yaml");

    // init joint num first
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));

    // init rl
    this->InitObservations();
    this->InitOutputs();
    this->InitControl();

    // init obs history
    const auto& observations_history = this->params.Get<std::vector<int>>("observations_history");  // avoid dangling reference
    if (!observations_history.empty())
    {
        int history_length = *std::max_element(observations_history.begin(), observations_history.end()) + 1;
        this->history_obs_buf = ObservationBuffer(1, this->obs_dims, history_length, this->params.Get<std::string>("observations_history_priority"));
    }

    // init model
    std::string model_path = std::string(POLICY_DIR) + "/" + robot_config_path + "/" + this->params.Get<std::string>("model_name");
    std::filesystem::path model_fs_path(model_path);
    if (!std::filesystem::exists(model_fs_path))
    {
        throw std::runtime_error("Model file does not exist: " + model_path);
    }
    std::error_code ec;
    const auto model_size = std::filesystem::file_size(model_fs_path, ec);
    if (!ec)
    {
        std::cout << LOGGER::INFO
                  << "[InitRL] Loading model: " << model_path
                  << " (" << model_size << " bytes)" << std::endl;
    }
    else
    {
        std::cout << LOGGER::INFO
                  << "[InitRL] Loading model: " << model_path
                  << " (size unavailable)" << std::endl;
    }
    this->model = InferenceRuntime::ModelFactory::load_model(model_path);
    if (!this->model)
    {
        throw std::runtime_error(
            "Failed to load model from: " + model_path +
            " (check runtime log for Torch/ONNX backend error details)"
        );
    }

    this->OnPolicyConfigLoaded();
}

float RL::GetPolicyStepTime() const
{
    if (this->params.Has("policy_step_time"))
    {
        return this->params.Get<float>("policy_step_time");
    }
    return this->params.Get<float>("dt") * static_cast<float>(this->params.Get<int>("decimation"));
}

void RL::ComputeOutput(const std::vector<float> &actions, std::vector<float> &output_dof_pos, std::vector<float> &output_dof_vel, std::vector<float> &output_dof_tau)
{
    const int num_dofs = this->params.Get<int>("num_of_dofs");
    const bool disable_action_clip = this->params.Get<bool>("disable_action_clip", false);
    std::vector<float> actions_scaled = actions * this->params.Get<std::vector<float>>("action_scale");
    // Align with training-side processing order: scale first, then clip.
    std::vector<float> actions_processed = actions_scaled;
    const auto clip_lower = this->params.Get<std::vector<float>>("clip_actions_lower");
    const auto clip_upper = this->params.Get<std::vector<float>>("clip_actions_upper");
    if (!disable_action_clip && !clip_lower.empty() && !clip_upper.empty())
    {
        actions_processed = clamp(actions_processed, clip_lower, clip_upper);
    }

    // Use true actuator state for control targets/errors (not noisy observation tensors).
    const std::vector<float>& q_current = this->robot_state.motor_state.q;
    const std::vector<float>& dq_current = this->robot_state.motor_state.dq;

    // Optional post-processing safety (opt-in by config):
    // 1) q_target clamp to configured lower/upper bounds
    // 2) per-step rate limit: |q_target - q_current| <= max_step
    // 3) transition ramp: q_target <- q_current + alpha * (q_target - q_current)
    auto apply_q_target_safety = [&](std::vector<float>& q_target) {
        if (static_cast<int>(q_target.size()) != num_dofs) return;

        if (this->params.Get<bool>("enable_q_target_clamp", false))
        {
            auto q_lower = this->params.Get<std::vector<float>>("q_target_lower");
            auto q_upper = this->params.Get<std::vector<float>>("q_target_upper");
            if (static_cast<int>(q_lower.size()) == num_dofs && static_cast<int>(q_upper.size()) == num_dofs)
            {
                q_target = clamp(q_target, q_lower, q_upper);
            }
        }

        if (this->params.Get<bool>("enable_q_target_rate_limit", false))
        {
            std::vector<float> max_step = this->params.Get<std::vector<float>>("q_target_max_step");
            if (max_step.empty())
            {
                const float step_scalar = this->params.Get<float>("q_target_max_step_scalar", 0.0f);
                max_step.assign(num_dofs, step_scalar);
            }
            if (static_cast<int>(max_step.size()) == num_dofs &&
                static_cast<int>(q_current.size()) == num_dofs)
            {
                for (int i = 0; i < num_dofs; ++i)
                {
                    const float s = std::max(0.0f, max_step[i]);
                    if (s <= 0.0f) continue;
                    const float lo = q_current[i] - s;
                    const float hi = q_current[i] + s;
                    q_target[i] = std::clamp(q_target[i], lo, hi);
                }
            }
        }

        if (this->params.Get<bool>("enable_action_ramp", false))
        {
            const float ramp_duration = this->params.Get<float>("action_ramp_duration", 0.0f);
            if (ramp_duration > 1e-6f &&
                static_cast<int>(q_current.size()) == num_dofs)
            {
                const float t = static_cast<float>(this->episode_length_buf) *
                                this->GetPolicyStepTime();
                const float alpha = std::clamp(t / ramp_duration, 0.0f, 1.0f);
                for (int i = 0; i < num_dofs; ++i)
                {
                    q_target[i] = q_current[i] + alpha * (q_target[i] - q_current[i]);
                }
            }
        }
    };

    // Optional sim2sim alignment branch:
    // q_target = offset + scale ⊙ a, where
    // offset = (q_max_soft + q_min_soft)/2, scale = q_max_soft - q_min_soft.
    // This branch is opt-in via yaml and only affects configs that enable it.
    const bool use_soft_joint_pos_action = this->params.Get<bool>("use_soft_joint_pos_action", false);
    if (use_soft_joint_pos_action)
    {
        std::vector<float> action_for_target = actions;
        if (this->params.Get<bool>("soft_limit_apply_action_scale", false))
        {
            action_for_target = actions_scaled;
        }
        else
        {
            if (!disable_action_clip && !clip_lower.empty() && !clip_upper.empty())
            {
                action_for_target = clamp(action_for_target, clip_lower, clip_upper);
            }
        }

        std::vector<float> q_min_soft = this->params.Get<std::vector<float>>("soft_joint_pos_min");
        std::vector<float> q_max_soft = this->params.Get<std::vector<float>>("soft_joint_pos_max");
        if (static_cast<int>(q_min_soft.size()) == num_dofs &&
            static_cast<int>(q_max_soft.size()) == num_dofs &&
            static_cast<int>(action_for_target.size()) == num_dofs)
        {
            output_dof_pos.resize(num_dofs, 0.0f);
            for (int i = 0; i < num_dofs; ++i)
            {
                const float offset = 0.5f * (q_max_soft[i] + q_min_soft[i]);
                const float scale = (q_max_soft[i] - q_min_soft[i]);
                output_dof_pos[i] = offset + scale * action_for_target[i];
            }
            apply_q_target_safety(output_dof_pos);

            std::vector<float> vel_actions_scaled(actions.size(), 0.0f);
            for (int i : this->params.Get<std::vector<int>>("wheel_indices"))
            {
                if (i >= 0 && i < static_cast<int>(vel_actions_scaled.size()) &&
                    i < static_cast<int>(action_for_target.size()))
                {
                    vel_actions_scaled[i] = action_for_target[i];
                }
            }
            output_dof_vel = vel_actions_scaled;
            output_dof_tau = this->params.Get<std::vector<float>>("rl_kp") * (output_dof_pos - q_current)
                           + this->params.Get<std::vector<float>>("rl_kd") * (output_dof_vel - dq_current);
            output_dof_tau = clamp(output_dof_tau,
                                   -this->params.Get<std::vector<float>>("torque_limits"),
                                   this->params.Get<std::vector<float>>("torque_limits"));
            this->output_dof_delta = output_dof_pos - this->params.Get<std::vector<float>>("default_dof_pos");
            return;
        }
        else
        {
            static bool warned_soft_limits_once = false;
            if (!warned_soft_limits_once)
            {
                warned_soft_limits_once = true;
                std::cout << LOGGER::WARNING
                          << "[ComputeOutput] use_soft_joint_pos_action=true but soft_joint_pos_min/max size mismatch. "
                          << "Fallback to legacy action mapping." << std::endl;
            }
        }
    }

    std::vector<float> pos_actions_scaled = actions_processed;
    std::vector<float> vel_actions_scaled(actions.size(), 0.0f);
    for (int i : this->params.Get<std::vector<int>>("wheel_indices"))
    {
        pos_actions_scaled[i] = 0.0f;
        vel_actions_scaled[i] = actions_processed[i];
    }

    // Deadzone is configured per-policy in yaml and loaded by InitRL.
    // Keep default as all-zero (disabled) for full backward compatibility.
    std::vector<float> deadzone_lower = this->params.Get<std::vector<float>>(
        "action_deadzone_lower", std::vector<float>(actions.size(), 0.0f)
    );
    std::vector<float> deadzone_upper = this->params.Get<std::vector<float>>(
        "action_deadzone_upper", std::vector<float>(actions.size(), 0.0f)
    );
    deadzone_lower.resize(actions.size(), 0.0f);
    deadzone_upper.resize(actions.size(), 0.0f);
    for (size_t i = 0; i < pos_actions_scaled.size(); ++i)
    {
        float& x = pos_actions_scaled[i];
        if (x > deadzone_lower[i] && x < deadzone_upper[i])
        {
            x = 0.0f;
        }
    }

    this->output_dof_delta = pos_actions_scaled;

    const std::string action_type = this->params.Get<std::string>("action_type", "default_position");
    if (action_type == "relative_position")
    {
        // Match IsaacLab RelativeJointPositionAction: q_target = q_current + delta_q.
        output_dof_pos = q_current + pos_actions_scaled;
    }
    else
    {
        output_dof_pos = pos_actions_scaled + this->params.Get<std::vector<float>>("default_dof_pos");
    }
    apply_q_target_safety(output_dof_pos);
    // Position mode uses zero velocity target; wheel indices carry their velocity targets above.
    output_dof_vel = vel_actions_scaled;
    if (action_type == "relative_position")
    {
        // Relative-position torque form requested:
        //   q_delta = clip(action * scale)
        //   tau = kp * q_delta + kd * (0 - v_current)
        output_dof_tau = this->params.Get<std::vector<float>>("rl_kp") * pos_actions_scaled
                       + this->params.Get<std::vector<float>>("rl_kd") * (output_dof_vel - dq_current);
    }
    else
    {
        output_dof_tau = this->params.Get<std::vector<float>>("rl_kp") * (output_dof_pos - q_current)
                       + this->params.Get<std::vector<float>>("rl_kd") * (output_dof_vel - dq_current);
    }
    output_dof_tau = clamp(output_dof_tau, -this->params.Get<std::vector<float>>("torque_limits"), this->params.Get<std::vector<float>>("torque_limits"));
}

int RL::InverseJointMapping(int idx) const
{
    auto joint_mapping = this->params.Get<std::vector<int>>("joint_mapping");
    for (size_t i = 0; i < joint_mapping.size(); ++i) {
        if (joint_mapping[i] == idx) return (int)i;
    }
    return -1;
}

void RL::TorqueProtect(const std::vector<float>& origin_output_dof_tau)
{
    std::vector<int> out_of_range_indices;
    std::vector<float> out_of_range_values;
    for (size_t i = 0; i < origin_output_dof_tau.size(); ++i)
    {
        float torque_value = origin_output_dof_tau[i];
        float limit_lower = -this->params.Get<std::vector<float>>("torque_limits")[i];
        float limit_upper = this->params.Get<std::vector<float>>("torque_limits")[i];

        if (torque_value < limit_lower || torque_value > limit_upper)
        {
            out_of_range_indices.push_back(i);
            out_of_range_values.push_back(torque_value);
        }
    }
    if (!out_of_range_indices.empty())
    {
        for (size_t i = 0; i < out_of_range_indices.size(); ++i)
        {
            int index = out_of_range_indices[i];
            float value = out_of_range_values[i];
            float limit_lower = -this->params.Get<std::vector<float>>("torque_limits")[index];
            float limit_upper = this->params.Get<std::vector<float>>("torque_limits")[index];

            std::cout << LOGGER::WARNING << "Torque(" << index + 1 << ")=" << value << " out of range(" << limit_lower << ", " << limit_upper << ")" << std::endl;
        }
        // Just a reminder, no protection
        // this->control.SetKeyboard(Input::Keyboard::P);
        std::cout << LOGGER::INFO << "Switching to STATE_POS_GETDOWN"<< std::endl;
    }
}

void RL::AttitudeProtect(const std::vector<float> &quaternion, float pitch_threshold, float roll_threshold)
{
    // Use QuaternionToEuler from vector_math.hpp
    std::vector<float> euler = QuaternionToEuler(quaternion);
    float roll = euler[0] * 57.2958f;   // Convert to degrees
    float pitch = euler[1] * 57.2958f;

    if (std::fabs(roll) > roll_threshold)
    {
        this->control.SetKeyboard(Input::Keyboard::P);
        std::cout << LOGGER::WARNING << "Roll exceeds " << roll_threshold << " degrees. Current: " << roll << " degrees." << std::endl;
    }
    if (std::fabs(pitch) > pitch_threshold)
    {
        this->control.SetKeyboard(Input::Keyboard::P);
        std::cout << LOGGER::WARNING << "Pitch exceeds " << pitch_threshold << " degrees. Current: " << pitch << " degrees." << std::endl;
    }
}

#include <termios.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>

static int kbhit()
{
    static bool initialized = false;
    static termios original_term;

    // Initialize terminal to non-canonical mode on first call
    if (!initialized)
    {
        tcgetattr(STDIN_FILENO, &original_term);

        termios new_term = original_term;
        new_term.c_lflag &= ~(ICANON | ECHO);  // Disable canonical mode and echo
        new_term.c_cc[VMIN] = 0;   // Non-blocking read
        new_term.c_cc[VTIME] = 0;  // No timeout

        tcsetattr(STDIN_FILENO, TCSANOW, &new_term);

        // Register cleanup function to restore terminal on exit
        static bool cleanup_registered = false;
        if (!cleanup_registered)
        {
            std::atexit([]() {
                tcsetattr(STDIN_FILENO, TCSANOW, &original_term);
            });
            cleanup_registered = true;
        }

        initialized = true;
    }

    // Non-blocking read of a single character
    char c;
    int result = read(STDIN_FILENO, &c, 1);

    return (result == 1) ? (unsigned char)c : -1;
}

void RL::KeyboardInterface()
{
    int c = kbhit();
    if (c > 0)
    {
        switch (c)
        {
        case '0': this->control.SetKeyboard(Input::Keyboard::Num0); break;
        case '1': this->control.SetKeyboard(Input::Keyboard::Num1); break;
        case '2': this->control.SetKeyboard(Input::Keyboard::Num2); break;
        case '3': this->control.SetKeyboard(Input::Keyboard::Num3); break;
        case '4': this->control.SetKeyboard(Input::Keyboard::Num4); break;
        case '5': this->control.SetKeyboard(Input::Keyboard::Num5); break;
        case '6': this->control.SetKeyboard(Input::Keyboard::Num6); break;
        case '7': this->control.SetKeyboard(Input::Keyboard::Num7); break;
        case '8': this->control.SetKeyboard(Input::Keyboard::Num8); break;
        case '9': this->control.SetKeyboard(Input::Keyboard::Num9); break;
        case 'a': case 'A': this->control.SetKeyboard(Input::Keyboard::A); break;
        case 'b': case 'B': this->control.SetKeyboard(Input::Keyboard::B); break;
        case 'c': case 'C': this->control.SetKeyboard(Input::Keyboard::C); break;
        case 'd': case 'D': this->control.SetKeyboard(Input::Keyboard::D); break;
        case 'e': case 'E': this->control.SetKeyboard(Input::Keyboard::E); break;
        case 'f': case 'F': this->control.SetKeyboard(Input::Keyboard::F); break;
        case 'g': case 'G': this->control.SetKeyboard(Input::Keyboard::G); break;
        case 'h': case 'H': this->control.SetKeyboard(Input::Keyboard::H); break;
        case 'i': case 'I': this->control.SetKeyboard(Input::Keyboard::I); break;
        case 'j': case 'J': this->control.SetKeyboard(Input::Keyboard::J); break;
        case 'k': case 'K': this->control.SetKeyboard(Input::Keyboard::K); break;
        case 'l': case 'L': this->control.SetKeyboard(Input::Keyboard::L); break;
        case 'm': case 'M': this->control.SetKeyboard(Input::Keyboard::M); break;
        case 'n': case 'N': this->control.SetKeyboard(Input::Keyboard::N); break;
        case 'o': case 'O': this->control.SetKeyboard(Input::Keyboard::O); break;
        case 'p': case 'P': this->control.SetKeyboard(Input::Keyboard::P); break;
        case 'q': case 'Q': this->control.SetKeyboard(Input::Keyboard::Q); break;
        case 'r': case 'R': this->control.SetKeyboard(Input::Keyboard::R); break;
        case 's': case 'S': this->control.SetKeyboard(Input::Keyboard::S); break;
        case 't': case 'T': this->control.SetKeyboard(Input::Keyboard::T); break;
        case 'u': case 'U': this->control.SetKeyboard(Input::Keyboard::U); break;
        case 'v': case 'V': this->control.SetKeyboard(Input::Keyboard::V); break;
        case 'w': case 'W': this->control.SetKeyboard(Input::Keyboard::W); break;
        case 'x': case 'X': this->control.SetKeyboard(Input::Keyboard::X); break;
        case 'y': case 'Y': this->control.SetKeyboard(Input::Keyboard::Y); break;
        case 'z': case 'Z': this->control.SetKeyboard(Input::Keyboard::Z); break;
        case ' ': this->control.SetKeyboard(Input::Keyboard::Space); break;
        case '\n': case '\r': this->control.SetKeyboard(Input::Keyboard::Enter); break;
        case 27:  // Escape sequence (for arrow keys on Unix/Linux/macOS)
        {
            char seq[2];
            // Try to read escape sequence non-blockingly
            if (read(STDIN_FILENO, &seq[0], 1) == 1)
            {
                if (seq[0] == '[')
                {
                    if (read(STDIN_FILENO, &seq[1], 1) == 1)
                    {
                        switch (seq[1])
                        {
                        case 'A': this->control.SetKeyboard(Input::Keyboard::Up); break;
                        case 'B': this->control.SetKeyboard(Input::Keyboard::Down); break;
                        case 'C': this->control.SetKeyboard(Input::Keyboard::Right); break;
                        case 'D': this->control.SetKeyboard(Input::Keyboard::Left); break;
                        default: break;
                        }
                    }
                }
                else
                {
                    // Plain escape key
                    this->control.SetKeyboard(Input::Keyboard::Escape);
                }
            }
            else
            {
                // Plain escape key
                this->control.SetKeyboard(Input::Keyboard::Escape);
            }
        } break;
        default:  break;
        }
    }
}

template <typename T>
std::vector<T> ReadVectorFromYaml(const YAML::Node &node)
{
    std::vector<T> values;
    for (const auto &val : node)
    {
        values.push_back(val.as<T>());
    }
    return values;
}

void RL::ReadYaml(const std::string& file_path, const std::string& file_name)
{
    std::string config_path = std::string(POLICY_DIR) + "/" + file_path + "/" + file_name;
    YAML::Node config;
    try
    {
        config = YAML::LoadFile(config_path)[file_path];
    }
    catch (YAML::BadFile &e)
    {
        std::cout << LOGGER::ERROR << "The file '" << config_path << "' does not exist" << std::endl;
        return;
    }

    for (auto it = config.begin(); it != config.end(); ++it)
    {
        std::string key = it->first.as<std::string>();
        this->params.config_node[key] = it->second;
    }
}

void RL::CSVInit(std::string robot_path)
{
    csv_filename = std::string(POLICY_DIR) + "/" + robot_path + "/motor";

    // Uncomment these lines if need timestamp for file name
    // auto now = std::chrono::system_clock::now();
    // std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    // std::stringstream ss;
    // ss << std::put_time(std::localtime(&now_c), "%Y%m%d%H%M%S");
    // std::string timestamp = ss.str();
    // csv_filename += "_" + timestamp;

    csv_filename += ".csv";
    std::ofstream file(csv_filename.c_str());

    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "torque_" << i << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "policy_delta_" << i << ","; }
    // for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "tau_est_" << i << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "joint_pos_" << i << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "joint_pos_target_" << i << ","; }
    // for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "joint_vel_" << i << ","; }

    file << std::endl;

    file.close();
}

void RL::CSVLogger(const std::vector<float>& torque, const std::vector<float>& tau_est, const std::vector<float>& joint_pos, const std::vector<float>& joint_pos_target, const std::vector<float>& joint_vel)
{
    std::ofstream file(csv_filename.c_str(), std::ios_base::app);

    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << torque[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << this->output_dof_delta[i] << ","; }
    // for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << tau_est[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << joint_pos[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << joint_pos_target[i] << ","; }
    // for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << joint_vel[i] << ","; }

    file << std::endl;

    file.close();
}

// ============================================================================
// Observation CSV logger (for sim-to-real gap analysis)
// ----------------------------------------------------------------------------
// Records the full policy pipeline per control step:
//   [t, cmd(3), base_ang_vel(3), base_quat(4), dof_pos(n), dof_pos_target(n),
//    dof_vel(n), action(n)]
// All tensors are written in the IsaacSim / sim joint order (same as obs fed
// to the network), so the csv can be replayed straight into an offline
// inference script.
// Safety: any field shorter than expected is zero-padded; empty vectors are
// logged as 0.0 rather than crashing.
// ============================================================================
static inline void CsvWriteFixed(std::ofstream& f, const std::vector<float>& v, int n)
{
    const int have = static_cast<int>(v.size());
    f << std::fixed << std::setprecision(4);
    for (int i = 0; i < n; ++i)
    {
        f << (i < have ? v[i] : 0.0f) << ",";
    }
}

void RL::CSVInitObs(std::string robot_path)
{
    csv_filename = std::string(POLICY_DIR) + "/" + robot_path + "/obs.csv";
    std::ofstream file(csv_filename.c_str());

    const int n = this->params.Get<int>("num_of_dofs");

    file << "t,";
    file << "cmd_x,cmd_y,cmd_yaw,";
    file << "base_ang_vel_x,base_ang_vel_y,base_ang_vel_z,";
    file << "base_quat_w,base_quat_x,base_quat_y,base_quat_z,";
    for (int i = 0; i < n; ++i) { file << "dof_pos_" << i << ","; }
    for (int i = 0; i < n; ++i) { file << "dof_pos_target_" << i << ","; }
    for (int i = 0; i < n; ++i) { file << "dof_vel_" << i << ","; }
    for (int i = 0; i < n; ++i) { file << "action_" << i << ","; }

    file << std::endl;
    file.close();
}

void RL::CSVLoggerObs(
    int t,
    const std::vector<float>& commands,
    const std::vector<float>& base_ang_vel,
    const std::vector<float>& base_quat,
    const std::vector<float>& dof_pos,
    const std::vector<float>& dof_pos_target,
    const std::vector<float>& dof_vel,
    const std::vector<float>& actions)
{
    std::ofstream file(csv_filename.c_str(), std::ios_base::app);

    const int n = this->params.Get<int>("num_of_dofs");

    file << t << ",";
    CsvWriteFixed(file, commands, 3);       // cmd_x, cmd_y, cmd_yaw
    CsvWriteFixed(file, base_ang_vel, 3);   // body-frame or world-frame omega
    CsvWriteFixed(file, base_quat, 4);      // (w, x, y, z)
    CsvWriteFixed(file, dof_pos, n);        // actual joint pos
    CsvWriteFixed(file, dof_pos_target, n); // output sent to low-level PD
    CsvWriteFixed(file, dof_vel, n);        // actual joint vel
    CsvWriteFixed(file, actions, n);        // raw policy output (pre-scale)

    file << std::endl;
    file.close();
}

// ============================================================================
// Full-trajectory trace logger
// ----------------------------------------------------------------------------
// Writes <policy_dir>/<robot>/trace.csv. Runs from construction to destruction
// of the RL_Real instance, independent of whether the RL policy is active.
// Schema (one row per tick):
//   t_sec, fsm_state, cmd(3), base_ang_vel(3), base_quat(4), projected_gravity(3), dof_pos(n),
//   dof_pos_target(n), dof_vel(n), action(n),
//   root_pos_relative_z(1), cee_root_rot_6d(6), key_body_pos_relative(13*3), progress(1)
// Where `dof_pos_target` is the q_cmd actually being written to low-level PD
// (GetUp interpolator output, RL policy output, or hold pose in Passive).
// ============================================================================
void RL::CSVInitTrace(std::string robot_path, std::string filename)
{
    csv_filename = std::string(POLICY_DIR) + "/" + robot_path + "/" + filename;
    std::ofstream file(csv_filename.c_str());

    const int n = this->params.Get<int>("num_of_dofs");

    file << "t_sec,fsm_state,";
    file << "cmd_x,cmd_y,cmd_yaw,";
    file << "base_ang_vel_x,base_ang_vel_y,base_ang_vel_z,";
    file << "base_quat_w,base_quat_x,base_quat_y,base_quat_z,";
    file << "projected_gravity_x,projected_gravity_y,projected_gravity_z,";
    for (int i = 0; i < n; ++i) { file << "dof_pos_" << i << ","; }
    for (int i = 0; i < n; ++i) { file << "dof_pos_target_" << i << ","; }
    for (int i = 0; i < n; ++i) { file << "dof_vel_" << i << ","; }
    for (int i = 0; i < n; ++i) { file << "policy_delta_" << i << ","; }
    for (int i = 0; i < n; ++i) { file << "action_" << i << ","; }
    // Dance-specific observation terms (written for all states; non-dance defaults to zeros).
    file << "root_pos_relative_z,";
    file << "cee_tangent_x,cee_tangent_y,cee_tangent_z,";
    file << "cee_normal_x,cee_normal_y,cee_normal_z,";
    for (int i = 0; i < 13 * 3; ++i) { file << "key_body_pos_relative_" << i << ","; }
    file << "progress,";

    file << std::endl;
    file.close();
}

void RL::CSVLoggerTrace(
    float t_sec,
    const std::string& fsm_state,
    const std::vector<float>& commands,
    const std::vector<float>& base_ang_vel,
    const std::vector<float>& base_quat,
    const std::vector<float>& projected_gravity,
    const std::vector<float>& dof_pos,
    const std::vector<float>& dof_pos_target,
    const std::vector<float>& dof_vel,
    const std::vector<float>& policy_delta,
    const std::vector<float>& actions)
{
    std::ofstream file(csv_filename.c_str(), std::ios_base::app);
    file << std::fixed << std::setprecision(4);  
    const int n = this->params.Get<int>("num_of_dofs");

    file << t_sec << "," << fsm_state << ",";
    CsvWriteFixed(file, commands, 3);
    CsvWriteFixed(file, base_ang_vel, 3);
    CsvWriteFixed(file, base_quat, 4);
    CsvWriteFixed(file, projected_gravity, 3);
    CsvWriteFixed(file, dof_pos, n);
    CsvWriteFixed(file, dof_pos_target, n);
    CsvWriteFixed(file, dof_vel, n);
    CsvWriteFixed(file, policy_delta, n);
    CsvWriteFixed(file, actions, n);

    // ------------------------------------------------------------------------
    // Dance observation extras for offline alignment/debug:
    //   root_pos_relative / cee(root_rot_6d) / key_body_pos_relative / progress
    // Use the exact same math/layout as ComputeObservation() to avoid train-deploy drift.
    // ------------------------------------------------------------------------
    float root_z = 0.0f;
    if (this->obs.base_pos.size() >= 3)
    {
        root_z = this->obs.base_pos[2];
    }
    file << root_z << ",";

    std::vector<float> root_rot_6d(6, 0.0f);
    if (this->obs.base_quat.size() == 4)
    {
        const std::vector<float> tangent = QuatApply(this->obs.base_quat, {1.0f, 0.0f, 0.0f});
        const std::vector<float> normal = QuatApply(this->obs.base_quat, {0.0f, 0.0f, 1.0f});
        root_rot_6d = {
            tangent[0], tangent[1], tangent[2],
            normal[0], normal[1], normal[2]
        };
        if (this->params.Get<bool>("root_rot_6d_swap_xy", false))
        {
            std::swap(root_rot_6d[0], root_rot_6d[1]);
            std::swap(root_rot_6d[3], root_rot_6d[4]);
        }
    }
    CsvWriteFixed(file, root_rot_6d, 6);

    const std::vector<int> default_key_body_joint_indices = {9, 10, 13, 14, 17, 18, 20, 19, 5, 7, 6, 12, 11};
    const auto key_body_joint_indices = this->params.Get<std::vector<int>>(
        "key_body_joint_indices", default_key_body_joint_indices
    );
    std::vector<float> key_body_pos_relative;
    key_body_pos_relative.reserve(key_body_joint_indices.size() * 3);
    const bool has_key_body_cache =
        !this->obs.key_body_pos_rel.empty() &&
        this->obs.key_body_pos_rel.size() >= static_cast<size_t>(this->params.Get<int>("num_of_dofs")) * 3;
    if (has_key_body_cache)
    {
        for (int joint_idx : key_body_joint_indices)
        {
            const int base = joint_idx * 3;
            if (joint_idx >= 0 && base + 2 < static_cast<int>(this->obs.key_body_pos_rel.size()))
            {
                key_body_pos_relative.push_back(this->obs.key_body_pos_rel[base + 0]);
                key_body_pos_relative.push_back(this->obs.key_body_pos_rel[base + 1]);
                key_body_pos_relative.push_back(this->obs.key_body_pos_rel[base + 2]);
            }
            else
            {
                key_body_pos_relative.insert(key_body_pos_relative.end(), {0.0f, 0.0f, 0.0f});
            }
        }
    }
    else
    {
        key_body_pos_relative.assign(key_body_joint_indices.size() * 3, 0.0f);
    }
    CsvWriteFixed(file, key_body_pos_relative, 13 * 3);

    float progress = 0.0f;
    if (this->config_name == "whole_body_tracking")
    {
        const int max_episode_length = this->params.Get<int>("max_episode_length", -1);
        if (max_episode_length > 1 && this->episode_length_buf >= 1)
        {
            progress = static_cast<float>(this->episode_length_buf - 1) /
                       static_cast<float>(max_episode_length - 1);
        }
        progress = std::clamp(progress, 0.0f, 1.0f);
    }
    file << progress << ",";

    file << std::endl;
    file.close();
}

void RL::CSVInitPolicyObs(const std::string& robot_path, int obs_dim, const std::string& filename)
{
    this->policy_obs_csv_filename = std::string(POLICY_DIR) + "/" + robot_path + "/" + filename;
    std::ofstream file(this->policy_obs_csv_filename.c_str());
    file << "episode_step,progress";
    for (int i = 0; i < obs_dim; ++i)
    {
        file << ",obs_" << i;
    }
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        file << ",action_" << i;
    }
    file << std::endl;
    file.close();
}

void RL::CSVLoggerPolicyObs(int episode_step, float progress,
                            const std::vector<float>& policy_obs,
                            const std::vector<float>& actions)
{
    if (this->policy_obs_csv_filename.empty())
    {
        return;
    }

    std::ofstream file(this->policy_obs_csv_filename.c_str(), std::ios_base::app);
    file << std::fixed << std::setprecision(6);
    file << episode_step << "," << progress << ",";
    CsvWriteFixed(file, policy_obs, static_cast<int>(policy_obs.size()));
    file << std::fixed << std::setprecision(4);
    const int n_actions = static_cast<int>(actions.size());
    for (int i = 0; i < n_actions; ++i)
    {
        file << (i < n_actions ? actions[i] : 0.0f);
        if (i + 1 < n_actions)
        {
            file << ",";
        }
    }
    file << std::endl;
    file.close();
}

bool RLFSMState::Interpolate(
    float& percent,
    const std::vector<float>& start_pos,
    const std::vector<float>& target_pos,
    float duration_seconds,
    const std::string& description,
    bool use_fixed_gains)
{
    if (percent >= 1.0f)
    {
        return false;
    }

    if (percent == 0.0f)
    {
        float max_diff = 0.0f;
        for (size_t i = 0; i < start_pos.size() && i < target_pos.size(); ++i)
        {
            max_diff = std::max(max_diff, std::abs(start_pos[i] - target_pos[i]));
        }

        if (max_diff < 0.1f)
        {
            percent = 1.0f;
        }
    }

    int required_frames = std::max(1, static_cast<int>(std::ceil(duration_seconds / rl.params.Get<float>("dt"))));
    float step = 1.0f / required_frames;

    percent += step;
    percent = std::min(percent, 1.0f);

    auto kp = use_fixed_gains ? rl.params.Get<std::vector<float>>("fixed_kp") : rl.params.Get<std::vector<float>>("rl_kp");
    auto kd = use_fixed_gains ? rl.params.Get<std::vector<float>>("fixed_kd") : rl.params.Get<std::vector<float>>("rl_kd");

    for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
    {
        fsm_command->motor_command.q[i] = (1 - percent) * start_pos[i] + percent * target_pos[i];
        fsm_command->motor_command.dq[i] = 0;
        fsm_command->motor_command.kp[i] = kp[i];
        fsm_command->motor_command.kd[i] = kd[i];
        fsm_command->motor_command.tau[i] = 0;
    }

    if (!description.empty())
    {
        LOGGER::PrintProgress(percent, description);
    }

    if (percent >= 1.0f)
    {
        return false;
    }

    return true;
}

void RLFSMState::RLControl()
{
    std::vector<float> _output_dof_pos, _output_dof_vel;
    const bool has_new_pos = rl.output_dof_pos_queue.try_pop(_output_dof_pos);
    const bool has_new_vel = rl.output_dof_vel_queue.try_pop(_output_dof_vel);
    bool has_new_output = has_new_pos || has_new_vel;

    // Avoid dropping a valid packet when pos/vel queues are briefly out-of-sync.
    // Reuse the previous counterpart from blend targets when only one side is available.
    if (has_new_pos && !has_new_vel)
    {
        _output_dof_vel = rl.rl_blend_dq_target;
    }
    else if (!has_new_pos && has_new_vel)
    {
        _output_dof_pos = rl.rl_blend_q_target;
    }

    if (has_new_output)
    {
        if (!_output_dof_pos.empty())
        {
            rl.rl_blend_q_target = _output_dof_pos;
        }
        if (!_output_dof_vel.empty())
        {
            rl.rl_blend_dq_target = _output_dof_vel;
        }
    }

    if (rl.rl_wait_first_frame && !has_new_output)
    {
        const auto hold_kp = rl.params.Get<std::vector<float>>("fixed_kp");
        const auto hold_kd = rl.params.Get<std::vector<float>>("fixed_kd");
        const bool hold_size_valid = rl.hold_q_on_enter.size() == static_cast<size_t>(rl.params.Get<int>("num_of_dofs"));

        for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
        {
            fsm_command->motor_command.q[i] = hold_size_valid ? rl.hold_q_on_enter[i] : fsm_state->motor_state.q[i];
            fsm_command->motor_command.dq[i] = 0;
            fsm_command->motor_command.kp[i] = hold_kp[i];
            fsm_command->motor_command.kd[i] = hold_kd[i];
            fsm_command->motor_command.tau[i] = 0;
        }
        return;
    }

    if (rl.rl_wait_first_frame && has_new_output)
    {
        rl.rl_wait_first_frame = false;
        if (rl.rl_blend_duration > 0.0f)
        {
            rl.rl_blend_active = true;
            rl.rl_blend_time = 0.0f;
        }
    }

    if (rl.rl_blend_active)
    {
        const int num_dofs = rl.params.Get<int>("num_of_dofs");
        const auto fixed_kp = rl.params.Get<std::vector<float>>("fixed_kp");
        const auto fixed_kd = rl.params.Get<std::vector<float>>("fixed_kd");
        const auto rl_kp = rl.params.Get<std::vector<float>>("rl_kp");
        const auto rl_kd = rl.params.Get<std::vector<float>>("rl_kd");
        const bool hold_size_valid = rl.hold_q_on_enter.size() == static_cast<size_t>(num_dofs);
        const bool blend_size_valid =
            rl.rl_blend_q_target.size() == static_cast<size_t>(num_dofs) &&
            rl.rl_blend_dq_target.size() == static_cast<size_t>(num_dofs);

        rl.rl_blend_time += rl.params.Get<float>("dt");
        float alpha = std::min(1.0f, rl.rl_blend_time / std::max(1e-6f, rl.rl_blend_duration));
        float smooth_alpha = 3.0f * alpha * alpha - 2.0f * alpha * alpha * alpha;

        for (int i = 0; i < num_dofs; ++i)
        {
            float q_hold = hold_size_valid ? rl.hold_q_on_enter[i] : fsm_state->motor_state.q[i];
            float q_target = q_hold;
            if (blend_size_valid)
            {
                q_target = rl.rl_blend_q_target[i];
            }
            float dq_target = blend_size_valid ? rl.rl_blend_dq_target[i] : 0.0f;

            fsm_command->motor_command.q[i] = (1.0f - smooth_alpha) * q_hold + smooth_alpha * q_target;
            fsm_command->motor_command.dq[i] = smooth_alpha * dq_target;
            fsm_command->motor_command.kp[i] = (1.0f - smooth_alpha) * fixed_kp[i] + smooth_alpha * rl_kp[i];
            fsm_command->motor_command.kd[i] = (1.0f - smooth_alpha) * fixed_kd[i] + smooth_alpha * rl_kd[i];
            fsm_command->motor_command.tau[i] = 0;
        }

        if (alpha >= 1.0f)
        {
            rl.rl_blend_active = false;
        }
        return;
    }

    if (has_new_output)
    {
        for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
        {
            if (!_output_dof_pos.empty())
            {
                fsm_command->motor_command.q[i] = _output_dof_pos[i];
            }
            if (!_output_dof_vel.empty())
            {
                fsm_command->motor_command.dq[i] = _output_dof_vel[i];
            }
            fsm_command->motor_command.kp[i] = rl.params.Get<std::vector<float>>("rl_kp")[i];
            fsm_command->motor_command.kd[i] = rl.params.Get<std::vector<float>>("rl_kd")[i];
            fsm_command->motor_command.tau[i] = 0;
        }
    }
}
