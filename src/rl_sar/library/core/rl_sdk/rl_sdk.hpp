/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RL_SDK_HPP
#define RL_SDK_HPP

#include <iostream>
#include <string>
#include <exception>
#include <unistd.h>
#include <algorithm>
#include <tbb/concurrent_queue.h>
#include <vector>
#include <deque>
#include <memory>
#include <fstream>
#include <mutex>
#include <random>

#include <yaml-cpp/yaml.h>
#include "fsm.hpp"
#include "observation_buffer.hpp"
#include "vector_math.hpp"
#include "inference_runtime.hpp"
#include "logger.hpp"
#include "motion_loader.hpp"

// Forward declare 让 RL 基类可以持有 unique_ptr<KinematicsFK>，
// 但不污染 rl_sdk.hpp 的所有用户（kinematics_fk.hpp 是 mujoco 相关，单独条件编译）。
class KinematicsFK;

/**
 * @brief 就地均匀噪声注入（Sim-to-Real 域随机化工具）
 *        数学模型：o~ = o + ε，ε ~ U(-scale, +scale)
 *        - scale <= 0 即视为禁用（实物端 yaml 默认 0 即可关闭）
 *        - thread_local RNG，保证多线程调用安全且互不干扰
 *        - 物理量纲：调用方保证 scale 与 obs 同量纲（rad/s、rad、无量纲）
 */
inline void AddUniformNoiseInPlace(std::vector<float>& v, float scale)
{
    if (scale <= 0.0f) return;
    thread_local std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto& x : v) x += dist(gen);
}

template <typename T>
struct RobotCommand
{
    struct MotorCommand
    {
        std::vector<int> mode;
        std::vector<T> q;
        std::vector<T> dq;
        std::vector<T> tau;
        std::vector<T> kp;
        std::vector<T> kd;

        void resize(size_t num_joints)
        {
            mode.resize(num_joints, 0);
            q.resize(num_joints, 0.0f);
            dq.resize(num_joints, 0.0f);
            tau.resize(num_joints, 0.0f);
            kp.resize(num_joints, 0.0f);
            kd.resize(num_joints, 0.0f);
        }
    } motor_command;
};

template <typename T>
struct RobotState
{
    struct IMU
    {
        std::vector<T> quaternion = {1.0f, 0.0f, 0.0f, 0.0f}; // w, x, y, z
        std::vector<T> gyroscope = {0.0f, 0.0f, 0.0f};
        std::vector<T> accelerometer = {0.0f, 0.0f, 0.0f};
    } imu;

    // Base position in world frame [x, y, z].
    // Real-robot backends may leave this as zeros when unavailable.
    std::vector<T> base_pos = {0.0f, 0.0f, 0.0f};

    struct MotorState
    {
        std::vector<T> q;
        std::vector<T> dq;
        std::vector<T> ddq;
        std::vector<T> tau_est;
        std::vector<T> cur;

        void resize(size_t num_joints)
        {
            q.resize(num_joints, 0.0f);
            dq.resize(num_joints, 0.0f);
            ddq.resize(num_joints, 0.0f);
            tau_est.resize(num_joints, 0.0f);
            cur.resize(num_joints, 0.0f);
        }
    } motor_state;
};

namespace Input
{
    // Recommend: Num0-GetUp Num9-GetDown N-ToggleNavMode
    //            R-SimReset Enter-SimToggle
    //            M-MotorEnable K-MotorDisable P-MotorPassive
    //            Num1-BaseLocomotion Num2-Num8-Skills(7)
    //            WS-AxisX AD-AxisY QE-AxisYaw Space-AxisClear
    enum class Keyboard
    {
        None = 0,
        A, B, C, D, E, F, G, H, I, J, K, L, M,
        N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
        Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
        Space, Enter, Escape,
        Up, Down, Left, Right
    };

    // Recommend: A-GetUp B-GetDown X-ToggleNavMode Y-None
    //            RB_Y-SimReset RB_X-SimToggle
    //            LB_A-MotorEnable LB_B-MotorDisable LB_X-MotorPassive
    //            RB_DPadUp-BaseLocomotion RB_DPadOthers/LB_DPadOthers-Skills(7)
    //            LY-AxisX LX-AxisY RX-AxisYaw
    enum class Gamepad
    {
        None = 0,
        A, B, X, Y, LB, RB, LStick, RStick, DPadUp, DPadDown, DPadLeft, DPadRight,
        LB_A, LB_B, LB_X, LB_Y, LB_LStick, LB_RStick, LB_DPadUp, LB_DPadDown, LB_DPadLeft, LB_DPadRight,
        RB_A, RB_B, RB_X, RB_Y, RB_LStick, RB_RStick, RB_DPadUp, RB_DPadDown, RB_DPadLeft, RB_DPadRight,
        LB_RB
    };
}

struct Control
{
    Input::Keyboard current_keyboard = Input::Keyboard::None, last_keyboard = Input::Keyboard::None;
    Input::Gamepad current_gamepad = Input::Gamepad::None, last_gamepad = Input::Gamepad::None;

    float x = 0.0f;
    float y = 0.0f;
    float yaw = 0.0f;
    bool navigation_mode = false;

    void SetKeyboard(Input::Keyboard keyboad)
    {
        if (current_keyboard != keyboad)
        {
            last_keyboard = current_keyboard;
            current_keyboard = keyboad;
        }
    }

    void SetGamepad(Input::Gamepad gamepad)
    {
        if (current_gamepad != gamepad)
        {
            last_gamepad = current_gamepad;
            current_gamepad = gamepad;
        }
    }

    void ClearInput()
    {
        current_keyboard = last_keyboard;
        current_gamepad = Input::Gamepad::None;
    }
};

struct YamlParams
{
    YAML::Node config_node;

    // Get config value by key
    // WARNING: For vectors/containers, store result in a variable before using iterators/references:
    //   ✓ auto vec = params.Get<std::vector<int>>("key"); vec.begin()
    //   ✗ params.Get<std::vector<int>>("key").begin()  // dangling reference!
    template<typename T>
    T Get(const std::string& key, const T& default_value = T()) const
    {
        if (config_node[key])
        {
            return config_node[key].as<T>();
        }
        return default_value;
    }

    bool Has(const std::string& key) const
    {
        return config_node[key].IsDefined();
    }
};

template <typename T>
struct Observations
{
    std::vector<T> lin_vel;
    std::vector<T> ang_vel;
    std::vector<T> gravity_vec;
    std::vector<T> commands;
    std::vector<T> base_quat;
    // Base position in world frame [x, y, z].
    std::vector<T> base_pos;
    std::vector<T> dof_pos;
    std::vector<T> dof_vel;
    // Flattened key-body position relative to root: [joint0_xyz, joint1_xyz, ...] in training joint order.
    std::vector<T> key_body_pos_rel;
    std::vector<T> actions;
    // Sim-to-Real: 动作历史 FIFO（内部 newest-first，ComputeObservation 反向展开给 policy）
    // index 0 = a_{t-1}（最近一次 raw policy action），index K-1 = a_{t-K}（最旧）
    // 让策略显式看到自己最近 K 步的指令序列，从而学到对自身延迟动力学的内部模型
    std::deque<std::vector<T>> actions_history;
};

class RL
{
public:
    RL();
    // out-of-line dtor: KinematicsFK 在本头文件里是 forward-decl，
    // unique_ptr<KinematicsFK> 的析构需要看见完整类型，因此实现放到 rl_sdk.cpp。
    virtual ~RL();

    YamlParams params;
    Observations<float> obs;
    std::vector<int> obs_dims;
    std::vector<float> last_policy_obs;

    RobotState<float> robot_state;
    RobotCommand<float> robot_command;
    tbb::concurrent_queue<std::vector<float>> output_dof_pos_queue;
    tbb::concurrent_queue<std::vector<float>> output_dof_vel_queue;
    tbb::concurrent_queue<std::vector<float>> output_dof_tau_queue;

    FSM fsm;
    RobotState<float> start_state;
    RobotState<float> now_state;
    bool rl_init_done = false;
    bool rl_wait_first_frame = false;
    std::vector<float> hold_q_on_enter;
    bool rl_blend_active = false;
    float rl_blend_time = 0.0f;
    float rl_blend_duration = 0.2f;
    std::vector<float> rl_blend_q_target;
    std::vector<float> rl_blend_dq_target;
    // init
    void InitObservations();
    void InitOutputs();
    void InitControl();
    void InitRL(std::string robot_config_path);
    void InitJointNum(size_t num_joints);
    // Policy control step (seconds). Uses policy_step_time from yaml when set,
    // otherwise dt * decimation from base config.
    float GetPolicyStepTime() const;
    virtual void OnPolicyConfigLoaded() {}
    // Sim2sim: teleport sim state to ampobs init row (MuJoCo override).
    virtual bool ApplyWholeBodyTrackingInitPoseFromAmpObs() { return false; }

    // ----------------------------------------------------------------------
    // Online forward kinematics (rl_real_myrobot 用到):
    //   实机部署时 obs.key_body_pos_rel 必须由真实关节角 + base 朝向算出来，
    //   否则跟训练分布脱钩。这里把功能集中在 rl_sdk，所有 backend 都可调用。
    //   USE_MUJOCO=OFF 时 LoadFkModel 直接返回 false，调用方应回退到静态参考。
    // ----------------------------------------------------------------------
    /**
     * 加载 MJCF + 建立 policy_idx -> mujoco body 的缓存。dance 切换前调一次即可。
     * @return true 成功，可后续调用 ComputeKeyBodyPosFK。
     */
    bool LoadFkModel(const std::string& mjcf_path,
                     const std::vector<int>& joint_mapping,
                     const std::string& root_body_name = "base_link");

    /**
     * 用当前 obs.dof_pos + obs.base_quat 跑一次 mj_kinematics，
     * 写入 obs.key_body_pos_rel（layout 与 rl_sim_mujoco.cpp 一致）。
     * @return true 成功，false 表示 FK 未加载或维度不匹配（调用方应保留静态 fallback）。
     */
    bool ComputeKeyBodyPosFK();

    bool IsFkAvailable() const;

    // Sim-to-Real: episode 边界（reset / 进入 RL state）调用，清空动作历史避免上 episode 残留污染
    // K 由 yaml 字段 actions_history_length 决定，缺省即 1（向后兼容旧 policy）
    void ClearActionsHistory();

    // rl functions
    virtual std::vector<float> Forward() = 0;
    std::vector<float> ComputeObservation();
    virtual void GetState(RobotState<float> *state) = 0;
    virtual void SetCommand(const RobotCommand<float> *command) = 0;
    void StateController(const RobotState<float> *state, RobotCommand<float> *command);
    void ComputeOutput(const std::vector<float> &actions, std::vector<float> &output_dof_pos, std::vector<float> &output_dof_vel, std::vector<float> &output_dof_tau);

    // yaml params
    void ReadYaml(const std::string& file_path, const std::string& file_name);

    // csv logger
    std::string csv_filename;
    void CSVInit(std::string robot_name);
    void CSVLogger(const std::vector<float> &torque, const std::vector<float> &tau_est, const std::vector<float> &joint_pos, const std::vector<float> &joint_pos_target, const std::vector<float> &joint_vel);
    // Dedicated observation logger for sim-to-real gap analysis.
    // Writes a different file (obs.csv) with a complete obs/action schema and
    // is size-safe (zero-pads missing fields). Enabled independently of the
    // legacy CSVLogger above.
    void CSVInitObs(std::string robot_name);
    void CSVLoggerObs(int t,
                      const std::vector<float> &commands,
                      const std::vector<float> &base_ang_vel,
                      const std::vector<float> &base_quat,
                      const std::vector<float> &dof_pos,
                      const std::vector<float> &dof_pos_target,
                      const std::vector<float> &dof_vel,
                      const std::vector<float> &actions);

    // Full-trajectory trace logger.
    // Runs independently of the RL inference loop: every sample contains the
    // current FSM state name, so you can slice the csv by state offline
    // (Passive / GetUp / RLLocomotion / GetDown) and compare against sim
    // recordings. All vector fields are size-safe (zero-padded).
    void CSVInitTrace(std::string robot_name,
        std::string filename = "trace.csv");
    void CSVLoggerTrace(float t_sec,
                        const std::string &fsm_state,
                        const std::vector<float> &commands,
                        const std::vector<float> &base_ang_vel,
                        const std::vector<float> &base_quat,
                        const std::vector<float> &projected_gravity,
                        const std::vector<float> &dof_pos,
                        const std::vector<float> &dof_pos_target,
                        const std::vector<float> &dof_vel,
                        const std::vector<float> &policy_delta,
                        const std::vector<float> &actions);

    void CSVInitPolicyObs(const std::string& robot_path, int obs_dim,
                          const std::string& filename);
    void CSVLoggerPolicyObs(int episode_step, float progress,
                            const std::vector<float>& policy_obs,
                            const std::vector<float>& actions);

    // control
    Control control;
    void KeyboardInterface();

    // history buffer
    ObservationBuffer history_obs_buf;
    std::vector<float> history_obs;

    // others
    int motiontime = 0;
    std::string robot_name, config_name;
    std::string policy_obs_csv_filename;
    bool simulation_running = true;
    std::string ang_vel_axis = "body";  // "world" or "body"
    unsigned long long episode_length_buf = 0;
    float motion_length = 0.0;
    int InverseJointMapping(int idx) const;

    // Motion tracking (for mimic/dance tasks)
    std::unique_ptr<MotionLoader> motion_loader;

    // protect func
    void TorqueProtect(const std::vector<float> &origin_output_dof_tau);
    void AttitudeProtect(const std::vector<float> &quaternion, float pitch_threshold, float roll_threshold);

    // rl module
    std::unique_ptr<InferenceRuntime::Model> model;
    // output buffer
    std::vector<float> output_dof_tau;
    std::vector<float> output_dof_pos;
    std::vector<float> output_dof_vel;
    std::vector<float> output_dof_delta;

    // thread safety
    std::mutex model_mutex;

    // online forward kinematics helper (lazy 创建; USE_MUJOCO=OFF 时永远 IsLoaded()==false)
    std::unique_ptr<KinematicsFK> fk_helper_;
};

class RLFSMState : public FSMState
{
public:
    RLFSMState(RL& rl, const std::string& name)
        : FSMState(name), rl(rl), fsm_state(nullptr), fsm_command(nullptr) {}

    RL& rl;
    const RobotState<float> *fsm_state;
    RobotCommand<float> *fsm_command;

    bool Interpolate(
        float& percent,
        const std::vector<float>& start_pos,
        const std::vector<float>& target_pos,
        float duration_seconds,
        const std::string& description = "",
        bool use_fixed_gains = true
    );

    void RLControl();
};

#endif // RL_SDK_HPP
