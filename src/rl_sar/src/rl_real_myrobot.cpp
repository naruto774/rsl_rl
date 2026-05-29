/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 *
 * rl_real_myrobot.cpp
 * Real robot deployment via ZMQ communication with Raspberry Pi.
 * Protocol matches laptop_zmqtest.py:
 *   - SUB low_state: topic + " " + ts(f64) + flat49(f64[49])
 *   - PUB low_cmd:   topic + " " + ts(f64) + q_cmd(f64[21])
 */

#include "rl_real_myrobot.hpp"
#include <array>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <yaml-cpp/yaml.h>

static float WrapToPi(float angle)
{
    constexpr float kPi = 3.14159265358979323846f;
    while (angle > kPi) angle -= 2.0f * kPi;
    while (angle < -kPi) angle += 2.0f * kPi;
    return angle;
}

RL_Real::RL_Real(int argc, char **argv)
{
#if defined(USE_ROS1) && defined(USE_ROS)
    ros::NodeHandle nh;
    this->cmd_vel_subscriber = nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 10, &RL_Real::CmdvelCallback, this);
#elif defined(USE_ROS2) && defined(USE_ROS)
    ros2_node = std::make_shared<rclcpp::Node>("rl_real_node");
    this->cmd_vel_subscriber = ros2_node->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", rclcpp::SystemDefaultsQoS(),
        [this] (const geometry_msgs::msg::Twist::SharedPtr msg) {this->CmdvelCallback(msg);}
    );
#endif

    // Read params from yaml
    // Set config_name to "robot_lab_real" BEFORE FSM init, so FSM uses the correct config
    this->ang_vel_axis = "body";
    this->robot_name = "myrobot";
    this->config_name = "robot_lab_real";  // IsaacSim joint order for real robot
    this->ReadYaml(this->robot_name + "/robot_lab_real", "base.yaml");  // base params with IsaacSim joint order

    // Auto load FSM by robot_name
    if (FSMManager::GetInstance().IsTypeSupported(this->robot_name))
    {
        auto fsm_ptr = FSMManager::GetInstance().CreateFSM(this->robot_name, this);
        if (fsm_ptr)
        {
            this->fsm = *fsm_ptr;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "[FSM] No FSM registered for robot: " << this->robot_name << std::endl;
    }

    // Init robot
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();

    // Init ZMQ communication
    if (!this->InitZMQ())
    {
        throw std::runtime_error("Failed to initialize ZMQ communication");
    }

    // Loop
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Real::KeyboardInterface, this));
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Real::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Real::RunModel, this));
    
    this->loop_keyboard->start();
    this->loop_control->start();
    this->loop_rl->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    this->plot_target_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.002, std::bind(&RL_Real::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    // Full-trajectory logger: runs independently of the RL loop, writes
    // every FSM state (Passive -> GetUp -> RLLocomotion -> GetDown -> ...)
    // into <policy_dir>/myrobot/trace.csv until the program is killed.
    // Period 20 ms (50 Hz) is sufficient for offline motion diagnosis and
    // keeps file size reasonable (~6 MB / hour per 21-DoF robot).
    this->CSVInitTrace(this->robot_name);
    this->log_t0 = std::chrono::steady_clock::now();
    this->loop_log = std::make_shared<LoopFunc>("loop_log", 0.02, std::bind(&RL_Real::LogTick, this));
    this->loop_log->start();
#endif

    std::cout << LOGGER::INFO << "RL_Real (myrobot) started" << std::endl;
    std::cout << LOGGER::INFO << "[ZMQ] SUB: tcp://" << zmq_pi_ip << ":" << zmq_state_port << " topic=" << zmq_state_topic << std::endl;
    std::cout << LOGGER::INFO << "[ZMQ] PUB: tcp://*:" << zmq_cmd_port << " topic=" << zmq_cmd_topic << std::endl;
}

RL_Real::~RL_Real()
{
    this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef CSV_LOGGER
    if (this->loop_log) this->loop_log->shutdown();
#endif
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    this->CloseZMQ();
    std::cout << LOGGER::INFO << "RL_Real exit" << std::endl;
}

bool RL_Real::InitZMQ()
{
    zmq_context = zmq_ctx_new();
    if (!zmq_context)
    {
        std::cout << LOGGER::ERROR << "[ZMQ] Failed to create context" << std::endl;
        return false;
    }

    // SUB socket: connect to Pi's state publisher
    zmq_sub = zmq_socket(zmq_context, ZMQ_SUB);
    if (!zmq_sub)
    {
        std::cout << LOGGER::ERROR << "[ZMQ] Failed to create SUB socket" << std::endl;
        return false;
    }

    // Enable CONFLATE to keep only the newest message (real-time control)
    int conflate = 1;
    zmq_setsockopt(zmq_sub, ZMQ_CONFLATE, &conflate, sizeof(conflate));

    std::string sub_endpoint = "tcp://" + zmq_pi_ip + ":" + std::to_string(zmq_state_port);
    if (zmq_connect(zmq_sub, sub_endpoint.c_str()) != 0)
    {
        std::cout << LOGGER::ERROR << "[ZMQ] Failed to connect SUB to " << sub_endpoint << std::endl;
        return false;
    }

    // Subscribe to state topic (with trailing space as per protocol)
    std::string sub_filter = zmq_state_topic + " ";
    zmq_setsockopt(zmq_sub, ZMQ_SUBSCRIBE, sub_filter.c_str(), sub_filter.size());

    // PUB socket: bind for Pi to connect
    zmq_pub = zmq_socket(zmq_context, ZMQ_PUB);
    if (!zmq_pub)
    {
        std::cout << LOGGER::ERROR << "[ZMQ] Failed to create PUB socket" << std::endl;
        return false;
    }

    std::string pub_endpoint = "tcp://*:" + std::to_string(zmq_cmd_port);
    if (zmq_bind(zmq_pub, pub_endpoint.c_str()) != 0)
    {
        std::cout << LOGGER::ERROR << "[ZMQ] Failed to bind PUB to " << pub_endpoint << std::endl;
        return false;
    }

    return true;
}

void RL_Real::CloseZMQ()
{
    if (zmq_sub)
    {
        zmq_close(zmq_sub);
        zmq_sub = nullptr;
    }
    if (zmq_pub)
    {
        zmq_close(zmq_pub);
        zmq_pub = nullptr;
    }
    if (zmq_context)
    {
        zmq_ctx_destroy(zmq_context);
        zmq_context = nullptr;
    }
}

bool RL_Real::ReceiveState()
{
    // Non-blocking receive
    zmq_msg_t msg;
    zmq_msg_init(&msg);

    int rc = zmq_msg_recv(&msg, zmq_sub, ZMQ_DONTWAIT);
    if (rc < 0)
    {
        zmq_msg_close(&msg);
        return false;  // No message available
    }

    // Parse message: "topic " + payload
    size_t msg_size = zmq_msg_size(&msg);
    const char *data = static_cast<const char *>(zmq_msg_data(&msg));

    std::string topic_prefix = zmq_state_topic + " ";
    if (msg_size < topic_prefix.size() + ZMQ_STATE_PAYLOAD_BYTES)
    {
        zmq_msg_close(&msg);
        return false;
    }

    // Skip topic prefix
    const char *payload = data + topic_prefix.size();

    // Parse payload: ts(f64) + flat49(f64[49])
    std::lock_guard<std::mutex> lock(state_mutex);

    std::memcpy(&last_state_ts, payload, sizeof(double));
    payload += sizeof(double);

    const double *flat = reinterpret_cast<const double *>(payload);

    // flat[0:21] = q, flat[21:42] = qd, flat[42:45] = ang_vel, flat[45:49] = quat
    for (int i = 0; i < 21; ++i)
    {
        state_q[i] = static_cast<float>(flat[i]);
        state_qd[i] = static_cast<float>(flat[21 + i]);
    }
    for (int i = 0; i < 3; ++i)
    {
        state_ang_vel[i] = static_cast<float>(flat[42 + i]);
    }
    for (int i = 0; i < 4; ++i)
    {
        state_quat[i] = static_cast<float>(flat[45 + i]);
    }

    state_received.store(true);

    // ------------------------------------------------------------------------
    // [上半身 obs 字段诊断打印] — 节流到 ~1Hz
    //
    // 排查链路（rl_sar/dance）：
    //   电机端编码器 → CAN → Pi motor_reader → ZMQ "low_state" 包 flat[0:42]
    //                                            ↑↑↑  这里就是当前位置
    //   → PC SUB → state_q/state_qd → robot_state.motor_state.q/dq
    //   → obs.dof_pos/dof_vel → policy.forward(obs)
    //
    // PC 端 (上面循环 flat[i] → state_q[i]) 是无脑全量复制，21 路都填，必然 OK。
    // 因此若打印值有以下任一异常 → 故障点铁定在 Pi 端及更上游：
    //
    //   1) 上半身 q 长期为 ~0 且与下半身/IMU 抖动幅度不匹配
    //      → Pi motor_reader 没读到这几路电机
    //        (motor_ids 配置缺失 / CAN bus 未 attach / 电机扭矩使能未开)
    //   2) 上半身 q 完全不变但下半身实时跟随
    //      → 上半身电机在线但反馈被 freeze（电机进入掉电/待机模式）
    //   3) dq 始终为 0 而 q 在变
    //      → Pi 端只填了位置、速度估计未做 (差分窗口失效)
    //
    // 仅用于诊断，对策略行为无影响。
    // 上半身关节（IsaacSim 训练顺序，与 obs.dof_pos 完全对齐）：
    //   2=waist_pitch  5=waist_yaw  8=head
    //   9=sh_pitch_R 10=sh_pitch_L 13=sh_roll_R 14=sh_roll_L
    //  17=elbow_R   18=elbow_L
    // ------------------------------------------------------------------------
    {
        static const std::array<int, 9> upper_idx = {2, 5, 8, 9, 10, 13, 14, 17, 18};
        static const char* upper_name[9] = {
            "waP", "waY", "hd", "spR", "spL", "srR", "srL", "elR", "elL"
        };
        static auto last_dbg_t = std::chrono::steady_clock::now();
        auto now_t = std::chrono::steady_clock::now();
        if (std::chrono::duration<float>(now_t - last_dbg_t).count() >= 1.0f)
        {
            last_dbg_t = now_t;

            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3);
            oss << "[UpperObs] q:";
            for (int k = 0; k < 9; ++k)
            {
                oss << " " << upper_name[k] << "=" << state_q[upper_idx[k]];
            }
            oss << std::setprecision(2) << " | dq:";
            for (int k = 0; k < 9; ++k)
            {
                oss << " " << upper_name[k] << "=" << state_qd[upper_idx[k]];
            }
            std::cout << oss.str() << std::endl;
        }
    }

    zmq_msg_close(&msg);
    return true;
}

void RL_Real::SendCommand()
{
    // Build message: "topic " + ts(f64) + q_cmd(f64[21])
    std::string topic_prefix = zmq_cmd_topic + " ";
    size_t total_size = topic_prefix.size() + ZMQ_CMD_PAYLOAD_BYTES;

    std::vector<char> buffer(total_size);
    char *ptr = buffer.data();

    // Copy topic prefix
    std::memcpy(ptr, topic_prefix.c_str(), topic_prefix.size());
    ptr += topic_prefix.size();

    // Timestamp
    auto now = std::chrono::system_clock::now();
    double ts = std::chrono::duration<double>(now.time_since_epoch()).count();
    std::memcpy(ptr, &ts, sizeof(double));
    ptr += sizeof(double);

    // q_cmd as float64
    for (int i = 0; i < 21; ++i)
    {
        double val = static_cast<double>(cmd_q[i]);
        std::memcpy(ptr, &val, sizeof(double));
        ptr += sizeof(double);
    }

    zmq_send(zmq_pub, buffer.data(), total_size, ZMQ_DONTWAIT);
}

void RL_Real::GetState(RobotState<float> *state)
{
    // Try to receive new state (non-blocking)
    ReceiveState();

    // Copy latest state to RobotState structure
    std::lock_guard<std::mutex> lock(state_mutex);

    // IMU
    for (int i = 0; i < 4; ++i)
    {
        state->imu.quaternion[i] = state_quat[i];
    }

    for (int i = 0; i < 3; ++i)
    {
        // state->imu.gyroscope[i] = state_ang_vel[i];
        // x 轴限幅 [-1.5, 1.5]
        state->imu.gyroscope[0] = std::clamp(state_ang_vel[0], -1.5f, 1.5f);
        // y 轴限幅 [-0.5, 0.5]
        state->imu.gyroscope[1] = std::clamp(state_ang_vel[1], -0.5f, 0.5f);
        // z 轴限幅 [-1.0, 1.0]
        state->imu.gyroscope[2] = std::clamp(state_ang_vel[2], -1.0f, 1.0f);
    }

    // Joint state (no mapping needed - Pi uses IsaacSim order)
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        state->motor_state.q[i] = state_q[i];
        state->motor_state.dq[i] = state_qd[i];
        state->motor_state.tau_est[i] = 0.0f;
    }
}

void RL_Real::SetCommand(const RobotCommand<float> *command)
{
    // Copy command to send buffer (no mapping needed)
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        cmd_q[i] = command->motor_command.q[i];
    }

    // DEBUG: Print every 200 iterations (~1 second at 200Hz)
    static int cmd_debug_counter = 0;
    if (++cmd_debug_counter % 200 == 0)
    {
        std::cout << "[DEBUG] SetCommand cmd_q[0:3]: " << cmd_q[0] << ", " << cmd_q[1] << ", " << cmd_q[2] << std::endl;
    }

    // Send via ZMQ
    SendCommand();
}

void RL_Real::RobotControl()
{
    this->GetState(&this->robot_state);

    this->StateController(&this->robot_state, &this->robot_command);

    this->control.ClearInput();

    this->SetCommand(&this->robot_command);
}

// 让 loop_rl 与当前 config 的 policy_step_time 对齐。
// 数学含义：训练时每隔 dt_pol 调用一次 policy 推理，部署侧必须保持同样的采样率，
// 否则 obs 序列的相位 / 速度估计 / progress 都会偏移训练分布，导致 OOD。
//   - locomotion (robot_lab_real/base.yaml + robot_lab_real/config.yaml):
//       base.yaml 设了 dt=0.005, decimation=4 → 20ms (50Hz)
//   - whole_body_tracking/config.yaml:
//       policy_step_time=0.01666... → ~16.7ms (60Hz)
// GetPolicyStepTime() 优先用 config 里的 policy_step_time，没有时回退到 dt*decimation。
void RL_Real::UpdatePolicyLoopPeriod()
{
    if (!this->loop_rl) return;
    const float period = this->GetPolicyStepTime();
    this->loop_rl->setPeriod(period);
    std::cout << LOGGER::INFO << "[PolicyLoop] period=" << period << "s ("
              << (1.0f / std::max(period, 1e-6f)) << " Hz) for config=" << this->config_name << std::endl;
}

// 离线 FK 结果加载：把 default_pose_static_ref.yaml 里的 root_z_ref 和
// 21×3 key_body_pos_rel_flat 读到成员变量。
//
// 数学约束：
//   * yaml 是通过 compute_default_pose_fk.py 在 default_dof_pos 姿态下、
//     base 朝向 = identity、base z = ROOT_Z_REF 时跑 mj_forward 得到的，
//     与 rl_sim_mujoco.cpp 中 `xpos[body] - xpos[root]` 的提取逻辑严格一致。
//   * 顺序：以 policy joint idx 为外层维度，3 维世界系坐标为内层；
//     这样 RunModel 里可以直接整段拷贝到 obs.key_body_pos_rel，
//     rl_sdk.cpp 用 key_body_joint_indices 索引 13 个 key body 时取值正确。
//
// 加载失败时返回 false，调用方退化为零填充。
bool RL_Real::LoadDanceStaticRef()
{
    const std::string yaml_path = std::string(POLICY_DIR) +
        "/" + this->robot_name + "/whole_body_tracking/default_pose_static_ref.yaml";
    const std::string root_key = "myrobot/whole_body_tracking_static_ref";

    YAML::Node root_node;
    try
    {
        root_node = YAML::LoadFile(yaml_path);
    }
    catch (const YAML::Exception& e)
    {
        std::cout << LOGGER::WARNING
                  << "[DanceStaticRef] failed to load " << yaml_path
                  << " : " << e.what() << std::endl;
        return false;
    }

    YAML::Node node = root_node[root_key];
    if (!node)
    {
        std::cout << LOGGER::WARNING
                  << "[DanceStaticRef] yaml missing root key '" << root_key << "'" << std::endl;
        return false;
    }

    try
    {
        this->dance_root_z_ref = node["root_z_ref"].as<float>();
        const int expected_dim = this->params.Get<int>("num_of_dofs") * 3;
        this->dance_key_body_ref.clear();
        for (const auto& v : node["key_body_pos_rel_flat"])
        {
            this->dance_key_body_ref.push_back(v.as<float>());
        }
        if (static_cast<int>(this->dance_key_body_ref.size()) != expected_dim)
        {
            std::cout << LOGGER::WARNING
                      << "[DanceStaticRef] key_body_pos_rel_flat size mismatch: got="
                      << this->dance_key_body_ref.size()
                      << ", expected=" << expected_dim << std::endl;
            this->dance_key_body_ref.clear();
            return false;
        }
    }
    catch (const YAML::Exception& e)
    {
        std::cout << LOGGER::WARNING
                  << "[DanceStaticRef] yaml field parse error: " << e.what() << std::endl;
        this->dance_key_body_ref.clear();
        return false;
    }

    std::cout << LOGGER::INFO
              << "[DanceStaticRef] loaded root_z_ref=" << this->dance_root_z_ref
              << ", key_body_pos_rel_flat.size=" << this->dance_key_body_ref.size()
              << " from " << yaml_path << std::endl;
    return true;
}

void RL_Real::OnPolicyConfigLoaded()
{
    UpdatePolicyLoopPeriod();

    // 首次切到 whole_body_tracking 时:
    //   1. 优先尝试在线 FK（rl_sdk 的 LoadFkModel + ComputeKeyBodyPosFK）；
    //   2. 在线 FK 不可用时回落到离线静态参考 yaml（compute_default_pose_fk.py 产出）；
    //   3. 两者都失败再退化为零填充，所有 ~40 维 key_body_pos 通道完全 OOD。
    //   同时打印一个安全告警，让操作者知道当前 obs 完整度。
    if (this->config_name == "whole_body_tracking" &&
        this->last_loaded_config_name != "whole_body_tracking")
    {
        // 1) 尝试加载在线 FK：用 dance config 的 joint_mapping + MJCF。
        //    成功后 RunModel 里每周期都会调一次 ComputeKeyBodyPosFK 写 obs.key_body_pos_rel。
        const std::string mjcf_path = std::string(CMAKE_CURRENT_SOURCE_DIR) +
            "/../rl_sar_zoo/" + this->robot_name + "_description/mjcf/" +
            this->robot_name + ".xml";
        const auto joint_mapping = this->params.Get<std::vector<int>>("joint_mapping");
        const bool fk_ok = this->LoadFkModel(mjcf_path, joint_mapping, "base_link");
        this->dance_online_fk_enabled = fk_ok;

        // 2) 加载离线静态参考（兜底，FK 不可用时用）；同时也提供 root_z_ref（实机无 base_pos 估计）
        this->dance_static_ref_loaded = this->LoadDanceStaticRef();

        // 真机操作安全提示。每次切到 dance 都打印一遍当前策略 / obs 完整度,
        // 让操作员上手前心里有数 (action 物理含义 / 哪些 obs 是 OOD)。
        const std::string action_type =
            this->params.Get<std::string>("action_type", "default_position");
        const float action_scale_first =
            this->params.Get<std::vector<float>>("action_scale", {0.25f}).front();
        const float action_clip_first =
            this->params.Get<std::vector<float>>("clip_actions_upper", {1.25f}).front();
        const int dance_obs_dim =
            this->params.Get<int>("num_observations", 91);

        std::cout << "\n" << LOGGER::WARNING
                  << "============================================================" << std::endl;
        std::cout << LOGGER::WARNING
                  << "[SafetyCheck] Entering DANCE (whole_body_tracking)." << std::endl;
        std::cout << LOGGER::WARNING
                  << "  - Policy obs dim: " << dance_obs_dim
                  << " (89 AMP obs + 2 policy-only base_xy)." << std::endl;
        if (action_type == "relative_position")
        {
            std::cout << LOGGER::WARNING
                      << "  - Action: DELTA (relative_position). per-step q_target =" << std::endl;
            std::cout << LOGGER::WARNING
                      << "    q_current + " << action_scale_first
                      << " * clip(a, +/- " << (action_clip_first / std::max(action_scale_first, 1e-6f))
                      << "), max |delta| <= " << action_clip_first << " rad." << std::endl;
            std::cout << LOGGER::WARNING
                      << "    Then q_target is clamped to [q_target_lower, q_target_upper]." << std::endl;
        }
        else
        {
            std::cout << LOGGER::WARNING
                      << "  - Action: ABSOLUTE (legacy soft-joint-pos-action), q_target" << std::endl;
            std::cout << LOGGER::WARNING
                      << "    can swing across the full soft joint range. NOT aligned with" << std::endl;
            std::cout << LOGGER::WARNING
                      << "    current training side, you probably want action_type=relative_position." << std::endl;
        }
        if (this->dance_online_fk_enabled)
        {
            std::cout << LOGGER::WARNING
                      << "  - key_body_pos_rel (39d): ONLINE FK (mj_kinematics on MJCF, tracks" << std::endl;
            std::cout << LOGGER::WARNING
                      << "    real-time joint angles + IMU quat). Self-consistent with training;" << std::endl;
            std::cout << LOGGER::WARNING
                      << "    residual OOD only on base_pos.z (no z-estimator)." << std::endl;
        }
        else if (this->dance_static_ref_loaded)
        {
            std::cout << LOGGER::WARNING
                      << "  - key_body_pos_rel (39d): STATIC reference from offline FK" << std::endl;
            std::cout << LOGGER::WARNING
                      << "    (default_pose_static_ref.yaml). Posture-locked, doesn't" << std::endl;
            std::cout << LOGGER::WARNING
                      << "    track motion -> partial OOD on these ~40 dims." << std::endl;
        }
        else
        {
            std::cout << LOGGER::WARNING
                      << "  - key_body_pos_rel (39d): ZERO-FILLED (FK + static ref both missing)" << std::endl;
            std::cout << LOGGER::WARNING
                      << "    -> policy operates strongly OOD on these ~40 dims." << std::endl;
        }
        std::cout << LOGGER::WARNING
                  << "  - base_xy (2d): forced to (0, 0) on real robot (no odometry)." << std::endl;
        std::cout << LOGGER::WARNING
                  << "    Lies inside training U(-obs_noise_base_xy, +obs_noise_base_xy)," << std::endl;
        std::cout << LOGGER::WARNING
                  << "    so first half of episode is in-distribution; long-horizon CoM" << std::endl;
        std::cout << LOGGER::WARNING
                  << "    drift cannot be inferred -> mild late-episode OOD." << std::endl;
        std::cout << LOGGER::WARNING
                  << "  - Make sure: gantry attached / e-stop in hand / clear area." << std::endl;
        std::cout << LOGGER::WARNING
                  << "============================================================\n" << std::endl;
    }
    this->last_loaded_config_name = this->config_name;
}

void RL_Real::RunModel()
{
    if (!this->rl_init_done)
    {
        this->yaw_hold_target_initialized = false;
        return;
    }

    this->episode_length_buf += 1;
    this->obs.ang_vel = this->robot_state.imu.gyroscope;
    this->obs.base_quat = this->robot_state.imu.quaternion;

    // Yaw hold PD controller (from rl_sim_mujoco.cpp)
    float cmd_yaw_final = this->control.yaw;
    if (this->params.Get<bool>("yaw_hold_enable", false))
    {
        std::vector<float> euler = QuaternionToEuler(this->obs.base_quat);
        float yaw_current = euler[2];
        float wz = this->obs.ang_vel[2];

        float kp = this->params.Get<float>("yaw_hold_kp", 0.5f);
        float kd = this->params.Get<float>("yaw_hold_kd", 0.08f);
        float correction_limit = this->params.Get<float>("yaw_hold_max_correction", 0.3f);
        float cmd_limit = this->params.Get<float>("yaw_hold_cmd_limit", 0.5f);
        float user_yaw_deadband = this->params.Get<float>("yaw_hold_user_cmd_deadband", 0.05f);
        float forward_cmd_min = this->params.Get<float>("yaw_hold_forward_cmd_min", 0.05f);

        bool user_turning = std::fabs(this->control.yaw) > user_yaw_deadband;
        bool forward_moving = std::fabs(this->control.x) > forward_cmd_min;

        if (!this->yaw_hold_target_initialized || user_turning || !forward_moving)
        {
            this->yaw_hold_target = yaw_current;
            this->yaw_hold_target_initialized = true;
        }

        if (!user_turning && forward_moving)
        {
            float yaw_err = WrapToPi(this->yaw_hold_target - yaw_current);
            float correction = kp * yaw_err - kd * wz;
            correction = clamp(correction, -correction_limit, correction_limit);
            cmd_yaw_final = clamp(this->control.yaw + correction, -cmd_limit, cmd_limit);
        }
    }
    else
    {
        this->yaw_hold_target_initialized = false;
    }

    this->obs.commands = {this->control.x, this->control.y, cmd_yaw_final};

#if !defined(USE_CMAKE) && defined(USE_ROS)
    if (this->control.navigation_mode)
    {
        this->obs.commands = {(float)this->cmd_vel.linear.x, (float)this->cmd_vel.linear.y, (float)this->cmd_vel.angular.z};
    }
#endif

    this->obs.dof_pos = this->robot_state.motor_state.q;
    this->obs.dof_vel = this->robot_state.motor_state.dq;

    // ----------------------------------------------------------------------
    // Dance (whole_body_tracking) 专用笛卡尔观测的填充策略，按优先级 fallback:
    //
    //   [1] 在线 FK（最佳）: rl_sdk 用 MJCF 跑一次 mj_kinematics，
    //       输入 = (obs.dof_pos 当前关节角, obs.base_quat 当前 IMU 四元数)，
    //       输出 = obs.key_body_pos_rel(笛卡尔几何，~10-50μs)。
    //       与训练侧完全一致的 layout（actuator->joint->body 链）。
    //
    //   [2] 离线静态参考: 在线 FK 不可用（USE_MUJOCO 关 / MJCF 加载失败）时，
    //       用 default_dof_pos 姿态下离线烧的 yaml 常量。policy 一直看到机器人
    //       保持在 entry pose，~0.1m 量级 OOD。
    //
    //   [3] 零填充: 两者都失败时退化为 0，机器人在 root 一点上塌缩，~1m 量级 OOD，
    //       仅保证不崩。
    //
    // root_z 是 base 在世界系 z 坐标，实机没有外部位姿估计，三档都只能用训练标称值
    // dance_root_z_ref（默认 0.23m），这部分残留 OOD 不可避免。
    //
    // 对 locomotion config 这两项不在 observations 列表里，无副作用。
    if (this->config_name == "whole_body_tracking")
    {
        // root_z: 实机没有 base z 估计，统一用训练标称 0.23m（来自静态参考 yaml 或 hard-coded）
        const float root_z = this->dance_static_ref_loaded
                             ? this->dance_root_z_ref
                             : this->params.Get<float>("root_z_ref", 0.23f);
        this->obs.base_pos = {0.0f, 0.0f, root_z};

        // key_body_pos_rel: 优先在线 FK
        bool fk_filled = false;
        if (this->dance_online_fk_enabled && this->IsFkAvailable())
        {
            // ComputeKeyBodyPosFK 内部直接读 obs.dof_pos + obs.base_quat、写 obs.key_body_pos_rel
            fk_filled = this->ComputeKeyBodyPosFK();
        }
        if (!fk_filled)
        {
            if (this->dance_static_ref_loaded)
            {
                this->obs.key_body_pos_rel = this->dance_key_body_ref;
            }
            else
            {
                this->obs.key_body_pos_rel.assign(
                    this->params.Get<int>("num_of_dofs") * 3, 0.0f);
            }
        }
    }
    else
    {
        this->obs.base_pos = {0.0f, 0.0f, 0.0f};
        this->obs.key_body_pos_rel.assign(
            this->params.Get<int>("num_of_dofs") * 3, 0.0f);
    }
    // ----------------------------------------------------------------------

    this->obs.actions = this->Forward();

    // Sim-to-Real: maintain K frames of clipped action history (newest-first).
    // This matches IsaacLab's last_action timing: the next observation sees
    // the action produced at this inference step as a_{t-1}.
    {
        const int K = this->params.Get<int>("actions_history_length", 1);
        this->obs.actions_history.push_front(this->obs.actions);
        while (static_cast<int>(this->obs.actions_history.size()) > K)
        {
            this->obs.actions_history.pop_back();
        }
    }

    this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

    // Dance 状态的实时健康度监控（节流到 ~1Hz 输出，避免淹没终端）：
    //   mean|action|        :  policy 输出幅度，过小通常意味着归一化没烧进 jit / obs 异常；
    //                          过大说明 policy 在做剧烈姿态切换，需要确认机械准备就绪。
    //   mean|q_target-def|  :  实际下发到底层 PD 的目标 - default_dof_pos 的平均绝对偏离，
    //                          能直接判断"机器人当前是不是真的在跳，还是 hold 在默认位"。
    if (this->config_name == "whole_body_tracking" && !this->obs.actions.empty())
    {
        const int policy_hz = static_cast<int>(std::round(1.0f / std::max(this->GetPolicyStepTime(), 1e-6f)));
        const int every_n = std::max(1, policy_hz);  // ≈1Hz
        static int dance_dbg_counter = 0;
        if (++dance_dbg_counter % every_n == 0)
        {
            float mean_abs_action = 0.0f;
            for (float a : this->obs.actions) mean_abs_action += std::fabs(a);
            mean_abs_action /= static_cast<float>(this->obs.actions.size());

            float mean_abs_q_delta = 0.0f;
            const auto default_q = this->params.Get<std::vector<float>>("default_dof_pos");
            const size_t n = std::min(this->output_dof_pos.size(), default_q.size());
            for (size_t i = 0; i < n; ++i)
            {
                mean_abs_q_delta += std::fabs(this->output_dof_pos[i] - default_q[i]);
            }
            if (n > 0) mean_abs_q_delta /= static_cast<float>(n);

            std::cout << "\n" << LOGGER::INFO
                      << "[DanceMonitor] mean|action|=" << mean_abs_action
                      << " mean|q_target-default|=" << mean_abs_q_delta
                      << " ep_step=" << this->episode_length_buf
                      << std::endl;
        }
    }

    // DEBUG: Print every 50 iterations (~1 second at 50Hz)
    static int debug_counter = 0;
    // if (++debug_counter % 50 == 0)
    // {
    //     auto default_pos = this->params.Get<std::vector<float>>("default_dof_pos");
    //     std::cout << "\n[DEBUG] ======== RL Output ========" << std::endl;
    //     std::cout << "[DEBUG] obs.dof_pos[0:3]: " << this->obs.dof_pos[0] << ", " << this->obs.dof_pos[1] << ", " << this->obs.dof_pos[2] << std::endl;
    //     std::cout << "[DEBUG] default_dof_pos[0:3]: " << default_pos[0] << ", " << default_pos[1] << ", " << default_pos[2] << std::endl;
    //     std::cout << "[DEBUG] actions[0:3]: " << this->obs.actions[0] << ", " << this->obs.actions[1] << ", " << this->obs.actions[2] << std::endl;
    //     std::cout << "[DEBUG] output_dof_pos[0:3]: " << this->output_dof_pos[0] << ", " << this->output_dof_pos[1] << ", " << this->output_dof_pos[2] << std::endl;
    //     std::cout << "[DEBUG] commands (x,y,yaw): " << this->obs.commands[0] << ", " << this->obs.commands[1] << ", " << this->obs.commands[2] << std::endl;
    // }

    if (!this->output_dof_pos.empty())
    {
        output_dof_pos_queue.push(this->output_dof_pos);
    }
    if (!this->output_dof_vel.empty())
    {
        output_dof_vel_queue.push(this->output_dof_vel);
    }
    if (!this->output_dof_tau.empty())
    {
        output_dof_tau_queue.push(this->output_dof_tau);
    }

    // Per-step observation logging has been moved to the dedicated
    // `loop_log` (see LogTick below) so that Passive / GetUp / GetDown are
    // also recorded. Do not add CSV writes here.
}

std::vector<float> RL_Real::Forward()
{
    std::unique_lock<std::mutex> lock(this->model_mutex, std::try_to_lock);

    if (!lock.owns_lock())
    {
        std::cout << LOGGER::WARNING << "Model is being reinitialized, using previous actions" << std::endl;
        return this->obs.actions;
    }

    std::vector<float> clamped_obs = this->ComputeObservation();

    std::vector<float> actions;
    if (!this->params.Get<std::vector<int>>("observations_history").empty())
    {
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.Get<std::vector<int>>("observations_history"));
        actions = this->model->forward({this->history_obs});
    }
    else
    {
        actions = this->model->forward({clamped_obs});
    }

    if (!this->params.Get<std::vector<float>>("clip_actions_upper").empty() && !this->params.Get<std::vector<float>>("clip_actions_lower").empty())
    {
        return clamp(actions, this->params.Get<std::vector<float>>("clip_actions_lower"), this->params.Get<std::vector<float>>("clip_actions_upper"));
    }
    else
    {
        return actions;
    }
}

#ifdef PLOT
void RL_Real::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
        this->plot_real_joint_pos[i].push_back(this->robot_state.motor_state.q[i]);
        this->plot_target_joint_pos[i].push_back(this->robot_command.motor_command.q[i]);
        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    plt::pause(0.0001);
}
#endif

#if !defined(USE_CMAKE) && defined(USE_ROS)
void RL_Real::CmdvelCallback(
#if defined(USE_ROS1) && defined(USE_ROS)
    const geometry_msgs::Twist::ConstPtr &msg
#elif defined(USE_ROS2) && defined(USE_ROS)
    const geometry_msgs::msg::Twist::SharedPtr msg
#endif
)
{
    this->cmd_vel = *msg;
}
#endif

#ifdef CSV_LOGGER
// Full-trajectory tick. Runs at 50 Hz regardless of FSM state.
// We read `robot_state` (populated by GetState -> ReceiveState at loop_control
// rate) and `robot_command` (populated by StateController in the same loop),
// plus the last RL inputs/outputs stored on `this->obs`. In non-RL states
// `obs.actions` may be empty, which is fine because CSVLoggerTrace zero-pads.
void RL_Real::LogTick()
{
    const float t_sec = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - this->log_t0).count();

    // FSM state name (e.g. "RLFSMStatePassive", "RLFSMStateRLLocomotion").
    std::string fsm_state = "Unknown";
    if (this->fsm.current_state_)
    {
        fsm_state = this->fsm.current_state_->GetStateName();
    }

    // Effective velocity commands (after yaw_hold etc. these are what goes
    // into the policy; in non-RL states `obs.commands` may be empty so we
    // fall back to the raw user input).
    std::vector<float> commands = this->obs.commands;
    if (commands.size() < 3)
    {
        commands = {this->control.x, this->control.y, this->control.yaw};
    }

    // Joint target actually being written to the low-level PD right now:
    // this is robot_command.motor_command.q, valid in every FSM state
    // (Passive holds, GetUp interpolates, RL writes policy output).
    const std::vector<float>& cmd_q_live = this->robot_command.motor_command.q;
    const std::vector<float> projected_gravity =
        QuatRotateInverse(this->robot_state.imu.quaternion, std::vector<float>{0.0f, 0.0f, -1.0f});

    this->CSVLoggerTrace(
        t_sec,
        fsm_state,
        commands,
        this->robot_state.imu.gyroscope,
        this->robot_state.imu.quaternion,
        projected_gravity,
        this->robot_state.motor_state.q,
        cmd_q_live,
        this->robot_state.motor_state.dq,
        this->output_dof_delta,
        this->obs.actions   // empty in non-RL states, zero-padded by logger
    );
}
#endif

#if defined(USE_ROS1) && defined(USE_ROS)
void signalHandler(int signum)
{
    ros::shutdown();
    exit(0);
}
#endif

int main(int argc, char **argv)
{
#if defined(USE_ROS1) && defined(USE_ROS)
    signal(SIGINT, signalHandler);
    ros::init(argc, argv, "rl_sar");
    RL_Real rl_sar(argc, argv);
    ros::spin();
#elif defined(USE_ROS2) && defined(USE_ROS)
    rclcpp::init(argc, argv);
    auto rl_sar = std::make_shared<RL_Real>(argc, argv);
    rclcpp::spin(rl_sar->ros2_node);
    rclcpp::shutdown();
#elif defined(USE_CMAKE) || !defined(USE_ROS)
    RL_Real rl_sar(argc, argv);
    while (1) { sleep(10); }
#endif

    return 0;
}
