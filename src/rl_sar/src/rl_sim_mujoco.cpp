/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sim_mujoco.hpp"

#ifdef RL_MUJOCO_TEST_CSV
#include <chrono>
#include <iomanip>
#include <sstream>
#endif

RL_Sim* RL_Sim::instance = nullptr;

static float WrapToPi(float angle)
{
    constexpr float kPi = 3.14159265358979323846f;
    while (angle > kPi) angle -= 2.0f * kPi;
    while (angle < -kPi) angle += 2.0f * kPi;
    return angle;
}

RL_Sim::RL_Sim(int argc, char **argv)
{
    // Set static instance pointer early for signal handler
    instance = this;

    if (argc < 3)
    {
        std::cout << LOGGER::ERROR << "Usage: " << argv[0] << " robot_name scene_name" << std::endl;
        throw std::runtime_error("Invalid arguments");
    }
    else
    {
        this->robot_name = argv[1];
        this->scene_name = argv[2];
    }

    this->ang_vel_axis = "body";

    // now launch mujoco
    std::cout << LOGGER::INFO << "[MuJoCo] Launching..." << std::endl;

    // 条件编译宏：仅 macOS 下 AVX 指令集才需要 Rosetta 2 支持
#if defined(__APPLE__) && defined(__AVX__)
    if (rosetta_error_msg)
    {
        DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
        std::exit(1);
    }
#endif

    // 打印mujoco版本信息，检查兼容性
    std::cout << LOGGER::INFO << "[MuJoCo] Version: " << mj_versionString() << std::endl;
    if (mjVERSION_HEADER != mj_version())
    {
        mju_error("Headers and library have different versions");
    }

    // 扫描插件目录，加载插件
    scanPluginLibraries();
    //mujoco gui 相机
    mjvCamera cam;
    mjv_defaultCamera(&cam);
    //gui 可视化选项
    mjvOption opt;
    mjv_defaultOption(&opt);
    // gui 交互扰动对象
    mjvPerturb pert;
    mjv_defaultPerturb(&pert);

    //将相机、选项、交互扰动对象传给 mj::Simulate
    sim = std::make_unique<mj::Simulate>(
        std::make_unique<mj::GlfwAdapter>(),
        &cam, &opt, &pert, /* is_passive = */ false);

    std::string filename = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/../rl_sar_zoo/" + this->robot_name + "_description/mjcf/" + this->scene_name + ".xml";

    // start physics thread
    std::thread physicsthreadhandle(&PhysicsThread, sim.get(), filename.c_str());
    physicsthreadhandle.detach();

    while (1)
    {
        if (d)
        {
            std::cout << LOGGER::INFO << "[MuJoCo] Data prepared" << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    this->mj_model = m;
    this->mj_data = d;
    // =================================================================
    // 🚨 探针代码：探测 MuJoCo 底层的真实执行器 (Actuator) 顺序 
    // =================================================================
    // std::cout << "\n=============================================" << std::endl;
    // std::cout << "🚨 MuJoCo 底层物理引擎执行器顺序探测 🚨" << std::endl;
    // for (int i = 0; i < this->mj_model->nu; ++i) {
    //     // 获取第 i 个执行器所驱动的关节内部 ID
    //     int jnt_id = this->mj_model->actuator_trnid[i * 2]; 
    //     // 根据关节 ID 获取它的真实名字
    //     const char* jnt_name = mj_id2name(this->mj_model, mjOBJ_JOINT, jnt_id);
    //     std::cout << "[MuJoCo Index " << i << "] -> 绑定的关节: " << (jnt_name ? jnt_name : "Unknown") << std::endl;
    // }
    // std::cout << "=============================================\n" << std::endl;
    // =================================================================
    this->SetupSysJoystick("/dev/input/js0", 16); // 16 bits joystick

    // read params from yaml
    this->ReadYaml(this->robot_name, "base.yaml");

    // Sim-to-Real: base.yaml 里没有 action_delay_ms_*，所以这里的采样默认得到 0 = 禁用；
    // 真正的延时值会在进入 RL 状态（config.yaml 被 InitRL 再次加载后）或 R 键复位时重采。
    this->ResampleActionDelay();

    // auto load FSM by robot_name
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

    // init robot
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();

    // loop
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Sim::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Sim::RunModel, this));
    this->loop_control->start();
    this->loop_rl->start();

    // keyboard
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Sim::KeyboardInterface, this));
    this->loop_keyboard->start();

    // joystick
    this->loop_joystick = std::make_shared<LoopFunc>("loop_joystick", 0.01, std::bind(&RL_Sim::GetSysJoystick, this));
    this->loop_joystick->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    this->plot_target_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.001, std::bind(&RL_Sim::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInitTrace(this->robot_name, "trace_sim.csv");
    this->log_t0 = std::chrono::steady_clock::now();
    this->loop_log = std::make_shared<LoopFunc>(
        "loop_log", 0.02, std::bind(&RL_Sim::LogTick, this));
    this->loop_log->start();
#endif

    std::cout << LOGGER::INFO << "RL_Sim start" << std::endl;

    // start simulation UI loop (blocking call)
    sim->RenderLoop();
}

RL_Sim::~RL_Sim()
{
    this->loop_keyboard->shutdown();
    this->loop_joystick->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
#ifdef CSV_LOGGER
    if (this->loop_log) this->loop_log->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;
}

// ============ Sim-to-Real: Action Delay helpers ============
// 从 yaml 读 [min,max] 范围重采本 episode 延时。区间退化或 max<=0 → 关闭延时。
void RL_Sim::ResampleActionDelay()
{
    const float min_ms = this->params.Get<float>("action_delay_ms_min", 0.0f);
    const float max_ms = this->params.Get<float>("action_delay_ms_max", 0.0f);
    if (max_ms <= 0.0f || max_ms < min_ms)
    {
        this->current_delay_ms_ = 0.0f;
        return;
    }
    std::uniform_real_distribution<float> dist(min_ms, max_ms);
    this->current_delay_ms_ = dist(this->delay_rng_);
    std::cout << LOGGER::INFO << "[Sim-to-Real] Action delay resampled: "
              << this->current_delay_ms_ << " ms (U[" << min_ms << ", " << max_ms << "])" << std::endl;
}

void RL_Sim::ClearActionDelayBuffer()
{
    std::lock_guard<std::mutex> lk(this->action_delay_mutex_);
    this->action_delay_buffer_.clear();
}

// 200Hz 调用：扫一遍暂存的动作，到期的 flush 到 output_dof_*_queue
// 复杂度 O(n) 但 n 极小（10~20ms 延时 × 50Hz 产出 ≈ 1 条）
void RL_Sim::DrainActionDelayBuffer()
{
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(this->action_delay_mutex_);
    while (!this->action_delay_buffer_.empty()
           && this->action_delay_buffer_.front().release_time <= now)
    {
        auto& front = this->action_delay_buffer_.front();
        if (!front.dof_pos.empty()) output_dof_pos_queue.push(std::move(front.dof_pos));
        if (!front.dof_vel.empty()) output_dof_vel_queue.push(std::move(front.dof_vel));
        if (!front.dof_tau.empty()) output_dof_tau_queue.push(std::move(front.dof_tau));
        this->action_delay_buffer_.pop_front();
    }
}

void RL_Sim::GetState(RobotState<float> *state)
{
    if (mj_data)
    {
        state->imu.quaternion[0] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 0]; 
        state->imu.quaternion[1] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 1]; 
        state->imu.quaternion[2] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 2];
        state->imu.quaternion[3] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 3]; 

        state->imu.gyroscope[0] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 4];
        state->imu.gyroscope[1] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 5];
        state->imu.gyroscope[2] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 6];

        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            state->motor_state.q[i] = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i]];
            state->motor_state.dq[i] = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + this->params.Get<int>("num_of_dofs")];
            state->motor_state.tau_est[i] = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + 2 * this->params.Get<int>("num_of_dofs")];
        }
    }
}

void RL_Sim::SetCommand(const RobotCommand<float> *command)
{
    if (mj_data)
    {
        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            mj_data->ctrl[this->params.Get<std::vector<int>>("joint_mapping")[i]] =
                command->motor_command.tau[i] +
                command->motor_command.kp[i] * (command->motor_command.q[i] - mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i]]) +
                command->motor_command.kd[i] * (command->motor_command.dq[i] - mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + this->params.Get<int>("num_of_dofs")]);
        }
    }
}

void RL_Sim::RobotControl()
{
    // Lock the sim mutex once for the entire control cycle to prevent race conditions
    const std::lock_guard<std::recursive_mutex> lock(sim->mtx);

    this->GetState(&this->robot_state);

    this->StateController(&this->robot_state, &this->robot_command);

    // Sim-to-Real: 200Hz 节拍里把到期的延时动作 flush 给下游 FSM
    this->DrainActionDelayBuffer();

    if (this->control.current_keyboard == Input::Keyboard::R || this->control.current_gamepad == Input::Gamepad::RB_Y)
    {
        if (this->mj_model && this->mj_data)
        {
            mj_resetData(this->mj_model, this->mj_data);
            mj_forward(this->mj_model, this->mj_data);
        }
        // Sim-to-Real: R 键复位视作 new episode，重采延时并清掉上一 episode 的滞留动作
        this->ResampleActionDelay();
        this->ClearActionDelayBuffer();
        // 同时清空动作历史，避免上一 episode 的 a_{t-1..t-K} 污染新 episode 的首步观测
        this->ClearActionsHistory();
    }
    if (this->control.current_keyboard == Input::Keyboard::Enter || this->control.current_gamepad == Input::Gamepad::RB_X)
    {
        if (simulation_running)
        {
            sim->run = 0;
            std::cout << std::endl << LOGGER::INFO << "Simulation Stop" << std::endl;
        }
        else
        {
            sim->run = 1;
            std::cout << std::endl << LOGGER::INFO << "Simulation Start" << std::endl;
        }
        simulation_running = !simulation_running;
    }

    this->control.ClearInput();

    this->SetCommand(&this->robot_command);
}

void RL_Sim::SetupSysJoystick(const std::string& device, int bits)
{
    this->sys_js = std::make_unique<Joystick>(device);
    if (!this->sys_js->isFound())
    {
        std::cout << LOGGER::ERROR << "Joystick [" << device << "] open failed." << std::endl;
        // exit(1);
    }

    this->sys_js_max_value = (1 << (bits - 1));
}

void RL_Sim::GetSysJoystick()
{
    // Clear all button event states
    for (int i = 0; i < 20; ++i)
    {
        this->sys_js_button[i].on_press = false;
        this->sys_js_button[i].on_release = false;
    }

    // Check if joystick is valid before using
    if (!this->sys_js)
    {
        return;
    }

    while (this->sys_js->sample(&this->sys_js_event))
    {
        if (this->sys_js_event.isButton())
        {
            this->sys_js_button[this->sys_js_event.number].update(this->sys_js_event.value);
        }
        else if (this->sys_js_event.isAxis())
        {
            double normalized = double(this->sys_js_event.value) / this->sys_js_max_value;
            if (std::abs(normalized) < this->axis_deadzone)
            {
                this->sys_js_axis[this->sys_js_event.number] = 0;
            }
            else
            {
                this->sys_js_axis[this->sys_js_event.number] = this->sys_js_event.value;
            }
        }
    }

    if (this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::A);
    if (this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::B);
    if (this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::X);
    if (this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->sys_js_button[4].on_press) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->sys_js_button[5].on_press) this->control.SetGamepad(Input::Gamepad::RB);
    if (this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::LStick);
    if (this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::RStick);
    if (this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if (this->sys_js_button[4].pressed && this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::LB_A);
    if (this->sys_js_button[4].pressed && this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::LB_B);
    if (this->sys_js_button[4].pressed && this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::LB_X);
    if (this->sys_js_button[4].pressed && this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if (this->sys_js_button[4].pressed && this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if (this->sys_js_button[4].pressed && this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if (this->sys_js_button[5].pressed && this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::RB_A);
    if (this->sys_js_button[5].pressed && this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::RB_B);
    if (this->sys_js_button[5].pressed && this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::RB_X);
    if (this->sys_js_button[5].pressed && this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if (this->sys_js_button[5].pressed && this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if (this->sys_js_button[5].pressed && this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if (this->sys_js_button[4].pressed && this->sys_js_button[5].on_press) this->control.SetGamepad(Input::Gamepad::LB_RB);

    float ly = -float(this->sys_js_axis[1]) / float(this->sys_js_max_value);
    float lx = -float(this->sys_js_axis[0]) / float(this->sys_js_max_value);
    float rx = -float(this->sys_js_axis[3]) / float(this->sys_js_max_value);

    bool has_input = (ly != 0.0f || lx != 0.0f || rx != 0.0f);

    if (has_input)
    {
        this->control.x = ly;
        this->control.y = lx;
        this->control.yaw = rx;
        this->sys_js_active = true;
    }
    else if (this->sys_js_active)
    {
        this->control.x = 0.0f;
        this->control.y = 0.0f;
        this->control.yaw = 0.0f;
        this->sys_js_active = false;
    }
}

void RL_Sim::RunModel()
{
    if (!(this->rl_init_done && simulation_running))
    {
        this->yaw_hold_target_initialized = false;
        this->was_rl_init_done_ = false;  // 状态未激活时重置标志，下次进入时能触发重采
        return;
    }

    // Sim-to-Real: 首次进入 RL 状态时重采延时（此时 config.yaml 已由 InitRL 加载完毕）
    if (!this->was_rl_init_done_)
    {
        this->was_rl_init_done_ = true;
        this->ResampleActionDelay();
        this->ClearActionDelayBuffer();
        // K 在 InitRL 加载 config.yaml 后才已知，这里再 reset 一次保证 history 容器与 K 对齐
        this->ClearActionsHistory();
    }

    this->episode_length_buf += 1;
    this->obs.ang_vel = this->robot_state.imu.gyroscope;
    // Sim-to-Real: IMU 陀螺仪白噪声 U(-a,a)，a 典型值 0.3 rad/s
    AddUniformNoiseInPlace(this->obs.ang_vel, this->params.Get<float>("noise_ang_vel", 0.0f));
    // base_quat 本身不加噪；projected_gravity 的噪声在 rl_sdk.cpp 投影后注入
    this->obs.base_quat = this->robot_state.imu.quaternion;

    float cmd_yaw_final = this->control.yaw;
    if (this->params.Get<bool>("yaw_hold_enable", false)) //当航向保持开关打开时，执行航向保持外环
    {   
        std::vector<float> euler = QuaternionToEuler(this->obs.base_quat); //将四元数转换为欧拉角
        float yaw_current = euler[2];   //读取欧拉角中的yaw角
        float wz = this->obs.ang_vel[2]; //读取角速度中的yaw角速度

        float kp = this->params.Get<float>("yaw_hold_kp", 0.5f); //读取比例增益
        float kd = this->params.Get<float>("yaw_hold_kd", 0.08f); //读取微分增益
        float correction_limit = this->params.Get<float>("yaw_hold_max_correction", 0.3f);
        float cmd_limit = this->params.Get<float>("yaw_hold_cmd_limit", 0.5f); //读取指令限幅
        float user_yaw_deadband = this->params.Get<float>("yaw_hold_user_cmd_deadband", 0.05f);
        float forward_cmd_min = this->params.Get<float>("yaw_hold_forward_cmd_min", 0.05f); //读取最小速度阈值

        bool user_turning = std::fabs(this->control.yaw) > user_yaw_deadband;//当控制指令大于死区时，认为用户正在转向
        bool forward_moving = std::fabs(this->control.x) > forward_cmd_min;//当控制指令大于最小速度时，认为机器人正在前进

        if (!this->yaw_hold_target_initialized || user_turning || !forward_moving)//当目标航向未初始化或用户正在转向或机器人不在前进时，初始化目标航向
        {
            this->yaw_hold_target = yaw_current;//将当前航向设置为目标航向
            this->yaw_hold_target_initialized = true;//将目标航向初始化标志设置为true   
        }

        if (!user_turning && forward_moving) //当用户不在转向且机器人正在前进时，计算外环补偿
        {
            float yaw_err = WrapToPi(this->yaw_hold_target - yaw_current); //计算航向误差
            float correction = kp * yaw_err - kd * wz; //计算外环补偿
            correction = clamp(correction, -correction_limit, correction_limit);
            cmd_yaw_final = clamp(this->control.yaw + correction, -cmd_limit, cmd_limit); //将补偿叠加到原始指令上并限幅
        }
    }
    else //当航向保持开关关闭时，重置目标航向
    {
        this->yaw_hold_target_initialized = false;  //将目标航向初始化标志设置为false
    }

    this->obs.commands = {this->control.x, this->control.y, cmd_yaw_final};
    this->obs.dof_pos = this->robot_state.motor_state.q;
    // Sim-to-Real: 关节编码器噪声，典型 a≈0.03 rad
    AddUniformNoiseInPlace(this->obs.dof_pos, this->params.Get<float>("noise_dof_pos", 0.0f));
    this->obs.dof_vel = this->robot_state.motor_state.dq;
    // Sim-to-Real: 关节速度差分后放大的高频噪声，典型 a≈1.25 rad/s
    AddUniformNoiseInPlace(this->obs.dof_vel, this->params.Get<float>("noise_dof_vel", 0.0f));

#ifdef RL_MUJOCO_TEST_CSV
        // CSV: base 角速度(体轴 rad/s)、IMU 四元数 [w,x,y,z]、速度指令 [vx,vy,yaw_rate]
        if (!this->csv_initialized)
        {
            std::filesystem::create_directories("log/mujoco_test");

            auto now = std::chrono::system_clock::now();
            std::time_t now_c = std::chrono::system_clock::to_time_t(now);
            std::tm tm_buf = *std::localtime(&now_c);
            std::ostringstream file_name_ss;
            file_name_ss << "log/mujoco_test/"
                         << this->robot_name << "_" << this->scene_name << "_"
                         << std::put_time(&tm_buf, "%Y%m%d_%H%M%S") << ".csv";

            this->test_csv_file.open(file_name_ss.str(), std::ios::out | std::ios::trunc);
            if (this->test_csv_file.is_open())
            {
                this->test_csv_file
                    << "ang_vel_x,ang_vel_y,ang_vel_z,"
                    << "base_quat_w,base_quat_x,base_quat_y,base_quat_z,"
                    << "cmd_x,cmd_y,cmd_yaw"
                    << std::endl;
            }
            this->csv_initialized = true;
        }

        if (this->test_csv_file.is_open())
        {
            this->test_csv_file << std::fixed << std::setprecision(4);
            this->test_csv_file
                << this->obs.ang_vel[0] << "," << this->obs.ang_vel[1] << "," << this->obs.ang_vel[2] << ","
                << this->obs.base_quat[0] << "," << this->obs.base_quat[1] << ","
                << this->obs.base_quat[2] << "," << this->obs.base_quat[3] << ","
                << this->obs.commands[0] << "," << this->obs.commands[1] << "," << this->obs.commands[2]
                << std::endl;

            if (this->episode_length_buf % 50 == 0)
            {
                this->test_csv_file.flush();
            }
        }
#endif

    // =================================================================
    // 🚨 裸观测数据探针 (Raw Observation Probe)
    // =================================================================
    if (this->episode_length_buf % 50 == 0)
    {
        std::cout << "\n================ [ 裸观测数据检查 (Raw Obs) ] ================" << std::endl;

            // 1. 角速度 (Angular Velocity)
            // 期望：静止站立时应极小，接近 0
            std::cout << "[1. 陀螺仪角速度] ang_vel (x, y, z) : "
                      << this->obs.ang_vel[0] << ", "
                      << this->obs.ang_vel[1] << ", "
                      << this->obs.ang_vel[2] << std::endl;

            // 2. 指令 (Commands)
            std::cout << "[2. 速度指令] commands (x, y, yaw) : "
                      << this->obs.commands[0] << ", "
                      << this->obs.commands[1] << ", "
                      << this->obs.commands[2] << std::endl;

            // 3. 姿态四元数 (Base Quaternion)
            // 注意：你之前代码里存的是 [w, x, y, z]
            std::cout << "[3. 机身四元数] base_quat (w,x,y,z) : "
                      << this->obs.base_quat[0] << ", "
                      << this->obs.base_quat[1] << ", "
                      << this->obs.base_quat[2] << ", "
                      << this->obs.base_quat[3] << std::endl;

            // 4. 重力投影 (Projected Gravity) —— 极其致命的一项！
            // 我们直接调用你原有的计算逻辑，提前看看结果
            std::vector<float> proj_gravity = QuatRotateInverse(this->obs.base_quat, this->obs.gravity_vec);
            std::cout << "[4. 🚩重力投影] proj_gravity (x,y,z): "
                      << proj_gravity[0] << ", "
                      << proj_gravity[1] << ", "
                      << proj_gravity[2] << std::endl;

            // 5. 关节位置 (DOF Position) 
            // 挑前 4 个关节打印看看量级即可
            std::cout << "[5. 关节绝对位置] dof_pos (前4个)   : ";
            for (int i = 0; i < std::min(4, (int)this->obs.dof_pos.size()); ++i) {
                std::cout << "Idx " << i << ":" << this->obs.dof_pos[i] << "  ";
            }
            std::cout << std::endl;

            // 6. 关节速度 (DOF Velocity)
            std::cout << "[6. 关节真实速度] dof_vel (前4个)   : ";
            for (int i = 0; i < std::min(4, (int)this->obs.dof_vel.size()); ++i) {
                std::cout << "Idx " << i << ":" << this->obs.dof_vel[i] << "  ";
            }
            std::cout << std::endl;
        std::cout << "==============================================================\n" << std::endl;
    }
    //rl控制
    this->obs.actions = this->Forward();

    // Sim-to-Real: 维护 K 帧动作历史 (newest-first)
    // 时机对齐 IsaacLab 默认 _apply_action 之前的 self.actions（即 clipped action），
    // 下一次 ComputeObservation 中 obs.actions_history[0] 即为本步刚产出的 a_t（即 a_{t-1} from next-step's view）
    {
        const int K = this->params.Get<int>("actions_history_length", 1);
        this->obs.actions_history.push_front(this->obs.actions);
        while (static_cast<int>(this->obs.actions_history.size()) > K)
        {
            this->obs.actions_history.pop_back();
        }
    }

    this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

    // ===== Sim-to-Real: Actuator Action Delay =====
    // 当 current_delay_ms_<=0（未启用）时退化为原有的直接 push 路径，保证行为向后兼容；
    // 否则把动作带着 release_time 放进延时 FIFO，由 RobotControl 200Hz 负责在到期时 flush。
    if (this->current_delay_ms_ <= 0.0f)
    {
        if (!this->output_dof_pos.empty()) output_dof_pos_queue.push(this->output_dof_pos);
        if (!this->output_dof_vel.empty()) output_dof_vel_queue.push(this->output_dof_vel);
        if (!this->output_dof_tau.empty()) output_dof_tau_queue.push(this->output_dof_tau);
    }
    else
    {
        DelayedAction d;
        d.release_time = std::chrono::steady_clock::now()
                       + std::chrono::microseconds(static_cast<int64_t>(this->current_delay_ms_ * 1000.0f));
        d.dof_pos = this->output_dof_pos;
        d.dof_vel = this->output_dof_vel;
        d.dof_tau = this->output_dof_tau;
        std::lock_guard<std::mutex> lk(this->action_delay_mutex_);
        this->action_delay_buffer_.push_back(std::move(d));
    }

        // this->TorqueProtect(this->output_dof_tau);
        // this->AttitudeProtect(this->robot_state.imu.quaternion, 75.0f, 75.0f);

    // 注意：trace 数据由独立线程 LogTick() 持续写入 trace_sim.csv，
    // 这里不再调用旧的 CSVLogger() —— 否则它会用默认精度 + 不同 schema
    // 把同一个 csv_filename 污染掉（与 LogTick 写入的 4 位定点格式不一致）。
}

std::vector<float> RL_Sim::Forward()
{
    std::unique_lock<std::mutex> lock(this->model_mutex, std::try_to_lock);

    // If model is being reinitialized, return previous actions to avoid blocking
    if (!lock.owns_lock())
    {
        std::cout << LOGGER::WARNING << "Model is being reinitialized, using previous actions" << std::endl;
        return this->obs.actions;
    }

    std::vector<float> clamped_obs = this->ComputeObservation();

    std::vector<float> actions;
    if (this->params.Get<std::vector<int>>("observations_history").size() != 0)
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

void RL_Sim::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
        this->plot_real_joint_pos[i].push_back(mj_data->sensordata[i]);
        // this->plot_target_joint_pos[i].push_back();  // TODO
        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.01);
}

// Signal handler for Ctrl+C
void signalHandler(int signum)
{
    std::cout << LOGGER::INFO << "Received signal " << signum << ", exiting..." << std::endl;
    if (RL_Sim::instance && RL_Sim::instance->sim)
    {
        RL_Sim::instance->sim->exitrequest.store(1);
    }
}
#ifdef CSV_LOGGER
void RL_Sim::LogTick()
{
    const float t_sec = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - this->log_t0).count();

    std::string fsm_state = "Unknown";
    if (this->fsm.current_state_)
    {
        fsm_state = this->fsm.current_state_->GetStateName();
    }

    std::vector<float> commands = this->obs.commands;
    if (commands.size() < 3)
    {
        commands = {this->control.x, this->control.y, this->control.yaw};
    }

    const std::vector<float>& cmd_q_live = this->robot_command.motor_command.q;

    this->CSVLoggerTrace(
        t_sec,
        fsm_state,
        commands,
        this->robot_state.imu.gyroscope,
        this->robot_state.imu.quaternion,
        this->robot_state.motor_state.q,
        cmd_q_live,
        this->robot_state.motor_state.dq,
        this->obs.actions
    );
}
#endif
int main(int argc, char **argv)
{
    signal(SIGINT, signalHandler);
    RL_Sim rl_sar(argc, argv);
    return 0;
}