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

    // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
    if (rosetta_error_msg)
    {
        DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
        std::exit(1);
    }
#endif

    // print version, check compatibility
    std::cout << LOGGER::INFO << "[MuJoCo] Version: " << mj_versionString() << std::endl;
    if (mjVERSION_HEADER != mj_version())
    {
        mju_error("Headers and library have different versions");
    }

    // scan for libraries in the plugin directory to load additional plugins
    scanPluginLibraries();

    mjvCamera cam;
    mjv_defaultCamera(&cam);

    mjvOption opt;
    mjv_defaultOption(&opt);

    mjvPerturb pert;
    mjv_defaultPerturb(&pert);

    // simulate object encapsulates the UI
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
    this->CSVInit(this->robot_name);
#endif

    std::cout << LOGGER::INFO << "RL_Sim start" << std::endl;

    // start simulation UI loop (blocking call)
    sim->RenderLoop();
}

RL_Sim::~RL_Sim()
{
    // Clear static instance pointer
    instance = nullptr;

    this->loop_keyboard->shutdown();
    this->loop_joystick->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;
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

    if (this->control.current_keyboard == Input::Keyboard::R || this->control.current_gamepad == Input::Gamepad::RB_Y)
    {
        if (this->mj_model && this->mj_data)
        {
            mj_resetData(this->mj_model, this->mj_data);
            mj_forward(this->mj_model, this->mj_data);
        }
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
        return;
    }

    this->episode_length_buf += 1;
    this->obs.ang_vel = this->robot_state.imu.gyroscope;
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
    this->obs.dof_vel = this->robot_state.motor_state.dq;

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

    this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);
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

        // this->TorqueProtect(this->output_dof_tau);
        // this->AttitudeProtect(this->robot_state.imu.quaternion, 75.0f, 75.0f);

#ifdef CSV_LOGGER
    // CSVLogger 内对力矩列未写入；与 rl_sdk 一致，不传力矩估计与指令力矩。
    this->CSVLogger({}, {}, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
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

int main(int argc, char **argv)
{
    signal(SIGINT, signalHandler);
    RL_Sim rl_sar(argc, argv);
    return 0;
}