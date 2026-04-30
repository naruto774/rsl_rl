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
#include <chrono>
#include <ctime>

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
        state->imu.gyroscope[i] = state_ang_vel[i];
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

    this->CSVLoggerTrace(
        t_sec,
        fsm_state,
        commands,
        this->robot_state.imu.gyroscope,
        this->robot_state.imu.quaternion,
        this->robot_state.motor_state.q,
        cmd_q_live,
        this->robot_state.motor_state.dq,
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
