/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 *
 * kinematics_fk.hpp
 * ------------------------------------------------------------
 * Online forward kinematics helper for rl_sdk.
 *
 * Why this file exists:
 *   obs.key_body_pos_rel 是笛卡尔空间观测（每个关节体相对 root 的 3D 位置），
 *   它与 obs.dof_pos 通过 robot 的运动学树相关联：
 *       p_i = T_root · ∏_{j ∈ parent_chain(i)} T_j(q_j)
 *       p_i^rel = p_i - p_root
 *   实机部署时没有 mj_data->xpos 这样的"上帝视角"，必须在线 FK 自己算。
 *
 *   仿真侧 rl_sim_mujoco.cpp 直接复用了仿真本身的 mj_data；本 helper 给
 *   rl_real_myrobot 用同一份 MJCF 跑一个 *headless* MuJoCo 实例做纯 FK，
 *   保证训练-部署侧拿到的 key_body_pos_rel 几何上严格一致（同一份 MJCF +
 *   同一套 actuator→joint→body 解析链）。
 *
 * 编译策略：
 *   - USE_MUJOCO 打开：链接 libmujoco，全功能；
 *   - USE_MUJOCO 关闭：所有方法是空实现，IsLoaded() 永远返回 false，
 *     调用方退化到现有的静态参考方案，不影响其他真机 backend。
 *
 * 性能：21-DoF 一次 mj_kinematics ~10-50 μs，60Hz 控制频率下 CPU 占用 < 0.3%。
 *
 * 线程安全：单实例非线程安全；调用方需保证 FK loop 与训练时一致的串行顺序。
 *           rl_real_myrobot 的 RunModel 是单线程定时器，天然满足。
 */
#ifndef KINEMATICS_FK_HPP
#define KINEMATICS_FK_HPP

#include <string>
#include <vector>

// Forward-declare MuJoCo types so users of rl_sdk don't need to include
// <mujoco/mujoco.h>. Real definitions live inside the .cpp.
struct mjModel_;
struct mjData_;

class KinematicsFK
{
public:
    KinematicsFK();
    ~KinematicsFK();

    // No copy / no move (持有原始资源指针)
    KinematicsFK(const KinematicsFK&) = delete;
    KinematicsFK& operator=(const KinematicsFK&) = delete;

    /**
     * 加载 MJCF 并建立 policy_idx -> mujoco body / qpos addr 的缓存映射。
     *
     * @param mjcf_path        MJCF (xml) 路径。
     * @param joint_mapping    长度 = num_dofs；joint_mapping[i] = policy idx i
     *                         对应的 mujoco actuator id（与
     *                         policy/<robot>/<config>/config.yaml::joint_mapping
     *                         保持一致）。
     * @param root_body_name   根 body 名（默认 "base_link"）。
     * @return true 加载成功；false 即视为不可用，所有 Compute* 都会返回 false。
     */
    bool Load(const std::string& mjcf_path,
              const std::vector<int>& joint_mapping,
              const std::string& root_body_name = "base_link");

    bool IsLoaded() const;

    /**
     * 给定关节角 + base 四元数，跑一次 mj_kinematics 算 key_body_pos_rel。
     *
     *   输入：
     *     dof_pos   长度 num_dofs，policy 顺序，[rad]；
     *     base_quat 长度 4，[w, x, y, z]。
     *               * 平移分量我们故意填 (0,0,0)：因为最终 key_body_pos_rel
     *                 = xpos[body] - xpos[root]，root 平移分量精确抵消；
     *               * 旋转分量影响整棵树的方向，要传入真实 IMU quat，
     *                 否则与 ref_body_quat_tan_norm 通道脱钩、导致 OOD。
     *   输出：
     *     key_body_pos_rel 写入长度 num_dofs * 3：[j0_x, j0_y, j0_z, j1_x, ...]，
     *     与 rl_sim_mujoco.cpp 中 obs.key_body_pos_rel 的 layout 严格一致。
     *
     * @return true 成功；false 表示未加载、维度不匹配或 MuJoCo 不可用。
     */
    bool ComputeKeyBodyPosRel(const std::vector<float>& dof_pos,
                              const std::vector<float>& base_quat,
                              std::vector<float>& key_body_pos_rel);

    int NumDofs() const { return num_dofs_; }

private:
    mjModel_* model_ = nullptr;
    mjData_*  data_  = nullptr;

    // 缓存：避免每次 FK 都做字符串/索引解析
    std::vector<int> body_id_per_policy_idx_;
    std::vector<int> qpos_adr_per_policy_idx_;
    int root_body_id_ = -1;
    int num_dofs_ = 0;
    bool has_freejoint_ = false;  // 决定 qpos[3:7] 是否塞 base_quat
};

#endif  // KINEMATICS_FK_HPP
