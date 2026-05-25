/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 *
 * kinematics_fk.cpp
 * ------------------------------------------------------------
 * Online forward kinematics helper for rl_sdk (MuJoCo backend).
 *
 * 数学要点：
 *   1. policy 第 i 个关节通过 joint_mapping[i] = actuator_id 找到对应的
 *      mujoco joint，再通过 jnt_bodyid 找到该 joint 驱动的 body。
 *      这条链与 rl_sim_mujoco.cpp 仿真侧 obs.key_body_pos_rel 的写入逻辑
 *      （actuator_trnid → jnt_bodyid → xpos）严格一致。
 *   2. 控制周期里我们只需要 xpos（forward kinematics 输出），不需要力 / 接触 /
 *      接触动力学，所以用 mj_kinematics 而不是 mj_forward —— 前者只跑运动学
 *      部分，21-DoF 实测 ~10-50 μs。
 *   3. root 平移用 (0,0,0)：因为 key_body_pos_rel = xpos[body] - xpos[root]，
 *      平移分量在做差时精确抵消；旋转分量则必须用真实 IMU quat，否则
 *      key_body_pos_rel 在 body frame 而 ref_body_quat_tan_norm 在 world frame，
 *      二者脱钩，policy 看到的是训练分布外的几何状态。
 */
#include "kinematics_fk.hpp"

#include <iostream>

#ifdef USE_MUJOCO
#include <mujoco/mujoco.h>
#endif

KinematicsFK::KinematicsFK() = default;

KinematicsFK::~KinematicsFK()
{
#ifdef USE_MUJOCO
    if (data_)  mj_deleteData(data_);
    if (model_) mj_deleteModel(model_);
#endif
}

bool KinematicsFK::IsLoaded() const
{
#ifdef USE_MUJOCO
    return model_ != nullptr && data_ != nullptr;
#else
    return false;
#endif
}

bool KinematicsFK::Load(const std::string& mjcf_path,
                       const std::vector<int>& joint_mapping,
                       const std::string& root_body_name)
{
#ifndef USE_MUJOCO
    (void)mjcf_path; (void)joint_mapping; (void)root_body_name;
    std::cout << "[KinematicsFK] built without USE_MUJOCO -> online FK disabled" << std::endl;
    return false;
#else
    // 清掉前一次状态，允许重复调用
    if (data_)  { mj_deleteData(data_);  data_  = nullptr; }
    if (model_) { mj_deleteModel(model_); model_ = nullptr; }
    body_id_per_policy_idx_.clear();
    qpos_adr_per_policy_idx_.clear();
    root_body_id_ = -1;
    num_dofs_ = 0;
    has_freejoint_ = false;

    char err[1024] = {0};
    model_ = mj_loadXML(mjcf_path.c_str(), nullptr, err, sizeof(err));
    if (!model_)
    {
        std::cout << "[KinematicsFK] failed to load MJCF: " << mjcf_path
                  << " : " << err << std::endl;
        return false;
    }
    data_ = mj_makeData(model_);
    if (!data_)
    {
        std::cout << "[KinematicsFK] mj_makeData failed" << std::endl;
        mj_deleteModel(model_); model_ = nullptr;
        return false;
    }

    // root body
    root_body_id_ = mj_name2id(model_, mjOBJ_BODY, root_body_name.c_str());
    if (root_body_id_ < 0 || root_body_id_ >= model_->nbody)
    {
        std::cout << "[KinematicsFK] root body '" << root_body_name
                  << "' not found in MJCF" << std::endl;
        mj_deleteData(data_);   data_   = nullptr;
        mj_deleteModel(model_); model_  = nullptr;
        return false;
    }

    // 解析 policy_idx -> body / qpos 映射
    num_dofs_ = static_cast<int>(joint_mapping.size());
    body_id_per_policy_idx_.assign(num_dofs_, -1);
    qpos_adr_per_policy_idx_.assign(num_dofs_, -1);
    for (int i = 0; i < num_dofs_; ++i)
    {
        const int actuator_id = joint_mapping[i];
        if (actuator_id < 0 || actuator_id >= model_->nu)
        {
            std::cout << "[KinematicsFK] joint_mapping[" << i << "]=" << actuator_id
                      << " out of range (nu=" << model_->nu << ")" << std::endl;
            continue;
        }
        const int joint_id = model_->actuator_trnid[actuator_id * 2 + 0];
        if (joint_id < 0 || joint_id >= model_->njnt) continue;
        const int body_id = model_->jnt_bodyid[joint_id];
        if (body_id < 0 || body_id >= model_->nbody) continue;
        const int qpos_adr = model_->jnt_qposadr[joint_id];
        if (qpos_adr < 0 || qpos_adr >= model_->nq) continue;
        body_id_per_policy_idx_[i] = body_id;
        qpos_adr_per_policy_idx_[i] = qpos_adr;
    }

    // 检测是否有 freejoint（决定 base pos / quat 是否要塞 qpos[0:7]）
    for (int j = 0; j < model_->njnt; ++j)
    {
        if (model_->jnt_type[j] == mjJNT_FREE)
        {
            has_freejoint_ = true;
            break;
        }
    }

    std::cout << "[KinematicsFK] loaded " << mjcf_path
              << "  (root=" << root_body_name << ", id=" << root_body_id_
              << ", num_dofs=" << num_dofs_
              << ", has_freejoint=" << (has_freejoint_ ? "yes" : "no") << ")" << std::endl;
    return true;
#endif  // USE_MUJOCO
}

bool KinematicsFK::ComputeKeyBodyPosRel(const std::vector<float>& dof_pos,
                                       const std::vector<float>& base_quat,
                                       std::vector<float>& key_body_pos_rel)
{
#ifndef USE_MUJOCO
    (void)dof_pos; (void)base_quat;
    key_body_pos_rel.clear();
    return false;
#else
    if (!IsLoaded()) return false;
    if (static_cast<int>(dof_pos.size()) != num_dofs_)
    {
        std::cout << "[KinematicsFK] dof_pos size mismatch: got=" << dof_pos.size()
                  << " expected=" << num_dofs_ << std::endl;
        return false;
    }

    // 1) base pose 写入 qpos: 平移=0，旋转=真实 IMU quat
    if (has_freejoint_ && model_->nq >= 7)
    {
        data_->qpos[0] = 0.0;
        data_->qpos[1] = 0.0;
        data_->qpos[2] = 0.0;
        // base_quat 我们约定 [w,x,y,z]，与 obs.base_quat / mujoco freejoint qpos 顺序一致
        if (base_quat.size() == 4)
        {
            data_->qpos[3] = static_cast<mjtNum>(base_quat[0]);
            data_->qpos[4] = static_cast<mjtNum>(base_quat[1]);
            data_->qpos[5] = static_cast<mjtNum>(base_quat[2]);
            data_->qpos[6] = static_cast<mjtNum>(base_quat[3]);
        }
        else
        {
            data_->qpos[3] = 1.0; data_->qpos[4] = 0.0;
            data_->qpos[5] = 0.0; data_->qpos[6] = 0.0;
        }
    }

    // 2) scatter dof_pos to qpos
    for (int i = 0; i < num_dofs_; ++i)
    {
        const int adr = qpos_adr_per_policy_idx_[i];
        if (adr >= 0) data_->qpos[adr] = static_cast<mjtNum>(dof_pos[i]);
    }

    // 3) 只跑运动学（mj_kinematics << mj_forward）
    mj_kinematics(model_, data_);

    // 4) xpos[body] - xpos[root]，layout 与仿真侧严格一致
    key_body_pos_rel.assign(num_dofs_ * 3, 0.0f);
    const mjtNum* rp = &data_->xpos[root_body_id_ * 3];
    for (int i = 0; i < num_dofs_; ++i)
    {
        const int bid = body_id_per_policy_idx_[i];
        if (bid < 0) continue;
        const mjtNum* bp = &data_->xpos[bid * 3];
        const int b = i * 3;
        key_body_pos_rel[b + 0] = static_cast<float>(bp[0] - rp[0]);
        key_body_pos_rel[b + 1] = static_cast<float>(bp[1] - rp[1]);
        key_body_pos_rel[b + 2] = static_cast<float>(bp[2] - rp[2]);
    }
    return true;
#endif  // USE_MUJOCO
}
