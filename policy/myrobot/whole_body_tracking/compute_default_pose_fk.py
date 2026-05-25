"""
compute_default_pose_fk.py
------------------------------------------------------------
功能:
    离线计算 myrobot 在「dance config 中的 default_dof_pos」姿态下,
    每个 policy 关节所驱动的 body 相对 root(base_link)的 3D 位置,
    并输出成 yaml 供 rl_real_myrobot 在切到 whole_body_tracking 时
    作为 obs.key_body_pos_rel 的静态参考。

数学背景:
    obs.key_body_pos_rel 是笛卡尔空间观测(每 3 维 = body_pos - root_pos),
    与关节空间 obs.dof_pos 通过 forward kinematics 关联,无法从 dof_pos
    直接读出。实机没有 FK 库,所以这里离线烧一个常量,作为 dance 入口姿态
    的等价 obs。

运行方式:
    在 conda 环境(已装 mujoco-python)中执行:
        cd policy/myrobot/whole_body_tracking
        python compute_default_pose_fk.py
    会在当前目录生成 default_pose_static_ref.yaml,C++ 启动时加载即可。

注意:
    1. 这里把 base orientation 设为单位四元数 (1,0,0,0),与训练侧
       reset 时的标准 init 一致;
    2. 关节顺序按 dance config.yaml 的 joint_mapping(policy idx -> mujoco
       actuator id)scatter 到 qpos,确保和 rl_sim_mujoco.cpp 里
       MuJoCo->obs.key_body_pos_rel 的 layout 严格一致;
    3. 只跑一次,产物提交进仓库;以后 default_dof_pos 改了再重跑。
"""

import os
import sys
import numpy as np
import yaml

try:
    import mujoco
except ImportError:
    print("ERROR: mujoco python bindings not found. Activate your conda env first.")
    sys.exit(1)


# ---------------------------------------------------------------- params --
# 与 policy/myrobot/whole_body_tracking/config.yaml 保持一致
DEFAULT_DOF_POS = [
    -0.3, -0.3,  0.0,   # hip pitch L/R, waist pitch
     0.05, -0.05, 0.0,  # hip roll L/R, waist yaw
     0.004, -0.004, 0.0,# hip yaw L/R, head
     0.0,  0.0,         # shoulder pitch L/R
     0.5,  0.5,         # knee L/R
     0.0,  0.0,         # shoulder roll L/R
    -0.2, -0.2,         # ankle pitch L/R
     0.0,  0.0,         # elbow L/R
    -0.06, 0.06,        # ankle roll L/R
]

# policy idx -> mujoco actuator id (与 dance config.yaml 一致)
JOINT_MAPPING = [0, 6, 13, 1, 7, 12, 2, 8, 20, 14, 17, 3, 9, 15, 18, 4, 10, 16, 19, 5, 11]

# 训练时的初始质心高度(由用户指定)。这是单 episode reset 时 base_link 在世界系下的 z。
ROOT_Z_REF = 0.23

# MJCF 路径(脚本所在目录 -> 仓库根 -> rl_sar_zoo)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
MJCF_PATH = os.path.join(
    REPO_ROOT, "src", "rl_sar_zoo", "myrobot_description", "mjcf", "myrobot.xml"
)

OUT_PATH = os.path.join(SCRIPT_DIR, "default_pose_static_ref.yaml")
NUM_DOFS = 21


def main():
    if not os.path.exists(MJCF_PATH):
        raise FileNotFoundError(f"MJCF not found: {MJCF_PATH}")

    print(f"[FK] loading MJCF: {MJCF_PATH}")
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    # 1) base pose: 把 root 放到 (0,0,ROOT_Z_REF),朝向 = 单位四元数 ---------------
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = ROOT_Z_REF
    data.qpos[3] = 1.0  # qw
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = 0.0

    # 2) 按 joint_mapping 把 default_dof_pos scatter 到 mujoco qpos ----------------
    #    设置关节角度
    for i in range(NUM_DOFS):
        actuator_id = JOINT_MAPPING[i]
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        qpos_adr = int(model.jnt_qposadr[joint_id])
        if qpos_adr < 0 or qpos_adr >= model.nq:
            raise RuntimeError(f"invalid qpos_adr={qpos_adr} for policy joint {i}")
        data.qpos[qpos_adr] = DEFAULT_DOF_POS[i]

    # 3) 跑一次 forward kinematics,xpos 就是世界系下每个 body 的位置 ----------------
    #mj_forward 会从广义坐标 qpos 计算出每个 body 的 position, orientation
    mujoco.mj_forward(model, data)

    # 4) base_link 是 root body;取它的世界系位置作为减数 ----------------------------
    root_body_id = int(model.body("base_link").id)
    root_pos = data.xpos[root_body_id].copy()
    print(f"[FK] root (base_link) xpos = {root_pos}")
    print(f"[FK] note: root_z in MJCF default is 0.25 (spawn height); "
          f"this script forces it to ROOT_Z_REF={ROOT_Z_REF}")

    # 5) 与 rl_sim_mujoco.cpp 完全一致的提取逻辑:
    #    obs.key_body_pos_rel[i*3 + k] = xpos[body_of(joint_mapping[i])][k] - root_pos[k]
    #    base orientation = identity 时 world frame == root frame,无需旋转 ----------
    key_body_pos_rel = np.zeros((NUM_DOFS, 3), dtype=np.float64)
    joint_names_for_log = []
    for i in range(NUM_DOFS):
        actuator_id = JOINT_MAPPING[i]
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        body_id = int(model.jnt_bodyid[joint_id])
        body_pos = data.xpos[body_id]
        key_body_pos_rel[i] = body_pos - root_pos
        joint_names_for_log.append(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"j{joint_id}"
        )

    # 6) 打印一下 13 个 key body 的位置,方便人眼校对 --------------------------------
    key_body_joint_indices = [9, 10, 13, 14, 17, 18, 20, 19, 5, 7, 6, 12, 11]
    print("\n[FK] key body positions (policy idx, joint name, dx dy dz [m]):")
    for k in key_body_joint_indices:
        x, y, z = key_body_pos_rel[k]
        print(f"  idx={k:2d}  {joint_names_for_log[k]:>22s}  "
              f"({x:+.4f}, {y:+.4f}, {z:+.4f})")

    # 7) 输出 yaml -----------------------------------------------------------------
    out = {
        "myrobot/whole_body_tracking_static_ref": {
            "root_z_ref": float(ROOT_Z_REF),
            # flatten 成 63 个 float,layout = [j0_x, j0_y, j0_z, j1_x, ...]
            # 这与 rl_sim_mujoco.cpp 中 obs.key_body_pos_rel 的写入顺序一致
            "key_body_pos_rel_flat": [float(v) for v in key_body_pos_rel.flatten()],
            "num_of_dofs": NUM_DOFS,
            "comment": (
                "Static FK reference for default_dof_pos (dance entry pose). "
                "Generated by compute_default_pose_fk.py. "
                "Used as a constant placeholder when real robot has no online FK."
            ),
        }
    }
    with open(OUT_PATH, "w") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False)
    print(f"\n[FK] wrote -> {OUT_PATH}")


if __name__ == "__main__":
    main()
