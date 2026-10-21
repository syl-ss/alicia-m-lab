"""在同一MuJoCo模型中求末端tool0的位姿与有界数值IK；没有串口连接。"""
# FK：六个关节角q -> 末端位置/朝向；IK：末端目标位置/朝向 -> 六个关节角q。
# 先掌握solve_ik的输入与返回字典即可；最小二乘、旋转矩阵和四元数细节可后学。
# 本文件只在临时data中计算姿态/检查采样点，不执行机器人运动。
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def pose(sim, q, data=None):
    # q为六关节角，单位rad；返回三维位置(米)和3×3旋转矩阵。
    # 没传data就新建临时数据；传了则复用以降低大量IK试算的开销。
    data = mujoco.MjData(sim.model) if data is None else data
    data.qpos[:] = sim.data.qpos
    data.qpos[sim.arm_qpos] = q
    # 从实际场景复制状态，再仅替换临时数据中的机械臂关节角。
    mujoco.mj_forward(sim.model,data)
    # forward根据姿态计算各物体/参考点的位置，不是让仿真时间前进一步。
    return data.site_xpos[sim.tool_site].copy(), data.site_xmat[sim.tool_site].reshape(3,3).copy()


def solve_ik(sim, target_xyz, target_rotation, seed=None):
    # target_xyz：目标位置(3,)；target_rotation：目标朝向(3,3)。
    # seed是求解的起始猜测；最好给上一时刻关节角，倾向于找附近的解。
    xyz = np.asarray(target_xyz,dtype=float)
    rotation = np.asarray(target_rotation,dtype=float)
    if xyz.shape != (3,) or rotation.shape != (3,3) or not np.isfinite(xyz).all() or not np.isfinite(rotation).all():
        raise ValueError('目标位置必须是3个有限数，旋转必须是3x3有限矩阵')
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(rotation),1):
        # .T表示矩阵转置，@表示矩阵乘法；这里验证输入确实是合法旋转矩阵。
        raise ValueError('目标旋转矩阵必须正交且行列式为1')
    scratch = mujoco.MjData(sim.model)
    def residual(q):
        # 内部小函数：给一组候选关节角，返回“离目标还差多少”的六个误差分量。
        p,r = pose(sim,q,scratch)
        # np.r_拼接位置误差和旋转误差；位置权重乘5以调整优化时的相对权重。
        # as_rotvec把相对旋转转成三维旋转向量，其长度是旋转角(rad)。
        return np.r_[5*(p-xyz), Rotation.from_matrix(rotation@r.T).as_rotvec()]
    low,high = sim.joint_limits.T
    # low/high分别是六关节下限/上限；多个起始猜测降低单次局部求解失败的影响。
    seeds = [sim.get_joint_positions() if seed is None else np.asarray(seed),np.deg2rad([0,-100,-90,0,-45,0]),
             np.deg2rad([15,-120,-100,0,-60,15])]
    best = None
    for seed in seeds:
        # least_squares反复调整q让误差尽量小；bounds禁止解超关节范围。
        # clip把初始猜测限制在上下限内；max_nfev是求解计算预算，不是运动步数。
        result = least_squares(residual,np.clip(seed,low+1e-6,high-1e-6),
                               bounds=(low+1e-8,high-1e-8),max_nfev=350)
        p,r = pose(sim,result.x,scratch)
        position_error = float(np.linalg.norm(p-xyz))
        orientation_error = float(np.linalg.norm(Rotation.from_matrix(rotation@r.T).as_rotvec()))
        candidate = {'q':result.x,'position_error_m':position_error,
                     'orientation_error_rad':orientation_error,
                     'success':position_error<=0.0002 and orientation_error<=0.002}
        # q存最终关节角；只有位置误差<=0.2mm且朝向误差<=0.002rad才记为成功。
        # 求解器结束不等于满足本课精度，也不代表之后的整条路径无碰撞。
        if best is None or np.linalg.norm(residual(result.x)) < np.linalg.norm(residual(best['q'])):
            best = candidate
        if candidate['success']:
            return candidate
    return best
    # 所有猜测都没达标时返回误差较小的候选；调用者必须检查其success。


def check_sampled_path(sim, positions, carry_cube=False):
    """检查采样关节姿态；实际执行期间还会记录接触，非连续碰撞证明。"""
    scratch = mujoco.MjData(sim.model)
    # scratch是检查专用副本；positions每行是一组关节角，逐行检查采样姿态。
    if carry_cube:
        # 只在临时检查数据里预测被夹物体随手运动；不改实际仿真物体状态。
        old_xyz,old_rotation=pose(sim,sim.get_joint_positions())
        cube_joint=mujoco.mj_name2id(sim.model,mujoco.mjtObj.mjOBJ_JOINT,'cube_free')
        cube_adr=int(sim.model.jnt_qposadr[cube_joint])
        cube_q=sim.data.qpos[cube_adr:cube_adr+7].copy()
        relative_xyz=old_rotation.T@(cube_q[:3]-old_xyz)
        # 世界坐标偏移乘工具旋转矩阵的转置，换成“相对于夹爪”的局部偏移。
        # 预检假设方块随手运动，仅用于估计潜在接触，实际滑移仍由物理执行决定。
        cube_rotation=Rotation.from_quat(np.r_[cube_q[4:7],cube_q[3]]).as_matrix()
        # MuJoCo四元数顺序[w,x,y,z]，SciPy用[x,y,z,w]，所以这里必须换序。
    for index,q in enumerate(positions):
        xyz,rotation=pose(sim,q,scratch)
        if carry_cube:
            scratch.qpos[cube_adr:cube_adr+3]=xyz+rotation@relative_xyz
            # 将相对工具偏移转换回世界坐标，仅写到scratch，不移动真实方块。
            quat=Rotation.from_matrix(rotation@old_rotation.T@cube_rotation).as_quat()
            scratch.qpos[cube_adr+3:cube_adr+7]=np.r_[quat[3],quat[:3]]
            # 将SciPy四元数换回MuJoCo的[w,x,y,z]顺序，再计算此检查姿态的接触。
            mujoco.mj_forward(sim.model,scratch)
        hits = sim.robot_penetrations(scratch)
        if hits:
            raise ValueError(f'参考轨迹采样点{index}出现机器人穿透接触：{hits}')
