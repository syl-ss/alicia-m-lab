"""第一课：已知方块XYZ，让模型末端tool0到达方块上方。"""
# 本课只到方块上方，不闭爪抓取。
# 数据流：目标XYZ/朝向 -> IK关节角 -> 五次关节轨迹 -> 仿真反馈 -> 误差验收。
# np是numpy的常用简称，负责数组计算；Rotation用于描述和比较朝向。
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scene import DesktopSim, CUBE_XYZ, HOVER_XYZ, DOWN_ROTATION, save_scene_image
from kinematics import solve_ik, pose, check_sampled_path
from simulation.alicia_m import quintic_trajectory
from simulation.run_trajectory import execute

MOVE_TIME_S = 8.0
# 单位秒；内部关节角用弧度rad，np.rad2deg只用于转成便于阅读的角度deg。


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless',action='store_true')
    args = parser.parse_args()
    sim = DesktopSim()
    output = Path(__file__).resolve().parent/'results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=True)

    # 1. 物体XYZ和夹爪目标位姿：同一世界坐标系，单位米。
    print('Cube center XYZ (m):',CUBE_XYZ,flush=True)
    print('Target tool0 XYZ (m):',HOVER_XYZ,flush=True)
    # 2. IK把目标位置与朝向，转换成六个关节角。
    solution = solve_ik(sim,HOVER_XYZ,DOWN_ROTATION)
    # solution是字典：success表示误差是否达标，q是六个关节角组成的数组。
    # not把真假反过来：success=False时not success=True，因此进入异常分支。
    if not solution['success']:
        raise RuntimeError(f'IK未满足位置和姿态要求：{solution}')
    print('IK joint angles (deg):',np.round(np.rad2deg(solution['q']),3),flush=True)
    # 3. 从准备姿态到IK解，生成平滑关节轨迹；末端路径不保证为直线。
    times,positions = quintic_trajectory(sim.get_joint_positions(),solution['q'],
                                        duration=MOVE_TIME_S,timestep=sim.model.opt.timestep)
    # times形状为(N,)，positions为(N,6)：每个时刻对应六个关节的目标角度。
    # [::20]表示每隔20行取一个姿态；[-1:]只取最后一行，确保终点也被检查。
    check_sampled_path(sim,positions[::20])
    check_sampled_path(sim,positions[-1:])
    save_scene_image(sim,output/'scene_start.png')
    context = None
    if not args.headless:
        from mujoco import viewer as mj_viewer
        context = mj_viewer.launch_passive(sim.model,sim.data)
    try:
        if context is not None:
            with context.lock():
                context.cam.lookat[:] = [0.22,0,0.20]
                context.cam.distance = 0.95
                context.cam.azimuth = 135
                context.cam.elevation = -30
        # 4. 位置伺服驱动模型，记录实际反馈；只在全程开始时初始化一次。
        summary = execute(sim,times,positions,output,viewer=context,settle=1.0)
        # execute沿轨迹给目标并记录反馈；settle是在轨迹结束后再等待1秒。
        # 5. 用实际关节状态做FK，核对模型末端的位置与朝向。
        actual_xyz,actual_rotation = pose(sim,sim.get_joint_positions())
        position_error = float(np.linalg.norm(actual_xyz-HOVER_XYZ))
        # norm计算三维距离，单位米；朝向误差是相对旋转的角度，单位弧度。
        orientation_error = float(np.linalg.norm(Rotation.from_matrix(DOWN_ROTATION@actual_rotation.T).as_rotvec()))
        report = {'scope':'Simulation-only hover; grasping has not been attempted',
                  'frame':'MuJoCo world, robot fixed base at origin','tool_reference':'tool0_site',
                  'cube_initial_center_m':CUBE_XYZ.tolist(),'target_tool_xyz_m':HOVER_XYZ.tolist(),
                  'actual_tool_xyz_m':actual_xyz.tolist(),'position_error_m':position_error,
                  'orientation_error_rad':orientation_error,'goal_joint_deg':np.rad2deg(solution['q']).tolist(),
                  'completed':summary['completed'],'warning_count':summary['warning_count'],
                  'robot_penetrations':sim.contact_violations,
                  'cube_final_xyz_m':sim.data.xpos[sim.cube_body].tolist()}
        # and要求所有条件同时满足：轨迹完成、无警告/异常接触，
        # 且实际位置误差小于2毫米、朝向误差小于0.02弧度（约1.15度）。
        report['hover_passed'] = bool(summary['completed'] and summary['warning_count']==0
                                     and not sim.contact_violations and position_error<0.002 and orientation_error<0.02)
        # json.dumps把字典转成JSON文字；indent=2让报告按缩进排版，方便打开查看。
        (output/'hover_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        (output/'scene.xml').write_text(sim.xml,encoding='utf-8')
        save_scene_image(sim,output/'scene_end.png')
        print(json.dumps(report,indent=2),flush=True)
        print(f'Results: {output}',flush=True)
        if context is not None and context.is_running():
            print('Holding above cube; close MuJoCo window to finish.',flush=True)
            while context.is_running():
                import time
                started = time.perf_counter()
                with context.lock():
                    sim.step(10)
                context.sync()
                time.sleep(max(0,0.02-(time.perf_counter()-started)))
    finally:
        # 不论正常结束还是中途异常，都尝试关闭查看器。
        if context is not None:
            context.close()
    return 0 if report['hover_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
