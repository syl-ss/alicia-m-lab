"""完整接触抓放的固定执行与记录程序；不绑定或重置搬运中的物体。"""
# 这是课程共用的执行程序，step02～step05都调用它。
# 推荐读法：先找到run()看动作顺序，再读verify_grasp_or_retreat()看失败分支；
# 最后读move_to() -> move_path() -> step()，了解目标怎样变成实际运动。
# 数据流：方块中心XYZ -> tool0目标XYZ -> IK关节角q -> 仿真执行 -> 实际状态 -> 报告。
# 单位约定：位置/开口用米m，时间用秒s，关节角用弧度rad，接触力用牛顿N。
from contextlib import nullcontext
import csv
import json
import time
import traceback

import mujoco
import numpy as np
import matplotlib
matplotlib.use('Agg')
# Agg让图表直接保存为图片，不需要弹出matplotlib绘图窗口。
import matplotlib.pyplot as plt

from scene import DesktopSim, TABLE_TOP_Z, CUBE_HALF_SIZE, DOWN_ROTATION, save_scene_image
from kinematics import pose, solve_ik, check_sampled_path
from simulation.alicia_m import quintic_trajectory

GRASP_TOOL_OFFSET_Z = -0.020  # tool0比方块中心低20mm；指尖仍比桌面高约3.5mm。
CARRY_TOOL_Z = 0.15
# 搬运时tool0的世界坐标高度：150毫米；不是方块中心的高度。
OPENING_M = 0.07
# 张爪目标开口70毫米；下方抓取开口默认26毫米，方块边长30毫米。
SPEED_PROFILES = {'normal': 1.0, 'fast': 0.5}
# 字典把名称映射为机械臂运动时间倍率；fast的数值0.5表示时间减半。
MIN_CONTACT_N = 0.02  # 当前MuJoCo教学场景阈值，不是实机传感器标定值。


class GraspNotConfirmed(RuntimeError):
    """可预期的任务失败，区别于IK、碰撞等执行异常。"""


class PickPlaceRunner:
    # 类把“一次任务的数据”和“操作这些数据的方法”放在一起。
    # self指当前任务对象；self.xxx可以被这个对象的其他方法继续读取。
    def __init__(self, output, headless=False, speed='normal', pick_xyz=None, place_xyz=None,
                 grasp_opening_m=.026):
        # __init__在创建对象时执行；isfinite排除NaN/无穷大，后半句检查数值范围。
        if not np.isfinite(grasp_opening_m) or not 0 <= grasp_opening_m <= OPENING_M:
            raise ValueError('抓取开口必须在0～0.07米之间')
        self.grasp_opening_m = grasp_opening_m
        self.grasp_check = None
        self.grasp_recovery = 'not_needed'
        self.arm_time_scale = SPEED_PROFILES[speed]
        self.speed_profile = speed
        self.output = output
        output.mkdir(parents=True,exist_ok=True)
        self.sim = DesktopSim(pick_xyz=pick_xyz,place_xyz=place_xyz)
        # sim是完整物理场景；copy()复制数组，避免两个变量意外共用同一份坐标。
        self.pick_xyz = self.sim.pick_xyz.copy()
        self.place_xyz = self.sim.place_xyz.copy()
        self.sim.set_gripper(OPENING_M)
        self.headless = headless
        self.viewer = None
        self.rows = []
        self.events = []
        # rows逐步保存连续反馈；events只保存接近、闭爪、抬升等关键时刻。
        self.phase = 'INITIALIZE'
        self.gripper_target = OPENING_M
        self.q_hold = self.sim.get_joint_positions()
        # q_hold是等待/开合爪时保持的六个关节目标；q通常表示关节角数组。
        self.max_cube_z = self.pick_xyz[2]
        self.cube_joint = mujoco.mj_name2id(self.sim.model,mujoco.mjtObj.mjOBJ_JOINT,'cube_free')
        # 名称先转换成模型编号，再找方块在大数组中的起始位置。
        # qpos保存位置/姿态；qvel保存速度；自由方块分别占7项和6项。
        self.cube_qpos = int(self.sim.model.jnt_qposadr[self.cube_joint])
        self.cube_dof = int(self.sim.model.jnt_dofadr[self.cube_joint])
        self.started = None

    def cube_xyz(self):
        # [起点:终点]取出起点及之后两项，即方块中心X/Y/Z；终点索引不包含。
        return self.sim.data.qpos[self.cube_qpos:self.cube_qpos+3].copy()

    def step(self, q_ref, v_ref=None):
        # 最小执行单元：给一次关节/夹爪目标 -> 前进一个仿真步 -> 读取并记录反馈。
        # q_ref为六关节参考角；v_ref为参考角速度(rad/s)，None表示没有提供。
        sim = self.sim
        if self.viewer is not None and not self.viewer.is_running():
            raise RuntimeError('Viewer closed before task completion')
        command = np.asarray(q_ref).copy()
        if v_ref is not None:
            # 模型位置伺服含速度阻尼；给目标角加kv/kp*v_ref作为速度前馈补偿。
            # command是补偿后的控制输入；日志里的q_ref仍是原参考轨迹。
            kp=sim.model.actuator_gainprm[sim.arm_actuators,0]
            kv=-sim.model.actuator_biasprm[sim.arm_actuators,2]
            command += kv/kp*v_ref
        with self.viewer.lock() if self.viewer is not None else nullcontext():
            # 有窗口时锁住共享仿真数据；无窗口时nullcontext让同一段代码正常执行。
            sim.set_joint_target(command)
            sim.set_gripper(self.gripper_target)
            sim.step()
            measured=sim.get_joint_state()
            cube=self.cube_xyz()
            forces=sim.finger_cube_forces()
            opening=sim.gripper_opening()
        self.max_cube_z=max(self.max_cube_z,float(cube[2]))
        self.rows.append([measured['time'],self.phase,*q_ref,*measured['positions'],
                          *measured['velocities'],self.gripper_target,opening,*cube,
                          forces['link7'],forces['link8'],*measured['torques']])
        # 上面的*把数组展开为多列；例如*q_ref展开成六列关节目标。
        # 这一行的列顺序必须与save()里的columns对应，才能读懂CSV。
        if sim.contact_violations:
            raise RuntimeError(f'Unexpected robot penetration: {sim.contact_violations[-1]}')
        if self.viewer is not None:
            if len(self.rows)%8==0:
                self.viewer.sync()
            delay=sim.data.time-(time.perf_counter()-self.started)
            # 有窗口时等待墙上时钟追上仿真时间，便于观察；无窗口时不做此等待。
            if delay>0:
                time.sleep(delay)

    def hold(self, seconds):
        # 等待不是停止计算：仍逐步保持q_hold并采集反馈，让接触和物体自然稳定。
        self.started=time.perf_counter()-self.sim.data.time
        for _ in range(round(seconds/self.sim.model.opt.timestep)):
            self.step(self.q_hold)

    def change_gripper(self, opening, seconds=.6):
        # 从当前目标开口平滑过渡到opening，机械臂保持q_hold不变。
        self.started=time.perf_counter()-self.sim.data.time
        start=self.gripper_target
        count=round(seconds/self.sim.model.opt.timestep)
        for index in range(1,count+1):
            u=index/count
            blend=10*u**3-15*u**4+6*u**5
            # u是0到1的进度；五次多项式让开口目标在两端速度/加速度为零。
            # 它是五次平滑插值，不是lesson02里的七段恒jerk S曲线。
            self.gripper_target=start+(opening-start)*blend
            self.step(self.q_hold)
        self.hold(.4)

    def checkpoint(self, name):
        # 记录关键时刻的实际状态并截图；pose用实际关节角算tool0位姿(FK)。
        xyz,rotation=pose(self.sim,self.sim.get_joint_positions())
        event={'name':name,'time_s':float(self.sim.data.time),'tool_xyz_m':xyz.tolist(),
               'cube_xyz_m':self.cube_xyz().tolist(),'physical_opening_m':self.sim.gripper_opening(),
               'finger_normal_force_n':self.sim.finger_cube_forces()}
        self.events.append(event)
        print(json.dumps(event),flush=True)
        save_scene_image(self.sim,self.output/f'{len(self.events):02d}_{name}.png')

    def verify_grasp_or_retreat(self):
        """抬升前检查双侧接触；没夹住则张爪、原地向上退让，并终止本次任务。"""
        forces = self.sim.finger_cube_forces()
        left_ok = forces['link7'] >= MIN_CONTACT_N
        right_ok = forces['link8'] >= MIN_CONTACT_N
        grasp_ok = left_ok and right_ok
        # and要求两侧都满足：True/True才是True；True/False仍是False。
        # 这里只确认当前瞬间双侧接触；后面还要检查方块确实抬起并放稳。
        self.grasp_check = {'time_s':float(self.sim.data.time),
                            'left_contact_n':forces['link7'],'right_contact_n':forces['link8'],
                            'threshold_n':MIN_CONTACT_N,'confirmed':bool(grasp_ok)}
        print(f'Grasp feedback: {self.grasp_check}',flush=True)
        if grasp_ok:
            return  # 回到run()，接着执行抬升。

        # 以下只会在双侧接触未通过时执行：先张爪，再竖直退让。
        self.grasp_recovery = 'started'
        self.phase = 'grasp_failed_open'
        self.change_gripper(OPENING_M)
        tool_xyz,_ = pose(self.sim,self.sim.get_joint_positions())
        # 保持实际X/Y，只向上退出；此路径仍须通过IK、采样碰撞和执行检查。
        retreat_xyz = [tool_xyz[0],tool_xyz[1],CARRY_TOOL_Z]
        self.move_to('grasp_failed_retreat',retreat_xyz,3)
        if self.sim.gripper_opening() < .06:
            raise RuntimeError('Grasp recovery failed: jaw did not open')
        self.grasp_recovery = 'completed'
        self.phase = 'stopped_grasp_failure'
        # raise与return不同：抛出指定异常，直接跳出run()的动作区，进入except。
        # 若这里只return，run()会错误地接着执行lift；所以退让后必须终止任务。
        raise GraspNotConfirmed('Grasp not confirmed; opened jaws and retreated; transport cancelled')

    def move_path(self, timestamps, positions):
        # 输入：N个时间点，以及形状(N,6)的参考关节角；按仿真步长细分后执行。
        carry_cube=self.phase in ('lift','transfer','lower')
        # in判断阶段名是否在元组中；搬运期间检查临时预测的随手运动方块。
        check_sampled_path(self.sim,positions[::max(1,len(positions)//100)],carry_cube=carry_cube)
        check_sampled_path(self.sim,positions[-1:],carry_cube=carry_cube)
        velocity=np.gradient(positions,timestamps,axis=0,edge_order=2)
        # 沿时间方向(axis=0)做差分，估计六关节参考角速度；首末速度强制为零。
        velocity[[0,-1]]=0
        duration=timestamps[-1]
        dt=self.sim.model.opt.timestep
        # IK、截图期间仿真没有前进；重新对齐时钟，避免查看器随后快速追赶。
        self.started=time.perf_counter()-self.sim.data.time
        for index in range(1,round(duration/dt)+1):
            t=min(index*dt,duration)
            q=np.array([np.interp(t,timestamps,positions[:,j]) for j in range(6)])
            v=np.array([np.interp(t,timestamps,velocity[:,j]) for j in range(6)])
            # positions[:,j]取第j个关节全部时刻的角度；interp插值得到当前t的值。
            # for j in range(6)依次处理编号0～5的六个关节，然后组合为数组。
            self.step(q,v)
        self.q_hold=positions[-1].copy()
        self.hold(.4)
        if np.max(np.abs(self.sim.get_joint_positions()-self.q_hold))>np.deg2rad(.3):
            # abs取每个误差的绝对值；max取最大项；任一关节误差超0.3度就失败。
            raise RuntimeError('Arm did not reach joint endpoint tolerance')

    def move_to(self, name, target_xyz, seconds, cartesian=True):
        # name标记动作阶段；target_xyz是tool0目标(不是方块中心)，seconds为原始时长。
        # cartesian=True：先在末端空间生成路点再逐点IK；False：只求终点IK后做关节插值。
        # 只缩短机械臂运动时间；夹爪闭合和接触稳定等待仍保留。
        seconds *= self.arm_time_scale
        self.phase=name
        start_q=self.sim.get_joint_positions()
        if cartesian:
            start_xyz,_=pose(self.sim,start_q)
            times=np.linspace(0,seconds,max(21,round(seconds/.04)+1))
            positions=[start_q]
            for t in times[1:]:
                u=t/seconds
                blend=10*u**3-15*u**4+6*u**5
                xyz=start_xyz+(np.asarray(target_xyz)-start_xyz)*blend
                # 位置从起点向终点按blend推进；IK用上一点关节角作seed，尽量保持连续。
                # DOWN_ROTATION为各求解路点的目标朝向；离散路点间仍由关节插值执行。
                result=solve_ik(self.sim,xyz,DOWN_ROTATION,seed=positions[-1])
                if not result['success']:
                    raise RuntimeError(f'Cartesian IK failed in {name}: {result}')
                if np.max(np.abs(result['q']-positions[-1]))>.25:
                    # 相邻IK解任一关节跳变超0.25rad(约14.3度)就终止，避免突跳。
                    raise RuntimeError(f'IK branch jump in {name}')
                positions.append(result['q'])
            positions=np.array(positions)
        else:
            # 首段接近使用此分支：关节平滑过渡，同时逐渐调整夹爪朝向。
            # 关节插值不保证tool0走直线，因此run()的最后下降另用末端空间路点。
            result=solve_ik(self.sim,target_xyz,DOWN_ROTATION,seed=start_q)
            if not result['success']:
                raise RuntimeError(f'IK failed in {name}: {result}')
            times,positions=quintic_trajectory(start_q,result['q'],seconds,self.sim.model.opt.timestep)
        self.move_path(times,positions)
        actual_xyz,_=pose(self.sim,self.sim.get_joint_positions())
        if np.linalg.norm(actual_xyz-target_xyz)>.004:
            # 执行完再核对实际末端；三维位置误差超过4毫米就报错。
            raise RuntimeError(f'Tool endpoint error in {name}: {actual_xyz-target_xyz}')
        self.checkpoint(name)

    def run(self):
        # 主流程入口，先读这里的编号1～7即可把握整体动作。
        # try执行动作；except处理失败；finally无论成败都保存已有记录并处理窗口。
        error=None
        completed=False
        try:
            if not self.headless:
                from mujoco import viewer as mj_viewer
                self.viewer=mj_viewer.launch_passive(self.sim.model,self.sim.data)
                with self.viewer.lock():
                    self.viewer.cam.lookat[:]=[.22,0,.15]
                    self.viewer.cam.distance=.95
                    self.viewer.cam.azimuth=135
                    self.viewer.cam.elevation=-30
            self.started=time.perf_counter()
            self.hold(.5)
            self.checkpoint('start')
            # 1. 斜向接近；夹爪朝向逐渐调整为向下。
            self.move_to('approach',[self.pick_xyz[0],self.pick_xyz[1],CARRY_TOOL_Z],5,cartesian=False)
            # 2. 最后竖直下降，避免侧推方块；tool0不等于夹指接触中心。
            grasp_xyz=self.pick_xyz+np.array([0,0,GRASP_TOOL_OFFSET_Z])
            # 方块中心XYZ + 工具相对中心的偏移 = 夹取时tool0应该到达的位置。
            self.move_to('descend',grasp_xyz,3)
            # 3. 用夹指实际接触夹住方块；物体保持自由刚体。
            self.phase='close'
            self.sim.allow_cube_contact=True
            # 开启夹指与方块正常接触的允许规则；并没有关闭桌面等其他接触检查。
            self.change_gripper(self.grasp_opening_m)
            self.checkpoint('grasp')
            self.verify_grasp_or_retreat()
            # 4. 抬升后必须观察到方块离开桌面。
            self.move_to('lift',[self.pick_xyz[0],self.pick_xyz[1],CARRY_TOOL_Z],3)
            if self.cube_xyz()[2] < self.pick_xyz[2]+.06:
                # [2]是Z高度；物体必须比初始中心高至少60毫米，光有闭爪命令不算抬起。
                raise RuntimeError('Cube did not lift from table')
            # 5. 保持高度平移到桌面另一个位置。
            self.move_to('transfer',[self.place_xyz[0],self.place_xyz[1],CARRY_TOOL_Z],4)
            if np.linalg.norm(self.cube_xyz()[:2]-self.place_xyz[:2])>.02:
                # [:2]只取X/Y；到目标上方后若物体水平偏差超过20毫米，认为发生滑移。
                raise RuntimeError('Cube slipped during transfer')
            # 6. 依据已夹住方块的实际偏移计算放置高度，预留1mm再释放。
            tool_xyz,_=pose(self.sim,self.sim.get_joint_positions())
            cube_offset=self.cube_xyz()-tool_xyz
            place_tool=self.place_xyz+np.array([0,0,.001])-cube_offset
            # 物体中心 = 工具位置 + 抓住后的实际偏移，因此反推工具位置时要减偏移。
            # +0.001让物体先略高于期望中心1毫米，再张爪靠接触自然落稳。
            self.move_to('lower',place_tool,3)
            self.phase='release'
            self.change_gripper(OPENING_M)
            self.checkpoint('release')
            # 7. 张爪退让，给方块时间在桌面上自然停稳。
            self.move_to('retreat',[place_tool[0],place_tool[1],CARRY_TOOL_Z],2.5)
            self.phase='settle'
            self.hold(1.0)
            self.checkpoint('finished')
            final=self.cube_xyz()
            speed=np.linalg.norm(self.sim.data.qvel[self.cube_dof:self.cube_dof+3])
            # 最终验收所有条件都要满足，按下面代码顺序分别是：
            # 水平误差<15mm；中心高度误差<4mm；物体平移速度<10mm/s；
            # 爪口>60mm；曾抬高超过60mm；最终两侧各自接触力均<0.02N。
            completed=bool(np.linalg.norm(final[:2]-self.place_xyz[:2])<.015
                           and abs(final[2]-(TABLE_TOP_Z+CUBE_HALF_SIZE))<.004
                           and speed<.01 and self.sim.gripper_opening()>.06
                           and self.max_cube_z>self.pick_xyz[2]+.06
                           and max(self.sim.finger_cube_forces().values())<.02)
            if not completed:
                raise RuntimeError('Final object placement/stability acceptance failed')
        except (Exception,KeyboardInterrupt) as exc:
            # 普通异常和Ctrl+C都会记录为失败；只有未确认夹持的专用分支会主动退让。
            completed=False
            error=str(exc) or type(exc).__name__
            if isinstance(exc,GraspNotConfirmed):
                print(str(exc),flush=True)
            elif not isinstance(exc,KeyboardInterrupt):
                traceback.print_exc()
            try:self.checkpoint('failed')
            except Exception:pass
        finally:
            # 即使没有完成，save仍保存已采集的数据；完成退让也不会把抓放记成成功。
            self.save(completed,error)
            if self.viewer is not None:
                try:
                    if completed or self.grasp_recovery == 'completed':
                        message='Task complete.' if completed else 'Grasp failed; recovery completed; no transport.'
                        print(message+' Final scene displayed; close MuJoCo window to exit.',flush=True)
                        # 任务结束后的终态展示，不再推进或改写仿真状态。
                        while self.viewer.is_running():
                            self.viewer.sync()
                            time.sleep(.02)
                except KeyboardInterrupt:
                    pass
                finally:
                    self.viewer.close()
        return completed

    def save(self,completed,error):
        # tracking.csv记录连续采样；pick_place_report.json记录总体结论与关键时刻。
        # range(1,7)产生1～6，用于表头编号；它与数组下标0～5是不同用途。
        columns=['time_s','phase']+[f'target_j{i}_rad' for i in range(1,7)]
        columns += [f'actual_j{i}_rad' for i in range(1,7)]+[f'actual_v{i}_rad_s' for i in range(1,7)]
        columns += ['target_physical_opening_m','physical_opening_m','cube_x_m','cube_y_m','cube_z_m',
                    'left_contact_n','right_contact_n']+[f'actuator_tau{i}_nm' for i in range(1,7)]
        with (self.output/'tracking.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream);writer.writerow(columns);writer.writerows(self.rows)
        cube=self.cube_xyz()
        report={'success':completed,'error':error,'duration_s':float(self.sim.data.time),
                'speed_profile':self.speed_profile,'arm_time_scale':self.arm_time_scale,
                'grasp_opening_command_m':self.grasp_opening_m,
                'grasp_check':self.grasp_check,'grasp_recovery':self.grasp_recovery,
                'cube_start_xyz_m':self.pick_xyz.tolist(),'place_target_xyz_m':self.place_xyz.tolist(),
                'cube_final_xyz_m':cube.tolist(),'max_cube_z_m':self.max_cube_z,
                'final_xy_error_m':float(np.linalg.norm(cube[:2]-self.place_xyz[:2])),
                'warning_count':sum(int(w.number) for w in self.sim.data.warning),
                'unexpected_robot_contacts':self.sim.contact_violations,'checkpoints':self.events,
                'mechanism':'Free body moved only by physical finger/table contact; no weld or position rewrite',
                'scope':'MuJoCo simulation; ideal external gravity support on robot; not hardware validation'}
        (self.output/'pick_place_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        (self.output/'scene.xml').write_text(self.sim.xml,encoding='utf-8')
        if self.rows:
            # rows每行索引20～26依次是：目标开口、实际开口、方块X/Y/Z、左/右接触力。
            # 加上时间后numeric列0为时间、列2为实际开口、列5为方块Z。
            numeric=np.array([[r[0],*r[20:27]] for r in self.rows],dtype=float)
            fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
            axes[0].plot(numeric[:,0],numeric[:,5]*1000,label='Cube center Z')
            # 乘1000把米变为毫米；图中画的是实际反馈，不是发出去的目标指令。
            axes[0].axhline((TABLE_TOP_Z+CUBE_HALF_SIZE)*1000,color='gray',linestyle='--',label='Resting height')
            axes[0].set_ylabel('Cube height (mm)')
            axes[1].plot(numeric[:,0],numeric[:,2]*1000,label='Physical jaw opening')
            axes[1].set_ylabel('Opening (mm)');axes[1].set_xlabel('Simulation time (s)')
            for ax in axes:ax.grid(alpha=.3);ax.legend()
            fig.suptitle('Physical contact pick and place | simulation only')
            fig.tight_layout();fig.savefig(self.output/'task_trace.png',dpi=145);plt.close(fig)
        print(json.dumps({k:v for k,v in report.items() if k!='checkpoints'},indent=2),flush=True)
        print(f'Results: {self.output}',flush=True)
