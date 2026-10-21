"""固定基础设施：Alicia-M、桌面、自由方块和目标标记。位置统一为世界坐标米。"""
# 本文件负责“搭建世界”和“读取接触反馈”；动作顺序请先读pick_place.py的run()。
# 初学时先看桌面/方块尺寸、set_gripper、finger_cube_forces，XML细节可稍后读。
# MuJoCo中model保存模型参数，data保存当前关节状态、接触和仿真时间等变化量。
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT.parent))
# 把上一级项目目录加入模块搜索路径，才能借用simulation.alicia_m中的已有仿真器。
from simulation.alicia_m import AliciaMSim

TABLE_TOP_Z = 0.05
CUBE_HALF_SIZE = 0.015
# 几何尺寸都用米；方块半边长15mm，完整边长30mm。
CUBE_XYZ = np.array([0.28, 0.05, TABLE_TOP_Z + CUBE_HALF_SIZE])
HOVER_CLEARANCE = 0.12  # 相对于方块上表面，模型tool0参考点的高度。
HOVER_XYZ = CUBE_XYZ + np.array([0, 0, CUBE_HALF_SIZE + HOVER_CLEARANCE])
PLACE_XYZ = np.array([0.28, -0.12, TABLE_TOP_Z + CUBE_HALF_SIZE])
FINGER_BASE_GAP = 0.10  # 原模型滑块为零时，两夹指内侧之间约100毫米。
# 夹爪局部Z轴朝世界-Z，局部X朝世界-X；一个明确的向下姿态。
DOWN_ROTATION = np.diag([-1.0, 1.0, -1.0])
# 这是3×3旋转矩阵，用来指定朝向；不是[X,Y,Z]位置，也不是欧拉角列表。


def table_point(xyz):
    """本课只处理完全位于当前桌面上的30mm方块；不是可达性判定。"""
    point=np.array(xyz,dtype=float,copy=True)
    # 把输入统一成浮点数组；shape==(3,)表示恰好三个数，isfinite排除无穷/NaN。
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('位置必须是三个有限数，单位为米')
    if not np.isclose(point[2],TABLE_TOP_Z+CUBE_HALF_SIZE,atol=1e-6,rtol=0):
        # 浮点数比较用isclose；中心高度必须等于桌面高度加方块半高，容差1微米。
        raise ValueError('本课方块中心Z必须为0.065米：桌高0.05米加半边长0.015米')
    if not (.12+CUBE_HALF_SIZE <= point[0] <= .56-CUBE_HALF_SIZE
            and abs(point[1]) <= .22-CUBE_HALF_SIZE):
        # 桌面X为0.12～0.56米，Y为-0.22～0.22米；扣掉半边长保证整个方块在桌上。
        raise ValueError('方块必须完全位于桌面范围内；桌面内的位置仍可能不可达')
    return point


class DesktopSim(AliciaMSim):
    # 继承AliciaMSim：复用机器人模型和控制接口，再添加本课桌面与方块。
    def __init__(self, pick_xyz=None, place_xyz=None):
        # None表示调用者没提供坐标，此时使用上方默认值；有输入则先检查格式和桌面范围。
        self.pick_xyz=table_point(CUBE_XYZ if pick_xyz is None else pick_xyz)
        self.place_xyz=table_point(PLACE_XYZ if place_xyz is None else place_xyz)
        hover_xyz=self.pick_xyz+np.array([0,0,CUBE_HALF_SIZE+HOVER_CLEARANCE])
        super().__init__()
        # super()先调用父类加载机器人；再把父类的XML转成可修改的树。
        root = ET.fromstring(self.xml)
        # 接触抓取使用更短的接触响应时间；配合平滑闭爪降低软接触穿透。
        for body in root.findall('.//body'):
            if body.get('name') in ('link7','link8'):
                for geom in body.findall('geom'):
                    if geom.get('contype','1') != '0':
                        geom.set('solref','0.006 1')
                        geom.set('solimp','0.95 0.99 0.001')
                        geom.set('friction','1.2 0.01 0.001')
        visual = ET.SubElement(root, 'visual')
        ET.SubElement(visual, 'global', offwidth='960', offheight='720')
        world = root.find('worldbody')
        # ET.SubElement是在XML树里添加元素；这里依次添加桌面、方块和可视标记。
        # MuJoCo box的size写半尺寸，所以桌面Z半高0.025、中心Z=0.025，顶面才是0.05。
        ET.SubElement(world, 'geom', name='desktop', type='box',
                      pos='0.34 0 0.025', size='0.22 0.22 0.025',
                      rgba='0.55 0.4 0.27 1', friction='1 0.01 0.001')
        cube = ET.SubElement(world, 'body', name='task_cube', pos=' '.join(map(str,self.pick_xyz)))
        ET.SubElement(cube, 'freejoint', name='cube_free')
        # freejoint让方块成为能自由平移/转动的刚体；抓取依赖接触和摩擦。
        ET.SubElement(cube, 'geom', name='cube_geom', type='box',
                      size=f'{CUBE_HALF_SIZE} {CUBE_HALF_SIZE} {CUBE_HALF_SIZE}',
                      mass='0.03', rgba='1 0.65 0.05 1', friction='1.2 0.01 0.001',
                      solref='0.006 1', solimp='0.95 0.99 0.001')
        ET.SubElement(world, 'site', name='hover_target', pos=' '.join(map(str,hover_xyz)),
                      type='sphere', size='0.008', rgba='0.1 1 0.2 0.6')
        # site只是可视参考标记，不会托住、吸住或推走方块。
        ET.SubElement(world, 'site', name='future_place_marker',
                      pos=f'{self.place_xyz[0]} {self.place_xyz[1]} {TABLE_TOP_Z+.001}',
                      type='cylinder', size='0.025 0.001', rgba='0.2 0.5 1 0.6')
        # 新增自由方块有7个位置坐标；为原ready关键帧补齐，不改变六轴定义。
        cube_qpos = ' '.join(map(str, [*self.pick_xyz, 1, 0, 0, 0]))
        # 自由刚体qpos的7项是XYZ + 四元数[w,x,y,z]；[1,0,0,0]表示初始无旋转。
        for key in root.findall('keyframe/key'):
            key.set('qpos', key.get('qpos')+' '+cube_qpos)
        self.xml = ET.tostring(root,encoding='unicode')
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        # 用修改后的XML重新构建model和data；名称转编号后才能从数组里定位相应对象。
        self.data = mujoco.MjData(self.model)
        self.tool_site = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'tool0_site')
        self.cube_body = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'task_cube')
        self.moving_bodies = {mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,f'link{i}') for i in range(1,9)}
        self.contact_violations = []
        self.allow_cube_contact = False
        self.reset()
        self.set_gripper(0.06)

    def set_gripper(self, opening_m):
        """本项目接口表示真实爪口宽度；父类接口实际表示滑块间距。"""
        if not np.isfinite(opening_m) or not 0 <= opening_m <= FINGER_BASE_GAP:
            raise ValueError('真实夹爪开口必须在0～0.10米之间')
        super().set_gripper(FINGER_BASE_GAP-opening_m)
        # 目标开口越大，父类所需滑块间距越小；例如70mm开口对应传入30mm。

    def gripper_opening(self):
        # 读取两滑块实际位置之差，再换算实际爪口；与set_gripper的目标值可能不同。
        return FINGER_BASE_GAP-float(np.diff(self.data.qpos[self.finger_qpos])[0])

    def finger_cube_forces(self):
        """两侧夹指与自由方块接触的法向力；不是位置绑定。"""
        forces = {'link7':0.0,'link8':0.0}
        # 左右夹指各存一个力，初值0表示尚无接触；enumerate给出接触编号和接触内容。
        for index,contact in enumerate(self.data.contact[:self.data.ncon]):
            bodies = self.model.geom_bodyid[[contact.geom1,contact.geom2]]
            if self.cube_body not in bodies:
                # continue只跳过本次循环：不涉及方块的接触无需计入夹持反馈。
                continue
            for name in forces:
                finger=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,name)
                if finger in bodies:
                    wrench=np.zeros(6)
                    mujoco.mj_contactForce(self.model,self.data,index,wrench)
                    # wrench有6项(力和力矩)，第0项是接触局部坐标系中的法向力。
                    # 同一夹指可能有多个接触点，用+=累加；max(0,...)只累计非负法向力。
                    forces[name] += max(0.0,float(wrench[0]))
        return forces

    def robot_penetrations(self, data=None):
        # 既能检查当前真实仿真data，也能检查IK路径预检时传入的临时data。
        # contact.dist为负表示软接触穿透；这里只按本教学模型阈值筛选异常。
        data = self.data if data is None else data
        hits = []
        for contact in data.contact[:data.ncon]:
            bodies = self.model.geom_bodyid[[contact.geom1,contact.geom2]]
            if self.allow_cube_contact and self.cube_body in bodies:
                names={mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,int(b)) for b in bodies}
                if names & {'link7','link8'} and contact.dist > -0.003:
                    # 集合的&取交集：判断接触对象中是否有夹指；正常抓取允许小于3mm软穿透。
                    continue  # 允许夹指-方块的正常接触；桌面碰撞仍检查。
            if contact.dist < -0.0005 and any(int(b) in self.moving_bodies for b in bodies):
                # 其余涉及运动机器人部件、穿透超过0.5mm的接触会被记入hits。
                hits.append({'time_s':float(data.time),'penetration_m':float(-contact.dist),
                             'geom1':mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,int(contact.geom1)),
                             'geom2':mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,int(contact.geom2))})
        return hits

    def step(self, steps=1):
        # 覆盖父类step：每前进一步，额外执行本课的异常接触检查。
        for _ in range(steps):
            super().step()
            hits = self.robot_penetrations()
            if hits and len(self.contact_violations) < 20:
                self.contact_violations.extend(hits)


def save_scene_image(sim, path):
    """渲染当前真实仿真状态；不修改机器人或方块位置。"""
    from PIL import Image
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.22,0,0.20]
    camera.distance = 0.95
    camera.azimuth = 135
    camera.elevation = -30
    with mujoco.Renderer(sim.model,height=720,width=960) as renderer:
        # 只渲染当前data并保存PNG；with负责释放渲染资源，不推进物理时间。
        renderer.update_scene(sim.data,camera=camera)
        Image.fromarray(renderer.render()).save(path)
