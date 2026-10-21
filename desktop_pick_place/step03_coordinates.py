"""第三课：只改两个位置，复用整套桌面抓放流程。"""
# 建议先读本文件：它只负责接收坐标、创建任务、启动任务。
# import 是借用别的模块；from pick_place import ... 借用已写好的执行程序。
# 完整调用链：main() -> PickPlaceRunner(...) -> task.run() -> 保存仿真结果。
# 第一次阅读只关注 PICK_XYZ、PLACE_XYZ 和 task.run()；底层算法可以稍后再看。
import argparse
from datetime import datetime
from pathlib import Path
from pick_place import PickPlaceRunner, SPEED_PROFILES

# 世界坐标，单位米；这里是方块中心，不是夹爪tool0的位置。
# [X, Y, Z] 是三元素列表；0.27米=270毫米，负Y表示世界坐标Y轴的负方向。
# 当前桌高0.05米、方块边长0.03米，所以中心Z固定为0.065米。
# 先只改X/Y；桌面内也不一定机械臂可达，后续还会检查IK和采样接触。
PICK_XYZ = [0.27, 0.07, 0.065]
PLACE_XYZ = [0.29, -0.10, 0.065]


def main():
    # argparse 把终端选项变成 args 上的属性；不填选项就使用默认值。
    parser=argparse.ArgumentParser(description=__doc__)
    # store_true：写了 --headless 就是True；不写就是False（显示仿真窗口）。
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--speed',choices=SPEED_PROFILES,default='fast')
    # nargs=3：后面必须跟三个数；type=float：把输入文字转换为小数。
    # 示例：--pick 0.27 0.07 0.065 --place 0.29 -0.10 0.065
    parser.add_argument('--pick',nargs=3,type=float,default=PICK_XYZ,metavar=('X','Y','Z'))
    parser.add_argument('--place',nargs=3,type=float,default=PLACE_XYZ,metavar=('X','Y','Z'))
    args=parser.parse_args()
    # Path 的 / 表示拼接目录；每次在当前脚本旁的results里生成带时间戳的目录。
    output=Path(__file__).resolve().parent/'results'/('coordinates_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    # 同一份输入传给场景、路点生成和最后的成功判断。
    # PickPlaceRunner 是类（任务模板）；task 是这一次任务的对象。
    # name=value 是按参数名传值；创建对象时搭建场景，run() 时才开始抓放。
    task=PickPlaceRunner(output,headless=args.headless,speed=args.speed,
                         pick_xyz=args.pick,place_xyz=args.place)
    # run() 返回True选0（成功）；False选1（失败），供终端识别结果。
    return 0 if task.run() else 1


if __name__=='__main__':
    # 这是直接运行脚本的入口；被step04导入坐标时，不会在这里启动任务。
    raise SystemExit(main())
