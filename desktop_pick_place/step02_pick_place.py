"""完整演示：靠物理接触夹起黄色方块，搬到桌面蓝色标记处并释放。"""
# 这是固定坐标的启动入口；想改坐标先读step03，想看具体动作读pick_place.run。
# main负责准备参数；PickPlaceRunner负责执行。先区分这两层就更容易读懂。
import argparse
from datetime import datetime
from pathlib import Path

from pick_place import PickPlaceRunner, SPEED_PROFILES


def main():
    # parser接收终端参数；--headless表示不显示窗口，仍然计算完整物理仿真。
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless',action='store_true')
    # choices限制可选名称；fast让机械臂各段时间乘0.5，normal乘1.0。
    # 夹爪动作和等待时间另算，因此整套任务总耗时不会刚好减半。
    parser.add_argument('--speed',choices=SPEED_PROFILES,default='fast',
                        help='fast: half arm motion time; normal: original teaching speed')
    args=parser.parse_args()
    output=Path(__file__).resolve().parent/'results'/('pick_place_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    # 没传pick_xyz/place_xyz，执行程序会使用scene.py里的默认位置。
    task=PickPlaceRunner(output,headless=args.headless,speed=args.speed)
    # 创建任务之后调用run()开始运动；True转换成成功退出码0，False转为1。
    return 0 if task.run() else 1


if __name__=='__main__':
    # 只有直接运行本文件才调用main；SystemExit将返回值作为终端退出码。
    raise SystemExit(main())
