"""第四课：用反馈决定继续搬运还是张爪退让。仅连接MuJoCo。"""
# 本文件选择实验条件；实际的左右接触力判断在
# pick_place.py -> verify_grasp_or_retreat()，由run()在闭爪后、抬升前调用。
# 阅读重点：发出闭爪命令 -> 检查真实仿真接触 -> 继续抬升，或退让并结束。
import argparse
from datetime import datetime
from pathlib import Path
from pick_place import PickPlaceRunner, OPENING_M
from step03_coordinates import PICK_XYZ, PLACE_XYZ
# 这里只借用上一课的坐标；上一课有 __main__ 入口保护，不会自动执行抓放。


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless',action='store_true')
    # 不加选项：正常抓取；加 --demo-empty-grasp：故意张着爪子检验失败分支。
    parser.add_argument('--demo-empty-grasp',action='store_true',
                        help='故意保持张爪，触发未夹持检测和退让；抓放结果仍为失败，退出码1')
    args=parser.parse_args()
    # 正常夹持给26mm开口；失败实验给70mm，故意让爪口大于30mm方块。
    # A if 条件 else B：条件为True选A，为False选B；这里的数值单位是米。
    # .026是目标开口；方块接触会阻止夹指完全闭到目标，实际开口需读反馈。
    grasp_opening = OPENING_M if args.demo_empty_grasp else .026
    print('Experiment: '+('gripper stays open; expect grasp failure and retreat'
                          if args.demo_empty_grasp else 'normal grasp; expect pick and place'),flush=True)
    output=Path(__file__).resolve().parent/'results'/('feedback_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    # 把选好的开口传给同一个执行程序，不需要复制一套抓放流程。
    task=PickPlaceRunner(output,headless=args.headless,speed='fast',
                         pick_xyz=PICK_XYZ,place_xyz=PLACE_XYZ,grasp_opening_m=grasp_opening)
    # 故意夹空时预期返回1；报告里grasp_recovery='completed'只代表退让完成。
    # 退让后通过抛出GraspNotConfirmed终止流程，不能继续执行lift/transfer。
    return 0 if task.run() else 1


if __name__=='__main__':
    raise SystemExit(main())
