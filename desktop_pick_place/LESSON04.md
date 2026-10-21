# 第四课：让反馈决定后续动作

目标：区分“发出闭爪命令”和“确认夹持”，理解传感结果怎样通过条件判断改变动作流程。

## 先运行两个实验

在本文件夹的PowerShell中依次运行，观察完一组后关闭MuJoCo窗口：

```powershell
& '..\.venv\Scripts\python.exe' .\step04_feedback.py
& '..\.venv\Scripts\python.exe' .\step04_feedback.py --demo-empty-grasp
```

第一组正常夹持，预计完整抓放。第二组故意让爪口保持70毫米，大于30毫米方块，预计没有双侧夹持接触。程序应张爪、竖直退回物体上方，不再进入抬升搬运流程，随后保留终态供观察。方块始终是自由刚体，没有改写它的位置或接触读数。

第二组抓放结果应为失败：报告`success=false`、进程退出码1是预期；`grasp_recovery=completed`只说明退让完成，不表示抓放成功。双击入口可用`run_feedback.cmd`，失败演示用`run_feedback.cmd --demo-empty-grasp`。

## 从入口到反馈

`step04_feedback.py`只选择实验条件并创建任务。正常指令开口为0.026米，失败实验指令为0.07米；其余执行与检查逻辑相同。

执行程序`pick_place.py`到抓取阶段时，依次调用：

```python
self.change_gripper(self.grasp_opening_m)
self.checkpoint('grasp')
self.verify_grasp_or_retreat()
```

第一行发出平滑开口指令并等待；第二行保存实际状态；第三行读取物理反馈、决定能不能继续。命令目标和实际反馈是两个方向的数据。

## 关键判断

`verify_grasp_or_retreat()`的核心为：

```python
forces = self.sim.finger_cube_forces()
left_ok = forces['link7'] >= MIN_CONTACT_N
right_ok = forces['link8'] >= MIN_CONTACT_N
grasp_ok = left_ok and right_ok
```

`forces`是字典，包含左、右夹指接触方块的法向力，单位牛顿。`MIN_CONTACT_N=0.02`是当前仿真教学阈值，不是实机经过标定的夹持标准。

如果左右分别为0.94和0.95，两个判断都为True，`grasp_ok=True`。如果有一侧为0，那么对应判断为False，`and`要求两侧都成立，因此`grasp_ok=False`。

```python
if grasp_ok:
    return
```

这里的`return`只退出检查函数，返回`run()`后继续执行下一行抬升。它不会结束整个抓放程序。

没夹住时，代码不会进入上面的return，会继续执行本函数剩下的失败分支：张爪→读实际末端位置→保持当前X/Y向上退到0.15米→记录退让结果→抛出`GraspNotConfirmed`。

这个异常会被外层`run()`捕获：保留失败原因、保存日志，然后结束任务。它会跳过原本排在后面的抬升、搬运和放置。仅仅张爪退让后普通return是不够的，那样run()仍会继续搬运。

## 为什么还有抬升检查

双侧接触只是允许尝试抬升的证据，不能保证物体一直夹稳。因此原流程还保留“物体是否真的抬高”的检查，通过后才水平搬运。当前高度来自仿真物体状态；未来真机需要相机、力觉等可用反馈，不能把它当作SDK天然提供的传感器。

本课只处理抬升前没有确认夹持的情况。IK失败、碰撞、查看器关闭或抬升中滑落不会自动复用这条张爪退让路径；那需要另外设计恢复策略。

## 看什么结果

结果在`results/feedback_<时间>/pick_place_report.json`与`tracking.csv`：

| 项目 | 正常实验预期 | 故意张爪实验预期 |
| --- | --- | --- |
| grasp_check.confirmed | true | false |
| grasp_recovery | not_needed | completed |
| success | true | false |
| 动作 | 抬升、搬运、放下 | 张爪、竖直退让 |
| 方块 | 移到目标区 | 留在原桌面位置 |

验收：你运行这两个实验，能解释“反馈→两个比较→and→return或失败分支→继续或取消搬运”的完整过程。所有结果仅为MuJoCo仿真。
