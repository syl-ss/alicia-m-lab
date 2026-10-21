# 第五课：从一次演示到一组实验

目标：给同一程序一组位置任务，逐个执行，保留成功与失败结果。先只改变位置，方块质量、大小、摩擦和速度档保持不变。

运行`run_batch.cmd`，或在本文件夹运行：

```powershell
& '..\.venv\Scripts\python.exe' .\step05_batch.py
```

本次默认无窗口自动执行，避免每个任务的终态查看器等待关闭；每项仍保存开始、抓取、抬升、放置等场景图片和完整日志。测试输入在`results/batch_<时间>/tasks.json`，总表为`summary.csv`，每项任务目录含`tracking.csv`与`pick_place_report.json`。

## 代码的数据流

1. `TASKS`列表保存三项任务，每项字典含名称、抓取中心和放置中心。
2. `for item in TASKS`依次取出一个任务。
3. `PickPlaceRunner(..., pick_xyz=item['pick'], place_xyz=item['place'])`创建独立场景。
4. `task.run()`执行前几课已有的整套抓放与反馈检查，返回本次成功与否。
5. 读取报告，把成功状态、仿真用时、放置误差和失败原因写入总表。

核心代码：

```python
for item in TASKS:
    task = PickPlaceRunner(
        output / item['name'],
        headless=True,
        speed='fast',
        pick_xyz=item['pick'],
        place_xyz=item['place'],
    )
    success = task.run()
```

这里的for循环只负责安排任务，内部抓放算法没有复制三份。以后增加一种轨迹或改进抓取判断，只需修改共享执行程序，三项任务都会使用更新后的逻辑。

每项任务重新创建场景，是独立测试之间的初始化，不是搬运过程中修改方块位置。当前也不是一条流水线上连续搬三个物体。

## 结果怎么解释

`success`来自原有物理验收：确认接触、物体真的抬高、放置后停稳并脱离夹爪。不能仅用“程序没有报错”当成功。

`final_xy_error_mm`是物体最终中心与目标中心的水平距离，不是机械臂tool0误差。`duration_s`是仿真运动时间，不含Python启动、IK计算与截图耗时。

三组全成功只说明这些选定位置在当前模型下通过。不能将其解释成通用抓取成功率100%，也不能推导真机精度。重复相同确定性场景不足以测试鲁棒性；后续再逐项加入位置扰动、质量与摩擦变化。

验收：运行一次，打开总表和其中一项的开始/抬升/完成图片，能够串起“任务输入→执行→物理反馈→验收→结果记录”。
