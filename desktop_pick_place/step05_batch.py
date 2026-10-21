"""第五课：多组位置复用同一抓放程序，逐项保存结果。只运行MuJoCo，无串口。"""
# 阅读路线：先看 step03_coordinates.py（一次任务的输入），再看
# step04_feedback.py（故意夹空），最后读本文件（循环执行多次任务）。
# 真正的动作在 pick_place.py 的 PickPlaceRunner.run()；本文件负责组织实验。
# 数据流：TASKS -> item -> PickPlaceRunner -> 单项报告 -> row -> rows -> 汇总文件。
import csv
from datetime import datetime
import json
from pathlib import Path
from pick_place import PickPlaceRunner

# 每项任务都重新创建场景；这是独立测试，不是一次连续分拣。
# TASKS 是列表；其中每个 {...} 是一份任务字典。
# name 是文件夹名；pick/place 是方块中心的 [X, Y, Z]，单位米。
# .28 就是 0.28；.065 米 = 65 毫米，来自桌高50毫米 + 方块半高15毫米。
TASKS = [
    {'name':'original', 'pick':[.28,.05,.065], 'place':[.28,-.12,.065]},
    {'name':'shifted', 'pick':[.27,.07,.065], 'place':[.29,-.10,.065]},
    {'name':'diagonal', 'pick':[.26,.04,.065], 'place':[.30,-.08,.065]},
]


def run_batch(output):
    # output 是 Path 路径对象；/ 在这里用于拼接路径，不是数值除法。
    # parents=True：缺少父目录也创建；exist_ok=True：目录已存在也可以继续。
    output.mkdir(parents=True,exist_ok=True)
    # 先保存测试输入，结果不能脱离测试条件。
    (output/'tasks.json').write_text(json.dumps(TASKS,indent=2),encoding='utf-8')
    # rows 是全部结果的列表，row 是当前这一项的结果字典。
    rows=[]
    for item in TASKS:
        # 每次循环，item 依次代表 original、shifted、diagonal 其中一份字典。
        # item['name'] 是按键取值；三个任务各用自己的子文件夹保存结果。
        case_output=output/item['name']
        print(f"Starting case: {item['name']}",flush=True)
        # 先按失败初始化；None 表示尚无数据，不能用0冒充已测量的误差。
        row={'name':item['name'],'success':False,'duration_s':None,
             'final_xy_error_mm':None,'error':None,'result_dir':str(case_output)}
        try:
            # try 中的代码若发生异常，会跳到下面 except，记录原因后继续下一项。
            # 创建新的执行对象也会创建新场景。headless=True：不弹出仿真窗口。
            task=PickPlaceRunner(case_output,headless=True,speed='fast',
                                 pick_xyz=item['pick'],place_xyz=item['place'])
            # run() 才真正执行动作；返回 True 表示通过最终抓放验收。
            success=task.run()
            # read_text 读取文件文字；json.loads 把JSON文字还原成Python字典。
            report=json.loads((case_output/'pick_place_report.json').read_text(encoding='utf-8'))
            # update 更新字典里的字段；报告中的误差是米，乘1000转为毫米。
            row.update(success=success,duration_s=report['duration_s'],
                       final_xy_error_mm=report['final_xy_error_m']*1000,error=report['error'])
        except Exception as exc:
            # 场景创建等异常也保留为失败项，不能跳过后只汇总成功结果。
            row['error']=f'{type(exc).__name__}: {exc}'
        # append 把当前结果追加到列表；失败项同样保留。
        rows.append(row)
        # 每完成一项立即落盘，后续任务失败也保留已有结果。
        with (output/'summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            # with 会自动关闭文件；'w' 会重写汇总，writerows(rows) 写回全部已有结果。
            # utf-8-sig 帮助Excel识别中文；fieldnames 用字典的键作为表头。
            writer=csv.DictWriter(stream,fieldnames=list(row))
            writer.writeheader();writer.writerows(rows)
        print(f"Case result: {row}",flush=True)
    # 布尔值参与求和时，True算1、False算0；三项全成功就得到3。
    passed=sum(row['success'] for row in rows)
    summary={'passed':passed,'total':len(rows),'cases':rows,
             'scope':'Three selected positions in deterministic MuJoCo simulation; not general reliability or hardware validation'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(f'Passed {passed}/{len(rows)} cases. Results: {output}',flush=True)
    # len(rows) 是任务总数；== 比较是否相等，返回True或False给调用者。
    return passed==len(rows)


if __name__=='__main__':
    # 直接运行此文件才进入这里；被其他文件import时不会自动跑批量实验。
    # __file__ 是当前脚本；resolve() 得到绝对路径；parent 是所在文件夹。
    # 时间戳让不同批次用不同目录；本入口默认无窗口运行。
    folder=Path(__file__).resolve().parent/'results'/('batch_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    # 条件表达式：全部成功选0，否则选1；SystemExit 把它交给终端作为退出码。
    raise SystemExit(0 if run_batch(folder) else 1)
