"""场景与执行必须共享本次坐标；错误输入不能生成一个看似正常的场景。"""
from pathlib import Path
import sys
import tempfile
import unittest
import mujoco
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scene import DesktopSim
from pick_place import PickPlaceRunner


class TaskCoordinatesTests(unittest.TestCase):
    def test_custom_coordinates_reach_scene_and_runner_without_global_leak(self):
        pick=np.array([.27,.07,.065])
        place=np.array([.29,-.10,.065])
        with tempfile.TemporaryDirectory() as directory:
            task=PickPlaceRunner(Path(directory),headless=True,pick_xyz=pick,place_xyz=place)
            np.testing.assert_allclose(task.cube_xyz(),pick)
            marker=mujoco.mj_name2id(task.sim.model,mujoco.mjtObj.mjOBJ_SITE,'future_place_marker')
            np.testing.assert_allclose(task.sim.data.site_xpos[marker][:2],place[:2])
            pick[:]=0;place[:]=0  # 调用者后续改数组不应悄悄改动任务。
            np.testing.assert_allclose(task.pick_xyz,[.27,.07,.065])
            np.testing.assert_allclose(task.place_xyz,[.29,-.10,.065])
            np.testing.assert_allclose(DesktopSim().data.xpos[task.sim.cube_body],[.28,.05,.065])

    def test_invalid_table_coordinates_are_rejected(self):
        for point in ([.28,.05], [.28,np.nan,.065], [.28,.05,.1], [.8,0,.065]):
            with self.subTest(point=point),self.assertRaises(ValueError):
                DesktopSim(pick_xyz=point)


if __name__=='__main__':
    unittest.main()
