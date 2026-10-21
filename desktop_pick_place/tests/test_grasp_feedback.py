"""真实物理仿真验证：检测没夹住后应退让，绝不进入搬运。"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pick_place import PickPlaceRunner, OPENING_M, CARRY_TOOL_Z
from kinematics import pose


class GraspFeedbackTests(unittest.TestCase):
    def test_unclosed_gripper_is_detected_and_retreats_without_transport(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            output=Path(directory)
            task=PickPlaceRunner(output,headless=True,speed='fast',grasp_opening_m=OPENING_M)
            self.assertFalse(task.run())
            report=json.loads((output/'pick_place_report.json').read_text())
            self.assertFalse(report['success'])
            self.assertFalse(report['grasp_check']['confirmed'])
            self.assertEqual(report['grasp_recovery'],'completed')
            phases={row[1] for row in task.rows}
            self.assertTrue(phases.isdisjoint({'lift','transfer','lower','release'}))
            self.assertIn('grasp_failed_retreat',phases)
            self.assertGreater(task.sim.gripper_opening(),.06)
            xyz,_=pose(task.sim,task.sim.get_joint_positions())
            self.assertAlmostEqual(xyz[2],CARRY_TOOL_Z,delta=.004)
            np.testing.assert_allclose(task.cube_xyz(),task.pick_xyz,atol=.001)
            self.assertEqual(report['warning_count'],0)
            self.assertEqual(report['unexpected_robot_contacts'],[])


if __name__=='__main__':
    unittest.main()
