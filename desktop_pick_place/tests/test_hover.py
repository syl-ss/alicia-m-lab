"""First-step acceptance: scene coordinates and pose IK agree in one model."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scene import DesktopSim, CUBE_XYZ, HOVER_XYZ, DOWN_ROTATION
from kinematics import solve_ik, pose


class HoverTests(unittest.TestCase):
    def test_scene_has_free_cube_and_hover_above_it(self):
        sim = DesktopSim()
        self.assertEqual(sim.model.nq, 15)
        np.testing.assert_allclose(sim.data.xpos[sim.cube_body], CUBE_XYZ)
        np.testing.assert_allclose(HOVER_XYZ[:2], CUBE_XYZ[:2])
        self.assertGreater(HOVER_XYZ[2], CUBE_XYZ[2])

    def test_ik_reaches_requested_position_and_downward_orientation(self):
        sim = DesktopSim()
        result = solve_ik(sim, HOVER_XYZ, DOWN_ROTATION)
        self.assertTrue(result['success'])
        xyz, rotation = pose(sim, result['q'])
        self.assertLess(np.linalg.norm(xyz-HOVER_XYZ), 0.0002)
        self.assertLess(np.linalg.norm(Rotation.from_matrix(DOWN_ROTATION@rotation.T).as_rotvec()), 0.002)
        self.assertTrue(np.all(result['q'] >= sim.joint_limits[:,0]))
        self.assertTrue(np.all(result['q'] <= sim.joint_limits[:,1]))


if __name__ == '__main__':
    unittest.main()
