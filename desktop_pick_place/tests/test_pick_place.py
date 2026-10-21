"""Grasp acceptance must use physical object motion and actual jaw geometry."""
from pathlib import Path
import sys
import unittest
import mujoco
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scene import DesktopSim
from kinematics import check_sampled_path


class PhysicalGripperTests(unittest.TestCase):
    def test_carried_object_path_check_never_changes_actual_object(self):
        sim=DesktopSim()
        before=sim.data.qpos.copy()
        sim.allow_cube_contact=True
        check_sampled_path(sim,np.array([sim.get_joint_positions()]),carry_cube=True)
        np.testing.assert_array_equal(sim.data.qpos,before)
        self.assertEqual(sim.model.neq,1)  # 唯一约束为原夹爪双指联动，没有物体绑定。

    def test_requested_opening_matches_collision_mesh_gap(self):
        sim=DesktopSim()
        sim.set_gripper(.06)
        sim.step(1200)
        rotation=sim.data.site_xmat[sim.tool_site].reshape(3,3)
        origin=sim.data.site_xpos[sim.tool_site]
        sides=[]
        for name in ('link7','link8'):
            body=mujoco.mj_name2id(sim.model,mujoco.mjtObj.mjOBJ_BODY,name)
            geom=next(int(g) for g in np.where(sim.model.geom_bodyid==body)[0] if sim.model.geom_contype[g])
            mesh=sim.model.geom_dataid[geom]
            a=sim.model.mesh_vertadr[mesh];n=sim.model.mesh_vertnum[mesh]
            world=sim.model.mesh_vert[a:a+n]@sim.data.geom_xmat[geom].reshape(3,3).T+sim.data.geom_xpos[geom]
            sides.append((world-origin)@rotation)
        gap=float(sides[1][:,1].min()-sides[0][:,1].max())
        self.assertAlmostEqual(gap,.06,delta=.0005)


if __name__=='__main__':
    unittest.main()
