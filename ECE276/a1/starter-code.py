import time
import numpy as np
import pybullet as p
import pybullet_data

import pdb

if __name__ == '__main__': 
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0., 0., -9.81)
    plane = p.loadURDF('plane.urdf')
    robot = p.loadURDF('three-link-robot.urdf', useFixedBase = True)

    # get three movable joints and the end-effector link's index
    num_joints = p.getNumJoints(robot)
    movable_joints = []
    end_effector_link_idx = None
    for idx in range(num_joints): 
        info = p.getJointInfo(robot, idx)
        print('Joint {}: {}'.format(idx, info))

        joint_type = info[2]
        if joint_type != p.JOINT_FIXED: 
            movable_joints.append(idx)

        link_name = info[12].decode('utf-8')
        if link_name == 'end_effector': 
            end_effector_link_idx = idx

    # TODO: Your code here

    # hold Ctrl and use the mouse to rotate, pan, or zoom
    for _ in range(1200): 
        p.stepSimulation()
        time.sleep(1./240.)

    p.disconnect()
