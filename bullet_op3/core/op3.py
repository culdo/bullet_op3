import time
from threading import Thread
import numpy as np
from ..util import action_file_parser

np.set_printoptions(precision=2)

import pybullet as p
import pybullet_data
import sys

if sys.platform == "win32":
    from ctypes import windll

    timeBeginPeriod = windll.winmm.timeBeginPeriod
    timeBeginPeriod(1)

op3_joints = ('l_hip_yaw',
              'l_hip_roll',
              'l_hip_pitch',
              'l_knee',
              'l_ank_pitch',
              'l_ank_roll',
              'r_hip_yaw',
              'r_hip_roll',
              'r_hip_pitch',
              'r_knee',
              'r_ank_pitch',
              'r_ank_roll',
              'l_sho_pitch',
              'l_sho_roll',
              'l_el',
              'r_sho_pitch',
              'r_sho_roll',
              'r_el',
              'head_pan',
              'head_tilt')


class OP3:
    play_action = action_file_parser.play_action

    def __init__(self, fallen_reset=False, sim_speed=1.0, model_file="../models/robotis_op3.urdf",
                 op3StartOrientation=p.getQuaternionFromEuler([0, 0, 0]), joints=op3_joints):
        self.model_file = model_file
        self.fallen_reset = fallen_reset
        self.physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.sld_sim_speed = p.addUserDebugParameter("sim_speed", 1.0, 1000.0, sim_speed)
        self.bt_rst = p.addUserDebugParameter("reset OP3", 1, 0, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -9.8)
        self.op3StartPos = [0, 0, 0.3]
        self.op3StartOrientation = op3StartOrientation
        self.planeId = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(self.model_file, self.op3StartPos, self.op3StartOrientation)
        self.numJoints = p.getNumJoints(self.robot)
        self.targetVel = 0
        self.maxForce = 100

        self.camera_follow()
        self.angles = None
        self.update_angle_th()
        self.check_reset_th()
        # You can use real-time simulation rather than call stepSimulation
        # p.setRealTimeSimulation(1)
        self.run_sim_th()
        self.enable_joint()

        self.joints = joints

    @property
    def sim_speed(self):
        return p.readUserDebugParameter(self.sld_sim_speed)

    def get_orientation(self):
        _, orientation = p.getBasePositionAndOrientation(self.robot)
        return np.array(orientation)

    def get_position(self):
        position, _ = p.getBasePositionAndOrientation(self.robot)
        return np.array(position)

    def get_joint_xyz(self, joint_name):
        return p.getLinkState(self.robot, self.joints.index(joint_name))[0]

    def camera_follow(self, distance=1.0, pitch=-35.0, yaw=50.0):
        lookat = self.get_position() - [0, 0, 0.1]
        p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)

    def is_fallen(self):
        """Decide whether the rex has fallen.
        If the up directions between the base and the world is large (the dot
        product is smaller than 0.85), the robot is considered fallen.
        Returns:
          Boolean value that indicates whether the robot has fallen.
        """
        rot_mat = p.getMatrixFromQuaternion(self.get_orientation())
        local_up = rot_mat[6:]
        return np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85

    def get_angles(self):
        if self.joints is None: return None
        if self.angles is None: return None
        return dict(zip(self.joints, self.angles))

    def get_angle(self, joint):
        return self.get_angles()[joint]

    def set_angles(self, angles_info):
        for j, v in angles_info.items():
            if j not in self.joints:
                AssertionError("Invalid joint name " + j)
                continue
            p.setJointMotorControl(self.robot, self.joints.index(j), p.POSITION_CONTROL, v, self.maxForce)

    def set_angles_slow(self, stop_angles, delay=2):
        start_angles = self.get_angles()
        start = time.time()
        stop = start + delay
        while True:
            t = time.time()
            if t > stop: break
            ratio = (t - start) / (delay / self.sim_speed)
            angles = interpolate(stop_angles, start_angles, ratio)
            self.set_angles(angles)
            time.sleep(0.1 / self.sim_speed)

    def run_sim_th(self):
        def _cb_sim():
            while True:
                p.stepSimulation()
                time.sleep(1.0 / (240.0 * self.sim_speed))

        Thread(target=_cb_sim).start()

    def check_reset_th(self):
        def _cb_reset():
            self.prev_state = 1.0
            while True:
                curr_state = p.readUserDebugParameter(self.bt_rst)
                if curr_state != self.prev_state or (self.fallen_reset and self.is_fallen()):
                    self.reset()
                    self.prev_state = curr_state
                time.sleep(0.001)

        Thread(target=_cb_reset).start()

    def update_angle_th(self):
        def _cb_angles():
            while True:
                angles = []
                for joint in range(self.numJoints):
                    angles.append(p.getJointState(self.robot, joint)[0])
                self.angles = angles
                time.sleep(0.001)

        Thread(target=_cb_angles).start()

    def enable_joint(self):
        for joint in range(self.numJoints):
            print(p.getJointInfo(self.robot, joint))
            p.setJointMotorControl(self.robot, joint, p.POSITION_CONTROL, self.targetVel, self.maxForce)

    def loop(self):
        try:
            while True:
                # p.stepSimulation()
                time.sleep(1. / 240.)
        finally:
            OP3Pos, OP3Orn = p.getBasePositionAndOrientation(self.robot)
            print(OP3Pos, OP3Orn)
            p.disconnect()

    def reset(self):
        p.resetBasePositionAndOrientation(self.robot, self.op3StartPos, self.op3StartOrientation)
        time.sleep(0.5)


def interpolate(anglesa, anglesb, coefa):
    z = {}
    joints = anglesa.keys()
    for j in joints:
        z[j] = anglesa[j] * coefa + anglesb[j] * (1 - coefa)
    return z


if __name__ == '__main__':
    op3 = OP3()
    op3.loop()
