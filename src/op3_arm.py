import os
import time
from threading import Thread

from src.core.op3 import OP3, p
import numpy as np


class OP3Arm(OP3):
    def __init__(self):
        super().__init__(model_file="../models/robotis_op3_arm.urdf",
                         op3StartOrientation=p.getQuaternionFromEuler([0, 0.1, 0]))
        self.goal_label = None
        self.l_sho_pitch = p.addUserDebugParameter("l_sho_pitch", -3.14, 3.14, 0)
        self.l_sho_roll = p.addUserDebugParameter("l_sho_roll", -3.14, 3.14, 0)
        self.l_el = p.addUserDebugParameter("l_el", -3.14, 3.14, 0)
        self.save_data = p.addUserDebugParameter("Save data", 1, 0, 1)
        self.prev_list = [None, None, None]

        self.check_joint_th()
        self.save_data_th()

    def read_all_data(self):
        curr_pitch = p.readUserDebugParameter(self.l_sho_pitch)
        curr_roll = p.readUserDebugParameter(self.l_sho_roll)
        curr_el = p.readUserDebugParameter(self.l_el)
        self.curr_list = [curr_pitch, curr_roll, curr_el]

    def check_joint_th(self):
        def _cb_joint():
            while True:
                self.read_all_data()

                for i, (curr, prev) in enumerate(zip(self.curr_list, self.prev_list)):
                    if curr != prev:
                        self.prev_list[i] = curr
                        p.setJointMotorControl(self.robot, 12+i, p.POSITION_CONTROL, curr, self.maxForce)
                        self.goal_label = p.getLinkState(self.robot, 15)[0]
                        print(self.goal_label)
                time.sleep(0.01)

        Thread(target=_cb_joint).start()

    def save_data_th(self):
        def _cb_save_data():
            self.data_state = 1.0
            while True:
                curr_state = p.readUserDebugParameter(self.save_data)
                if curr_state != self.data_state:
                    data_arr = []
                    if os.path.exists("data/arm_data.npy"):
                        data_arr = np.load("data/arm_data.npy").tolist()

                    data_arr.append([self.curr_list, self.goal_label])
                    np.save("data/arm_data.npy", data_arr)
                    self.data_state = curr_state
                time.sleep(0.001)

        Thread(target=_cb_save_data).start()

    def run_predict(self):
        answers = np.load("data/answers.npy")
        for goal in answers:
            for label, xyz_in_algo in goal:
                goal_vector = label - p.getLinkState(self.robot, 15)[0]
                self.set_angles({"l_sho_pitch": xyz_in_algo[0],
                                 "l_sho_roll": xyz_in_algo[1],
                                 "l_el": xyz_in_algo[2]})


if __name__ == '__main__':
    op3 = OP3Arm()
    op3.play_action()
    op3.run()
