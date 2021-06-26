import os
import time
from threading import Thread

from bullet_op3.core.op3 import OP3, p
import numpy as np

op3_arm_joints = ('l_hip_yaw',
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
                  'l_ha',
                  'r_sho_pitch',
                  'r_sho_roll',
                  'r_el',
                  'head_pan',
                  'head_tilt')


class OP3Arm(OP3):
    def __init__(self):
        super().__init__(model_file="../bullet_op3/data/models/robotis_op3_arm.urdf",
                         op3StartOrientation=p.getQuaternionFromEuler([0, 0.1, 0]), joints=op3_arm_joints)
        self.ball = p.loadURDF("../bullet_op3/data/models/sphere.urdf", [0, 0, 1], self.op3StartOrientation)
        self.l_ha_pos = None
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
                        p.setJointMotorControl(self.robot, 12 + i, p.POSITION_CONTROL, curr, self.maxForce)
                        self.l_ha_pos = p.getLinkState(self.robot, 15)[0]
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

                    data_arr.append([self.curr_list, self.l_ha_pos])
                    np.save("data/arm_data.npy", data_arr)
                    self.data_state = curr_state
                time.sleep(0.001)

        Thread(target=_cb_save_data).start()

    def run_predict(self, stop=40):
        answers = np.load("data/answers.npy")
        algo_names = ["svr", "dtr", "rfr"]
        op3.play_action()
        op3.reset()
        y_data = np.zeros([3, 3, 41, 3])
        for i, data in enumerate(answers):
            print("Goal %s" % i)
            goal = data[0]
            p.resetBasePositionAndOrientation(self.ball, goal, self.op3StartOrientation)
            algos = data[1:]
            # I was a idiot that hide the 8 scale factor in original mstp_op3_arm DDPG file.
            start_goal_vector = self.get_joint_xyz("l_ha") - goal
            print("Goal %s before_goal_vec: %s" % (i, start_goal_vector))
            start_ang = np.array([self.get_angle("l_sho_pitch"),
                                  self.get_angle("l_sho_roll"),
                                  self.get_angle("l_el")])
            for j, (name, move) in enumerate(zip(algo_names, algos)):
                y_data[i, j, 0] = start_goal_vector
                print("Algo %s" % name)
                move_diff = (move - start_ang) / stop
                for step in range(stop):
                    move_offset = start_ang + move_diff * (step + 1)
                    self.set_angles({"l_sho_pitch": move_offset[0],
                                     "l_sho_roll": move_offset[1],
                                     "l_el": move_offset[2]})
                    time.sleep(0.1)
                    # Same above
                    goal_vector = self.get_joint_xyz("l_ha") - goal
                    y_data[i, j, step+1] = goal_vector
                    print("Goal %s after_goal_vec: %s" % (i, goal_vector))
                op3.play_action()
                op3.reset()
        np.save("../../MSTP_op3_arm/logs/SVM/predict_state_test.npy", y_data[:, 0])
        np.save("../../MSTP_op3_arm/logs/DecisionTree/predict_state_test.npy", y_data[:, 1])
        np.save("../../MSTP_op3_arm/logs/RandomForest/predict_state_test.npy", y_data[:, 2])


if __name__ == '__main__':
    op3 = OP3Arm()
    op3.run_predict()
    # op3.loop()
