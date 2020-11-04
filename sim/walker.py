#!/usr/bin/env python
import time
from threading import Thread

import pybullet as p

import sys
from op3 import OP3
from wfunc import WFunc


class Walker:
    """
    Class for making Darwin walk
    """

    def __init__(self, op3):
        self.op3 = op3
        self.running = False

        self.velocity = [0, 0, 0]
        self.walking = False
        self.wfunc = WFunc()

        # ~ self.ready_pos=get_walk_angles(10)
        self.ready_pos = self.wfunc.get(True, 0, [0, 0, 0])
        self.ready_pos.update({"r_sho_pitch": 0, "l_sho_pitch": 0,
                               "r_sho_roll": -1.0, "l_sho_roll": 1.0,
                               "r_el": 0.5, "l_el": -0.5})
        if sys.platform == "win32":
            self.sld_interval = p.addUserDebugParameter("step_interval", 0.001, 0.1, 0.01)
        elif sys.platform == "linux":
            self.sld_interval = p.addUserDebugParameter("step_interval", 0.001, 0.1, 0.01)
        self._th_walk = None

    def cmd_vel(self, vx, vy, vt):
        print("cmdvel", (vx, vy, vt))
        self.start()
        self.set_velocity(vx, vy, vt)

    def init_walk(self):
        """
        If not there yet, go to initial walk position
        """
        if self.get_dist_to_ready() > 0.02:
            self.op3.set_angles_slow(self.ready_pos)

    def start(self):
        if not self.running:
            self.running = True
            self.init_walk()
            self._th_walk = Thread(target=self._do_walk)
            self._th_walk.start()
            self.walking = True

    def stop(self):
        if self.running:
            self.walking = False
            self.running = False

    def set_velocity(self, x, y, t):
        self.velocity = [x, y, t]

    def _do_walk(self):
        """
        Main walking loop, smoothly update velocity vectors and apply corresponding angles
        """

        # Global walk loop
        n = 50
        phrase = True
        i = 0
        self.current_velocity = [0, 0, 0]
        while self.walking or i < n or self.is_walking():
            if not self.walking:
                self.velocity = [0, 0, 0]
            if not self.is_walking() and i == 0:  # Do not move if nothing to do and already at 0
                self.update_velocity(self.velocity, n)
                time.sleep(p.readUserDebugParameter(self.sld_interval))
                continue
            x = float(i) / n
            angles = self.wfunc.get(phrase, x, self.current_velocity)
            self.update_velocity(self.velocity, n)
            self.op3.set_angles(angles)
            i += 1
            if i > n:
                i = 0
                phrase = not phrase
            time.sleep(p.readUserDebugParameter(self.sld_interval))
            self.op3.camera_follow(0.5, 0, 0)

        self._th_walk = None

    def is_walking(self):
        e = 0.02
        for v in self.current_velocity:
            if abs(v) > e: return True
        return False

    def rescale(self, angles, coef):
        z = {}
        for j, v in angles.items():
            offset = self.ready_pos[j]
            v -= offset
            v *= coef
            v += offset
            z[j] = v
        return z

    def update_velocity(self, target, n):
        a = 3 / float(n)
        b = 1 - a
        self.current_velocity = [a * t + b * v for (t, v) in zip(target, self.current_velocity)]

    def get_dist_to_ready(self):
        angles = self.op3.get_angles()
        return get_distance(self.ready_pos, angles)


def interpolate(anglesa, anglesb, coefa):
    z = {}
    joints = anglesa.keys()
    for j in joints:
        z[j] = anglesa[j] * coefa + anglesb[j] * (1 - coefa)
    return z


def get_distance(anglesa, anglesb):
    d = 0
    joints = anglesa.keys()
    if len(joints) == 0: return 0
    for j in joints:
        d += abs(anglesb[j] - anglesa[j])
    d /= len(joints)
    return d


if __name__ == "__main__":

    op3 = OP3()
    walker = Walker(op3)
    time.sleep(1)
    walker.start()
    walker.set_velocity(1, 0, 0)

    while True:
        time.sleep(1)
