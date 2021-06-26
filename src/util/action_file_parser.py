import time
from ctypes import *
from math import pi

action_joints = [
    'r_sho_pitch',
    'l_sho_pitch',
    'r_sho_roll',
    'l_sho_roll',
    'r_el',
    'l_el',
    'r_hip_yaw',
    'l_hip_yaw',
    'r_hip_roll',
    'l_hip_roll',
    'r_hip_pitch',
    'l_hip_pitch',
    'r_knee',
    'l_knee',
    'r_ank_pitch',
    'l_ank_pitch',
    'r_ank_roll',
    'l_ank_roll',
    'head_pan',
    'head_tilt']


class PageHeader(Structure):
    _fields_ = [('name', c_ubyte * 14),
                ('reserved1', c_ubyte),
                ('repeat', c_ubyte),
                ('schedule', c_ubyte),
                ('reserved2', c_ubyte * 3),
                ('stepnum', c_ubyte),
                ('reserved3', c_ubyte),
                ('speed', c_ubyte),
                ('reserved4', c_ubyte),
                ('accel', c_ubyte),
                ('next', c_ubyte),
                ('exit', c_ubyte),
                ('reserved5', c_ubyte * 4),
                ('checksum', c_ubyte),
                ('pgain', c_ubyte * 31),
                ('reserved6', c_ubyte)]


class Step(Structure):
    _fields_ = [('position', c_ushort * 31),
                ('pause', c_ubyte),
                ('time', c_ubyte)]


class Page(Structure):
    _fields_ = [('header', PageHeader),
                ('steps', Step * 7)]


class ActionFile(Structure):
    _fields_ = [('pages', Page * 256)]


def cvt_4095_to_rad(x):
    return (x - 2048) * pi / 2048.0


with open('data/motion_4095.bin', 'rb') as file:
    result = []
    x = ActionFile()
    a = bytearray()
    assert file.readinto(x) == sizeof(x)


def play_action(self, page_num=1):
    page = x.pages[page_num]
    print("Page %s %s" % (page_num, cast(page.header.name, c_char_p).value))
    for j, step in enumerate(page.steps[:page.header.stepnum]):
        angles = []
        for k, joint_val in enumerate(step.position):
            if len(action_joints) + 1 > k > 0:
                rad = cvt_4095_to_rad(joint_val)
                angles.append(rad)
        angle_dict = dict(zip(action_joints, angles))
        self.set_angles(angle_dict)
        if step.pause*8/1000 > 0:
            time.sleep(step.pause*8/1000)
        else:
            time.sleep(2)
