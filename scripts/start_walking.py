import time
from bullet_op3.walker import Walker

walker = Walker(fallen_reset=True)
time.sleep(1)
walker.reset()

while True:
    time.sleep(1)
