#!/usr/bin/env python
PKG = 'pkg1'
import roslib; roslib.load_manifest(PKG)

import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats


import numpy as np 

def talker():
    pub = rospy.Publisher('floats', numpy_msg(Floats),queue_size=10)
    rospy.init_node('talker', anonymous=True)
    r = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        a = np.array([np.random.random(), np.random.random(), np.random.random()], dtype=np.float32)
        pub.publish(a)
        r.sleep()

if __name__ == '__main__':
    talker()