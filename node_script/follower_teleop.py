#!/usr/bin/env python
import rospy
from node_utils import FollowJointTrajectoryClient, JointStateProvider


def main_loop():
    js_provider = JointStateProvider(namespace=None)
    fjt_client = FollowJointTrajectoryClient(namespace="follower")
    while not rospy.is_shutdown():
        joint_state = js_provider.joint_angles
        fjt_client.send_goal(joint_state, duration=0.2, blocking=False)
        rospy.sleep(0.05)


if __name__ == "__main__":
    rospy.init_node("follower_teleop")
    rospy.loginfo("Follower teleop node started")
    main_loop()
