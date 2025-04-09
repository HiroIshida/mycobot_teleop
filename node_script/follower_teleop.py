#!/usr/bin/env python
import rospy
from node_utils import FollowJointTrajectoryClient, JointStateProvider


def main_loop():
    js_provider_leader = JointStateProvider(namespace=None)
    js_provider_follower = JointStateProvider(namespace="follower")
    fjt_client = FollowJointTrajectoryClient(namespace="follower")
    while not rospy.is_shutdown():
        q_leader = js_provider_leader.joint_angles
        q_follower = js_provider_follower.joint_angles
        q_goal = q_leader + (q_leader - q_follower) * 1.0
        fjt_client.send_goal(q_goal, duration=0.5, blocking=False)
        rospy.sleep(0.05)


if __name__ == "__main__":
    rospy.init_node("follower_teleop")
    rospy.loginfo("Follower teleop node started")
    main_loop()
