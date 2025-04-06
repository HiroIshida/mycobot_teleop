from typing import Optional

import actionlib
import numpy as np
import rospy
import tf
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from sensor_msgs.msg import JointState
from skrobot.coordinates import Coordinates
from trajectory_msgs.msg import JointTrajectoryPoint


class CoordinatesProvider:
    """Provides coordinate transformations using tf."""

    def __init__(self, link_name: str):
        self.link_name = link_name
        self.listener = tf.TransformListener()
        rospy.sleep(1.0)

    def get_link_coords(self) -> Optional[Coordinates]:
        try:
            self.listener.waitForTransform(
                "base_link", self.link_name, rospy.Time(0), rospy.Duration(4.0)
            )
            trans, quat = self.listener.lookupTransform("base_link", self.link_name, rospy.Time(0))
            return Coordinates(trans, quat, input_quaternion_order="xyzw")
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Failed to get transform from base_link to %s", self.link_name)
            return None


class JointStateProvider:
    """Subscribes to joint states and provides the latest joint angles."""

    def __init__(self, namespace: Optional[str] = None):
        self._joint_angles = None
        topic = "/joint_states" if namespace is None else f"/{namespace}/joint_states"
        rospy.Subscriber(topic, JointState, self.callback)

    def callback(self, msg: JointState):
        self._joint_angles = np.array(msg.position)

    @property
    def joint_angles(self) -> np.ndarray:
        while self._joint_angles is None and not rospy.is_shutdown():
            rospy.sleep(0.05)
        return self._joint_angles


class FollowJointTrajectoryClient:
    """Sends joint trajectory goals to the action server."""

    def __init__(self, namespace: Optional[str] = None):
        if namespace is None:
            topic = "/arm_controller/follow_joint_trajectory"
        else:
            topic = f"/{namespace}/arm_controller/follow_joint_trajectory"
        self.client = actionlib.SimpleActionClient(topic, FollowJointTrajectoryAction)
        rospy.loginfo("Waiting for action server to start...")
        self.client.wait_for_server()
        rospy.loginfo("Action server started, sending goal.")
        self.goal = FollowJointTrajectoryGoal()
        self.goal.trajectory.joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]
        point = JointTrajectoryPoint()
        point.velocities = [0.0] * 6
        self.goal.trajectory.points.append(point)

    def send_goal(self, q: np.ndarray, duration: float, blocking: bool = True):
        self.goal.trajectory.points[0].positions = list(q)
        self.goal.trajectory.points[0].time_from_start = rospy.Duration(duration)
        self.goal.trajectory.header.stamp = rospy.Time.now()
        self.client.send_goal(self.goal)
        rospy.logdebug("Sent goal to action server")
        if blocking:
            self.client.wait_for_result()
            result = self.client.get_result()
            rospy.loginfo("Action finished with result: %s", result)
