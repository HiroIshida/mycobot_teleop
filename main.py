import select
import sys
import termios
from typing import Optional

import actionlib
import numpy as np
import rospy
import tf
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from geometry_msgs.msg import Pose
from moveit_commander import MoveGroupCommander, roscpp_initialize
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from sensor_msgs.msg import JointState
from skrobot.coordinates import Coordinates, matrix2quaternion
from trajectory_msgs.msg import JointTrajectoryPoint


class CoordinatesProvider:
    """Provides coordinate transformations using tf."""

    def __init__(self):
        self.listener = tf.TransformListener()
        rospy.sleep(1.0)

    def get_link6_flange_coords(self) -> Optional[Coordinates]:
        try:
            self.listener.waitForTransform(
                "base_link", "link6_flange", rospy.Time(0), rospy.Duration(4.0)
            )
            trans, quat = self.listener.lookupTransform("base_link", "link6_flange", rospy.Time(0))
            return Coordinates(trans, quat, input_quaternion_order="xyzw")
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Failed to get transform from base_link to link6_flange")
            return None


class JointStateProvider:
    """Subscribes to joint states and provides the latest joint angles."""

    def __init__(self):
        self._joint_angles = None
        rospy.Subscriber("/joint_states", JointState, self.callback)

    def callback(self, msg: JointState):
        self._joint_angles = np.array(msg.position)

    @property
    def joint_angles(self) -> np.ndarray:
        while self._joint_angles is None and not rospy.is_shutdown():
            rospy.sleep(0.05)
        return self._joint_angles


class FollowJointTrajectoryClient:
    """Sends joint trajectory goals to the action server."""

    def __init__(self):
        self.client = actionlib.SimpleActionClient(
            "/arm_controller/follow_joint_trajectory", FollowJointTrajectoryAction
        )
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
        rospy.loginfo("Sent goal to action server")
        if blocking:
            self.client.wait_for_result()
            result = self.client.get_result()
            rospy.loginfo("Action finished with result: %s", result)


class IKClient:
    """Computes inverse kinematics using the GetPositionIK service."""

    def __init__(self):
        rospy.wait_for_service("/compute_ik")
        compute_ik = rospy.ServiceProxy("/compute_ik", GetPositionIK, persistent=True)
        self.compute_ik_call = compute_ik

        request = GetPositionIKRequest()
        request.ik_request.group_name = "arm"
        request.ik_request.ik_link_name = "link6_flange"
        request.ik_request.robot_state.joint_state.name = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]
        request.ik_request.pose_stamped.header.frame_id = "base_link"
        request.ik_request.timeout = rospy.Duration(0.02)
        request.ik_request.pose_stamped.header.stamp = rospy.Time.now()
        self.request = request

    def compute_ik(self, pose: Pose, q: np.ndarray) -> Optional[np.ndarray]:
        self.request.ik_request.robot_state.joint_state.position = list(q)
        self.request.ik_request.pose_stamped.pose = pose
        try:
            response = self.compute_ik_call(self.request)
        except rospy.ServiceException:
            rospy.logerr("Service call failed")
            return None

        if response.error_code.val == 1:
            rospy.loginfo("IK solution found")
            return np.array(response.solution.joint_state.position)
        return None


def skcoords_to_rospose(coords: Coordinates) -> Pose:
    """Converts skrobot Coordinates to a ROS Pose message."""
    position = coords.worldpos()
    quat_wxyz = matrix2quaternion(coords.worldrot())
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = position
    pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = quat_wxyz
    return pose


class KeyboardControlLoop:
    def __init__(
        self,
        coords: Coordinates,
        js_provider: JointStateProvider,
        ik_client: IKClient,
        fjt_client: FollowJointTrajectoryClient,
        configuration_jump_threshold: float = 0.3,
        eps: float = 0.001,
    ):
        self.coords = coords.copy_worldcoords()
        self.js_provider = js_provider
        self.ik_client = ik_client
        self.fjt_client = fjt_client
        self.configuration_jump_threshold = configuration_jump_threshold
        self.eps = eps

    def run(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            new_settings = termios.tcgetattr(fd)
            new_settings[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(fd, termios.TCSANOW, new_settings)
            print("Press arrow keys, Enter, Backspace, or 'q' to quit.", flush=True)

            while not rospy.is_shutdown():
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    ch = sys.stdin.read(1)
                    coords_prev = self.coords.copy_worldcoords()

                    if ch == "\x1b":
                        seq = sys.stdin.read(2)
                        if seq == "[A":
                            self.coords.translate([0.0, 0.0, self.eps])
                        elif seq == "[B":
                            self.coords.translate([0.0, 0.0, -self.eps])
                        elif seq == "[C":
                            self.coords.translate([self.eps, 0.0, 0.0])
                        elif seq == "[D":
                            self.coords.translate([-self.eps, 0.0, 0.0])
                    elif ch in ("\r", "\n"):
                        self.coords.translate([0.0, self.eps, 0.0])
                    elif ch == "\x7f":
                        self.coords.translate([0.0, -self.eps, 0.0])
                    elif ch == "q":
                        break

                    rospy.logdebug(f"Key pressed: {ch}")
                    rospy.logdebug(f"Current Coordinates: {self.coords}")
                    current_joint_state = self.js_provider.joint_angles
                    new_q = self.ik_client.compute_ik(
                        skcoords_to_rospose(self.coords), current_joint_state
                    )
                    rospy.logdebug(f"IK solution: {new_q}")

                    if new_q is None:
                        rospy.logwarn("IK computation failed.")
                        self.coords = coords_prev
                        continue

                    if (
                        np.max(np.abs(new_q - current_joint_state))
                        > self.configuration_jump_threshold
                    ):
                        rospy.logwarn("L1 norm exceeded, not allowing jump")
                        self.coords = coords_prev
                    else:
                        self.fjt_client.send_goal(new_q, 0.5, blocking=False)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def send_initial_trajectory(
    fjt_client: FollowJointTrajectoryClient, ik_client: IKClient
) -> (np.ndarray, Coordinates):
    init_coords = Coordinates([0.163, -0.037, 0.22], [-0.5 * np.pi, 0.0, -0.5 * np.pi])
    init_q = np.array([-2.97, 0.026, -1.92, 0.33, 0, 0.16])
    q_solution = ik_client.compute_ik(skcoords_to_rospose(init_coords), init_q)
    if q_solution is None:
        rospy.logerr("Initial IK computation failed.")
        sys.exit(1)
    fjt_client.send_goal(q_solution, 3.0, blocking=True)
    return q_solution, init_coords


def plan_and_execute(q_solution: np.ndarray):
    move_group = MoveGroupCommander("arm")
    move_group.set_joint_value_target(q_solution)
    plan_success, plan_traj, _, _ = move_group.plan()
    if plan_success and not rospy.is_shutdown():
        move_group.execute(plan_traj, wait=True)
        rospy.loginfo("Execution finished.")


if __name__ == "__main__":
    rospy.init_node("get_link6_flange_transform", anonymous=True)
    roscpp_initialize(sys.argv)
    fjt_client = FollowJointTrajectoryClient()
    ik_client = IKClient()
    q_solution, init_coords = send_initial_trajectory(fjt_client, ik_client)
    plan_and_execute(q_solution)

    js_provider = JointStateProvider()
    kcl = KeyboardControlLoop(
        init_coords, js_provider, ik_client, fjt_client, configuration_jump_threshold=0.3, eps=0.001
    )
    kcl.run()
