#!/usr/bin/env python
import select
import sys
import termios
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import rospy
import yaml
from geometry_msgs.msg import Pose, PoseStamped
from moveit_commander import (
    MoveGroupCommander,
    PlanningSceneInterface,
    roscpp_initialize,
)
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from node_utils import FollowJointTrajectoryClient, JointStateProvider
from rospkg import RosPack
from skrobot.coordinates import Coordinates, matrix2quaternion


class IKClient:
    """Computes inverse kinematics using the GetPositionIK service."""

    def __init__(self, ik_link_name: str, ik_timeout: float):
        rospy.wait_for_service("/compute_ik")
        compute_ik = rospy.ServiceProxy("/compute_ik", GetPositionIK, persistent=True)
        self.compute_ik_call = compute_ik

        request = GetPositionIKRequest()
        request.ik_request.avoid_collisions = True
        request.ik_request.group_name = "arm"
        request.ik_request.ik_link_name = ik_link_name
        request.ik_request.robot_state.joint_state.name = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]
        request.ik_request.pose_stamped.header.frame_id = "base_link"
        request.ik_request.timeout = rospy.Duration(ik_timeout)
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
            rospy.logdebug("IK solution found")
            return np.array(response.solution.joint_state.position)
        rospy.logdebug("IK solution not found")
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
        configuration_jump_threshold: float,
        teleop_command_eps: float,
        send_goal_time: float,
    ):
        self.coords = coords.copy_worldcoords()
        self.js_provider = js_provider
        self.ik_client = ik_client
        self.fjt_client = fjt_client
        self.configuration_jump_threshold = configuration_jump_threshold
        self.teleop_command_eps = teleop_command_eps
        self.send_goal_time = send_goal_time

    def run(self):
        rospy.loginfo("Press arrow keys, Enter, Backspace, or 'q' to quit.")
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            new_settings = termios.tcgetattr(fd)
            new_settings[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(fd, termios.TCSANOW, new_settings)

            while not rospy.is_shutdown():
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    ch = sys.stdin.read(1)
                    coords_prev = self.coords.copy_worldcoords()

                    if ch == "\x1b":
                        seq = sys.stdin.read(2)
                        if seq == "[A":
                            self.coords.translate([0.0, 0.0, self.teleop_command_eps])
                        elif seq == "[B":
                            self.coords.translate([0.0, 0.0, -self.teleop_command_eps])
                        elif seq == "[C":
                            self.coords.translate([self.teleop_command_eps, 0.0, 0.0])
                        elif seq == "[D":
                            self.coords.translate([-self.teleop_command_eps, 0.0, 0.0])
                    elif ch in ("\r", "\n"):
                        self.coords.translate([0.0, self.teleop_command_eps, 0.0])
                    elif ch == "\x7f":
                        self.coords.translate([0.0, -self.teleop_command_eps, 0.0])
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
                        # Use the configured send_goal_time instead of a hard-coded value.
                        self.fjt_client.send_goal(new_q, self.send_goal_time, blocking=False)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def send_initial_trajectory(
    fjt_client: FollowJointTrajectoryClient,
    ik_client: IKClient,
    home_coords: dict,
    home_coords_ik_seed: list,
) -> (np.ndarray, Coordinates):
    # Create a Coordinates object from the config values.
    coords_obj = Coordinates(home_coords["position"], home_coords["orientation"])
    q_solution = ik_client.compute_ik(
        skcoords_to_rospose(coords_obj), np.array(home_coords_ik_seed)
    )
    if q_solution is None:
        rospy.logerr("Initial IK computation failed.")
        sys.exit(1)
    fjt_client.send_goal(q_solution, 3.0, blocking=True)
    return q_solution, coords_obj


def plan_and_execute(q_solution: np.ndarray):
    move_group = MoveGroupCommander("arm")
    move_group.set_joint_value_target(q_solution)
    plan_success, plan_traj, _, _ = move_group.plan()
    if plan_success and not rospy.is_shutdown():
        move_group.execute(plan_traj, wait=True)
        rospy.loginfo("Execution finished.")


@dataclass
class Config:
    control_frame_name: str
    ik_timeout: float
    home_coords: Dict[str, Any]
    home_coords_ik_seed: List[float]
    configuration_jump_threshold: float
    teleop_command_eps: float
    send_goal_time: float

    @staticmethod
    def from_yaml(path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return Config(**data)


if __name__ == "__main__":
    rospy.init_node("mycobot_teleop", anonymous=False)
    roscpp_initialize(sys.argv)

    pkg_path = Path(RosPack().get_path("mycobot_teleop"))
    conf_path = pkg_path.resolve() / "config" / "leader.yaml"
    config = Config.from_yaml(conf_path)

    scene = PlanningSceneInterface(synchronous=True)
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = "base_link"
    pose.pose.position.z = 0.005  # table safety margin
    scene.add_plane("table_plane", pose)

    fjt_client = FollowJointTrajectoryClient()
    ik_client = IKClient(ik_link_name=config.control_frame_name, ik_timeout=config.ik_timeout)
    js_provider = JointStateProvider()
    q_solution, init_coords = send_initial_trajectory(
        fjt_client, ik_client, config.home_coords, config.home_coords_ik_seed
    )
    plan_and_execute(q_solution)

    kcl = KeyboardControlLoop(
        init_coords,
        js_provider,
        ik_client,
        fjt_client,
        configuration_jump_threshold=config.configuration_jump_threshold,
        teleop_command_eps=config.teleop_command_eps,
        send_goal_time=config.send_goal_time,
    )
    kcl.run()
