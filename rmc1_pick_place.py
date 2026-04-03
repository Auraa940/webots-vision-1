#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rmc1_pick_place.py

Пример запуска:
python3 rmc1_pick_place.py \
  --model /home/user/runs/rmc1/yolo_boxes/weights/best.pt \
  --target 1 \
  --imgsz 640 \
  --conf 0.55

target:
1 - hammer
2 - wrench
3 - pliers

ВАЖНО:
1) Перед запуском откалибруй IMAGE_POINTS и BASE_POINTS_XY ниже.
2) Подстрой SAFE_Z, PICK_Z, PLACE_Z, HOME_POSE, PLACE_POINT_XY.
3) Скрипт рассчитан на работу в мире module3 и с запущенным MoveIt для ARM95.
"""

import argparse
import math
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf_transformations import quaternion_from_euler
from ultralytics import YOLO

from moveit.planning import MoveItPy


# -----------------------------
# НАСТРОЙКИ, КОТОРЫЕ НУЖНО ПОДОГНАТЬ ПОД ТВОЙ СТЕНД
# -----------------------------

CLASS_NAMES = ["hammer", "wrench", "pliers"]
TARGET_ID_TO_CLASS = {1: 0, 2: 1, 3: 2}

# 4 точки на изображении (пиксели) — например, углы рабочей зоны полки
# Заполни своими значениями после калибровки
IMAGE_POINTS = np.array([
    [120.0, 100.0],   # top-left
    [520.0, 100.0],   # top-right
    [520.0, 420.0],   # bottom-right
    [120.0, 420.0],   # bottom-left
], dtype=np.float32)

# Те же 4 точки, но уже в координатах Base_link манипулятора (x, y), метры
# Заполни своими значениями после калибровки
BASE_POINTS_XY = np.array([
    [-0.32, -0.62],   # top-left
    [-0.08, -0.62],   # top-right
    [-0.08, -0.32],   # bottom-right
    [-0.32, -0.32],   # bottom-left
], dtype=np.float32)

# Домашняя поза манипулятора (пример, подстрой под свой стенд)
HOME_POSE = (-0.20, -0.50, 0.70)

# Точка выкладки на пустое место на полке
PLACE_POINT_XY = (-0.12, -0.34)

# Высоты, которые нужно обязательно проверить в RViz/Webots
SAFE_Z = 0.42
PICK_Z = 0.23
PLACE_Z = 0.23

# Границы допустимой рабочей зоны
X_LIMITS = (-0.45, 0.05)
Y_LIMITS = (-0.75, -0.15)

# Стабилизация детекции
STABLE_FRAMES = 5
STABLE_STD_METERS = 0.008
TARGET_FRESHNESS_SEC = 1.0

# Ориентация схвата: вниз, как в примере MoveIt
ROLL = math.pi
PITCH = 0.0
YAW = 1.57


class RMC1PickPlaceNode(Node):
    def __init__(self, model_path: str, target_id: int, imgsz: int, conf_threshold: float):
        super().__init__("rmc1_pick_place_node")

        if target_id not in TARGET_ID_TO_CLASS:
            raise ValueError("target_id должен быть 1, 2 или 3")

        self.model_path = model_path
        self.target_id = target_id
        self.target_class = TARGET_ID_TO_CLASS[target_id]
        self.target_name = CLASS_NAMES[self.target_class]
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold

        self.bridge = CvBridge()
        self.detector = YOLO(self.model_path)

        self.robot = MoveItPy(node_name="moveit_py", name_space="/RMC1/arm95")
        self.arm = self.robot.get_planning_component("arm95_group")
        self.gripper = self.robot.get_planning_component("gripper")

        self.history_xy = deque(maxlen=STABLE_FRAMES)
        self.history_uv = deque(maxlen=STABLE_FRAMES)
        self.last_detection_time = 0.0

        self.busy = False
        self.finished = False

        self.H = self._compute_homography()

        self.image_sub = self.create_subscription(
            Image,
            "/RMC1/arm95/camera_gripper/image_color",
            self.image_callback,
            10,
        )

        self.control_timer = self.create_timer(0.2, self.control_loop)

        self.get_logger().info(f"Loaded model: {self.model_path}")
        self.get_logger().info(
            f"Target: {self.target_id} -> {self.target_name} (class={self.target_class})"
        )

    def _compute_homography(self) -> np.ndarray:
        if IMAGE_POINTS.shape != (4, 2) or BASE_POINTS_XY.shape != (4, 2):
            raise ValueError("IMAGE_POINTS и BASE_POINTS_XY должны быть формы (4, 2)")

        H, status = cv2.findHomography(IMAGE_POINTS, BASE_POINTS_XY)
        if H is None or status is None:
            raise RuntimeError("Не удалось вычислить гомографию")
        return H

    def pixel_to_base_xy(self, u: float, v: float) -> Tuple[float, float]:
        pt = np.array([[[u, v]]], dtype=np.float32)
        xy = cv2.perspectiveTransform(pt, self.H)[0, 0]
        return float(xy[0]), float(xy[1])

    def in_workspace(self, x: float, y: float) -> bool:
        return X_LIMITS[0] <= x <= X_LIMITS[1] and Y_LIMITS[0] <= y <= Y_LIMITS[1]

    def image_callback(self, msg: Image) -> None:
        if self.busy or self.finished:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        try:
            result = self.detector.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                verbose=False,
                iou=0.5,
                device="cpu",
            )[0]
        except Exception as e:
            self.get_logger().error(f"Model inference failed: {e}")
            return

        if result.boxes is None or len(result.boxes) == 0:
            return

        best_candidate = None
        best_score = -1.0

        for box in result.boxes:
            try:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                xyxy = box.xyxy[0].cpu().numpy()
            except Exception:
                continue

            if cls_id != self.target_class:
                continue

            x1, y1, x2, y2 = xyxy
            u = float((x1 + x2) / 2.0)
            v = float((y1 + y2) / 2.0)

            x_base, y_base = self.pixel_to_base_xy(u, v)
            if not self.in_workspace(x_base, y_base):
                continue

            # Можно усилить отбор по conf
            score = conf
            if score > best_score:
                best_score = score
                best_candidate = (u, v, x_base, y_base, conf)

        if best_candidate is None:
            return

        u, v, x_base, y_base, conf = best_candidate
        self.history_uv.append((u, v))
        self.history_xy.append((x_base, y_base))
        self.last_detection_time = time.time()

        self.get_logger().info(
            f"Detected {self.target_name}: "
            f"u={u:.1f}, v={v:.1f}, x={x_base:.3f}, y={y_base:.3f}, conf={conf:.2f}"
        )

    def control_loop(self) -> None:
        if self.busy or self.finished:
            return

        if len(self.history_xy) < STABLE_FRAMES:
            return

        if time.time() - self.last_detection_time > TARGET_FRESHNESS_SEC:
            return

        xy = np.array(self.history_xy, dtype=np.float32)
        mean_xy = xy.mean(axis=0)
        std_xy = xy.std(axis=0)

        if float(std_xy[0]) > STABLE_STD_METERS or float(std_xy[1]) > STABLE_STD_METERS:
            self.get_logger().info(
                f"Target not stable yet: std_x={std_xy[0]:.4f}, std_y={std_xy[1]:.4f}"
            )
            return

        x_target = float(mean_xy[0])
        y_target = float(mean_xy[1])

        self.get_logger().info(
            f"Stable target locked: x={x_target:.3f}, y={y_target:.3f}. Starting pick-and-place."
        )

        self.busy = True
        success = False

        try:
            success = self.execute_pick_and_place(x_target, y_target)
        except Exception as e:
            self.get_logger().error(f"Pick-and-place exception: {e}")
            success = False
        finally:
            self.busy = False

        if success:
            self.finished = True
            self.get_logger().info("Task finished successfully. Shutting down.")
            rclpy.shutdown()
        else:
            self.get_logger().error("Task failed. Detection history cleared; waiting for new stable target.")
            self.history_xy.clear()
            self.history_uv.clear()

    def plan_and_execute(self, planning_component, sleep_time: float = 0.0) -> bool:
        plan_result = planning_component.plan()
        if plan_result:
            self.get_logger().info("Executing plan")
            robot_trajectory = plan_result.trajectory
            self.robot.execute(robot_trajectory, controllers=[])
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            return True

        self.get_logger().error("Planning failed")
        return False

    def move_to_pose(self, x: float, y: float, z: float) -> bool:
        q = quaternion_from_euler(ROLL, PITCH, YAW)

        pose_goal = PoseStamped()
        pose_goal.header.frame_id = "Base_link"
        pose_goal.pose.orientation.x = float(q[0])
        pose_goal.pose.orientation.y = float(q[1])
        pose_goal.pose.orientation.z = float(q[2])
        pose_goal.pose.orientation.w = float(q[3])
        pose_goal.pose.position.x = float(x)
        pose_goal.pose.position.y = float(y)
        pose_goal.pose.position.z = float(z)

        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link="gripper_base")

        self.get_logger().info(f"Move to pose: x={x:.3f}, y={y:.3f}, z={z:.3f}")
        return self.plan_and_execute(self.arm, sleep_time=0.2)

    def open_gripper(self) -> bool:
        self.gripper.set_goal_state(configuration_name="open")
        self.get_logger().info("Open gripper")
        return self.plan_and_execute(self.gripper, sleep_time=0.2)

    def close_gripper(self) -> bool:
        self.gripper.set_goal_state(configuration_name="closed")
        self.get_logger().info("Close gripper")
        return self.plan_and_execute(self.gripper, sleep_time=0.2)

    def go_home(self) -> bool:
        x, y, z = HOME_POSE
        return self.move_to_pose(x, y, z)

    def execute_pick_and_place(self, x_target: float, y_target: float) -> bool:
        place_x, place_y = PLACE_POINT_XY

        sequence = [
            ("go_home_start", lambda: self.go_home()),
            ("open_gripper", lambda: self.open_gripper()),
            ("pre_grasp", lambda: self.move_to_pose(x_target, y_target, SAFE_Z)),
            ("grasp_down", lambda: self.move_to_pose(x_target, y_target, PICK_Z)),
            ("close_gripper", lambda: self.close_gripper()),
            ("lift", lambda: self.move_to_pose(x_target, y_target, SAFE_Z)),
            ("move_to_place_safe", lambda: self.move_to_pose(place_x, place_y, SAFE_Z)),
            ("move_to_place_down", lambda: self.move_to_pose(place_x, place_y, PLACE_Z)),
            ("release", lambda: self.open_gripper()),
            ("retreat", lambda: self.move_to_pose(place_x, place_y, SAFE_Z)),
            ("go_home_end", lambda: self.go_home()),
        ]

        for step_name, step_fn in sequence:
            self.get_logger().info(f"Step: {step_name}")
            ok = step_fn()
            if not ok:
                self.get_logger().error(f"Failed at step: {step_name}")
                # Попытка открыть захват и уйти домой
                try:
                    self.open_gripper()
                except Exception:
                    pass
                try:
                    self.go_home()
                except Exception:
                    pass
                return False

        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Путь к best.pt или best.onnx")
    parser.add_argument("--target", type=int, required=True, choices=[1, 2, 3],
                        help="1=hammer, 2=wrench, 3=pliers")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init(args=None)

    node: Optional[RMC1PickPlaceNode] = None
    try:
        node = RMC1PickPlaceNode(
            model_path=args.model,
            target_id=args.target,
            imgsz=args.imgsz,
            conf_threshold=args.conf,
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
