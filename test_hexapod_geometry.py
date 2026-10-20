import math
import unittest

from build_hexapod import (
    BODY_CENTER_Z,
    BODY_SIZE,
    HIP_ATTACHMENT_CLEARANCE,
    HIP_LIMITS,
    KNEE_LIMITS,
    LEG_COUNT,
    LOWER_LEG_LENGTH,
    LOWER_LEG_WIDTH,
    UPPER_LEG_LENGTH,
    UPPER_LEG_WIDTH,
    leg_geometry,
)
from tripod_gait_physics import (
    CYCLE_SECONDS,
    STANCE_DUTY_FACTOR,
    SWING_HEIGHT,
    foot_target,
    forward_kinematics,
    leg_joint_targets,
    solve_planar_ik,
)


def _dot(left, right):
    return left[0] * right[0] + left[1] * right[1]


def _point_segment_distance(point, start, end):
    segment = (end[0] - start[0], end[1] - start[1])
    length_squared = _dot(segment, segment)
    if length_squared == 0:
        return math.dist(point, start)
    offset = (point[0] - start[0], point[1] - start[1])
    ratio = max(0.0, min(1.0, _dot(offset, segment) / length_squared))
    closest = (start[0] + ratio * segment[0], start[1] + ratio * segment[1])
    return math.dist(point, closest)


def _cross(left, right):
    return left[0] * right[1] - left[1] * right[0]


def _segments_intersect(first_start, first_end, second_start, second_end):
    first = (first_end[0] - first_start[0], first_end[1] - first_start[1])
    second = (second_end[0] - second_start[0], second_end[1] - second_start[1])
    between = (
        second_start[0] - first_start[0],
        second_start[1] - first_start[1],
    )
    denominator = _cross(first, second)
    if abs(denominator) < 1e-12:
        return False
    first_ratio = _cross(between, second) / denominator
    second_ratio = _cross(between, first) / denominator
    return 0 <= first_ratio <= 1 and 0 <= second_ratio <= 1


def _segment_distance(first_start, first_end, second_start, second_end):
    if _segments_intersect(first_start, first_end, second_start, second_end):
        return 0.0
    return min(
        _point_segment_distance(first_start, second_start, second_end),
        _point_segment_distance(first_end, second_start, second_end),
        _point_segment_distance(second_start, first_start, first_end),
        _point_segment_distance(second_end, first_start, first_end),
    )


def _oriented_box_intersects_body(center, heading, half_length, half_width):
    body_half_x = BODY_SIZE[0] / 2
    body_half_y = BODY_SIZE[1] / 2
    radial = (math.cos(heading), math.sin(heading))
    tangent = (-radial[1], radial[0])

    for axis in ((1.0, 0.0), (0.0, 1.0), radial, tangent):
        upper_radius = (
            half_length * abs(_dot(radial, axis))
            + half_width * abs(_dot(tangent, axis))
        )
        body_radius = (
            body_half_x * abs(axis[0]) + body_half_y * abs(axis[1])
        )
        if abs(_dot(center, axis)) >= upper_radius + body_radius:
            return False
    return True


def _coppelia_euler_axes(orientation):
    alpha, beta, gamma = orientation
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    cg, sg = math.cos(gamma), math.sin(gamma)
    return (
        (cb * cg, ca * sg + sa * sb * cg, sa * sg - ca * sb * cg),
        (-cb * sg, ca * cg - sa * sb * sg, sa * cg + ca * sb * sg),
        (sb, -sa * cb, ca * cb),
    )


class FakeSim:
    """Small API double to verify generator object counts and local frames."""

    primitiveshape_cuboid = 0
    joint_revolute_subtype = 10
    jointmode_kinematic = 0
    jointmode_dynamic = 5
    simulation_stopped = 0
    handle_world = -1
    handle_all = -2

    def __init__(self):
        self.objects = {}
        self.next_handle = 1

    def _new_object(self, kind):
        handle = self.next_handle
        self.next_handle += 1
        self.objects[handle] = {
            "kind": kind,
            "position": None,
            "orientation": None,
            "parent": self.handle_world,
            "alias": None,
        }
        return handle

    def createPrimitiveShape(self, primitive_type, sizes, options):
        _ = (primitive_type, sizes, options)
        return self._new_object("shape")

    def createJoint(self, joint_type, joint_mode, options, sizes):
        _ = (joint_type, joint_mode, options, sizes)
        return self._new_object("joint")

    def setObjectAlias(self, handle, alias):
        self.objects[handle]["alias"] = alias

    def setObjectPosition(self, handle, position, relative_to):
        self.objects[handle]["position"] = (list(position), relative_to)

    def setObjectOrientation(self, handle, orientation, relative_to):
        self.objects[handle]["orientation"] = (list(orientation), relative_to)

    def setJointInterval(self, joint, cyclic, interval):
        _ = (joint, cyclic, interval)

    def setObjectParent(self, handle, parent, keep_in_place):
        _ = keep_in_place
        self.objects[handle]["parent"] = parent

    def getSimulationState(self):
        return self.simulation_stopped

    def getObject(self, path, options):
        _ = (path, options)
        return -1

    def getObjectsInTree(self, tree_base, object_type, options):
        _ = (tree_base, object_type, options)
        return []

    def removeObjects(self, handles):
        _ = handles


class HexapodGeometryTests(unittest.TestCase):
    def setUp(self):
        self.legs = [leg_geometry(index) for index in range(1, LEG_COUNT + 1)]

    def test_six_distinct_hip_points_and_left_right_symmetry(self):
        hip_points = {leg.hip_position for leg in self.legs}
        self.assertEqual(len(hip_points), LEG_COUNT)

        for left, right in (
            (self.legs[0], self.legs[1]),
            (self.legs[2], self.legs[3]),
            (self.legs[4], self.legs[5]),
        ):
            self.assertAlmostEqual(left.hip_position[0], right.hip_position[0])
            self.assertAlmostEqual(left.hip_position[1], -right.hip_position[1])
            self.assertAlmostEqual(left.angle, -right.angle)
            self.assertEqual(left.direction, right.direction)

        for leg in self.legs:
            radial_distance = math.hypot(*leg.hip_position[:2])
            radial_x = math.cos(leg.angle)
            radial_y = math.sin(leg.angle)
            edge_distances = []
            if abs(radial_x) > 1e-12:
                edge_distances.append((BODY_SIZE[0] / 2) / abs(radial_x))
            if abs(radial_y) > 1e-12:
                edge_distances.append((BODY_SIZE[1] / 2) / abs(radial_y))
            self.assertAlmostEqual(
                radial_distance - min(edge_distances),
                HIP_ATTACHMENT_CLEARANCE,
            )

    def test_front_middle_and_rear_upper_leg_directions_are_distinct(self):
        for index in (0, 1):
            leg = self.legs[index]
            self.assertGreater(leg.knee_position[0] - leg.hip_position[0], 0)
            self.assertGreater(
                abs(leg.knee_position[1] - leg.hip_position[1]),
                0,
            )

        for index, sign in ((2, 1), (3, -1)):
            leg = self.legs[index]
            self.assertAlmostEqual(leg.knee_position[0], leg.hip_position[0])
            self.assertGreater(
                sign * (leg.knee_position[1] - leg.hip_position[1]),
                0,
            )

        for index in (4, 5):
            leg = self.legs[index]
            self.assertLess(leg.knee_position[0] - leg.hip_position[0], 0)
            self.assertGreater(
                abs(leg.knee_position[1] - leg.hip_position[1]),
                0,
            )

    def test_joint_axes_and_local_segment_endpoints_align(self):
        for leg in self.legs:
            radial = (math.cos(leg.angle), math.sin(leg.angle))
            tangent = (-radial[1], radial[0])
            upper_delta = (
                leg.knee_position[0] - leg.hip_position[0],
                leg.knee_position[1] - leg.hip_position[1],
            )
            self.assertAlmostEqual(math.hypot(*upper_delta), UPPER_LEG_LENGTH)
            self.assertAlmostEqual(upper_delta[0], UPPER_LEG_LENGTH * radial[0])
            self.assertAlmostEqual(upper_delta[1], UPPER_LEG_LENGTH * radial[1])
            self.assertAlmostEqual(leg.upper_center[0], leg.hip_position[0] + upper_delta[0] / 2)
            self.assertAlmostEqual(leg.upper_center[1], leg.hip_position[1] + upper_delta[1] / 2)
            self.assertAlmostEqual(leg.lower_center[0], leg.knee_position[0])
            self.assertAlmostEqual(leg.lower_center[1], leg.knee_position[1])
            self.assertAlmostEqual(
                leg.lower_center[2] - LOWER_LEG_LENGTH / 2,
                0.0,
            )
            self.assertAlmostEqual(leg.knee_position[2], BODY_CENTER_Z)
            self.assertAlmostEqual(leg.hip_orientation[0], -math.pi / 2)
            self.assertAlmostEqual(leg.hip_orientation[1], -leg.angle)
            self.assertAlmostEqual(leg.hip_orientation[2], 0.0)
            self.assertAlmostEqual(leg.knee_orientation[0], -math.pi / 2)
            self.assertAlmostEqual(leg.knee_orientation[1], -leg.angle)
            self.assertAlmostEqual(leg.knee_orientation[2], 0.0)
            x_axis, y_axis, z_axis = _coppelia_euler_axes(leg.hip_orientation)
            self.assertAlmostEqual(x_axis[0], radial[0])
            self.assertAlmostEqual(x_axis[1], radial[1])
            self.assertAlmostEqual(x_axis[2], 0.0)
            self.assertAlmostEqual(y_axis[0], 0.0)
            self.assertAlmostEqual(y_axis[1], 0.0)
            self.assertAlmostEqual(y_axis[2], -1.0)
            self.assertAlmostEqual(z_axis[0], tangent[0])
            self.assertAlmostEqual(z_axis[1], tangent[1])
            self.assertAlmostEqual(z_axis[2], 0.0)
            self.assertAlmostEqual(tangent[0], -math.sin(leg.angle))
            self.assertAlmostEqual(tangent[1], math.cos(leg.angle))
            self.assertAlmostEqual(
                leg.lower_center[2] - LOWER_LEG_LENGTH / 2,
                0.0,
            )

    def test_neutral_leg_segments_do_not_intersect_body_or_each_other(self):
        upper_segments = [
            (leg.hip_position[:2], leg.knee_position[:2])
            for leg in self.legs
        ]

        for leg in self.legs:
            self.assertFalse(
                _oriented_box_intersects_body(
                    leg.upper_center[:2],
                    leg.angle,
                    UPPER_LEG_LENGTH / 2,
                    UPPER_LEG_WIDTH / 2,
                ),
                f"{leg.index}: upper link intersects body",
            )
            self.assertFalse(
                _oriented_box_intersects_body(
                    leg.knee_position[:2],
                    leg.angle,
                    LOWER_LEG_WIDTH / 2,
                    LOWER_LEG_WIDTH / 2,
                ),
                f"{leg.index}: lower link intersects body",
            )

        for first_index in range(LEG_COUNT):
            for second_index in range(first_index + 1, LEG_COUNT):
                distance = _segment_distance(
                    *upper_segments[first_index],
                    *upper_segments[second_index],
                )
                self.assertGreater(
                    distance,
                    UPPER_LEG_WIDTH,
                    f"upper links {first_index + 1} and {second_index + 1} overlap",
                )

        for first_index, first in enumerate(self.legs):
            for second_index, second in enumerate(self.legs):
                if first_index == second_index:
                    continue
                lower_to_upper_distance = _point_segment_distance(
                    second.knee_position[:2],
                    first.hip_position[:2],
                    first.knee_position[:2],
                )
                self.assertGreater(
                    lower_to_upper_distance,
                    UPPER_LEG_WIDTH / 2
                    + math.sqrt(2) * LOWER_LEG_WIDTH / 2,
                    f"upper link {first_index + 1} intersects lower link "
                    f"{second_index + 1}",
                )

        for first_index, first in enumerate(self.legs):
            for second in self.legs[first_index + 1 :]:
                self.assertGreater(
                    math.dist(first.knee_position[:2], second.knee_position[:2]),
                    math.sqrt(2) * LOWER_LEG_WIDTH,
                    f"lower links {first.index} and {second.index} overlap",
                )

    def test_generator_creates_six_legs_and_twelve_joint_frames(self):
        from build_hexapod import build_scene

        sim = FakeSim()
        scene = build_scene(sim, "kinematic")
        joints = [obj for obj in sim.objects.values() if obj["kind"] == "joint"]
        shapes = [obj for obj in sim.objects.values() if obj["kind"] == "shape"]
        self.assertEqual(len(scene["legs"]), LEG_COUNT)
        self.assertEqual(len(joints), 2 * LEG_COUNT)
        self.assertEqual(len(shapes), 2 + 2 * LEG_COUNT)

        for index, ((hip, knee), (upper, lower)) in enumerate(
            zip(scene["legs"], scene["leg_shapes"]),
            start=1,
        ):
            self.assertEqual(sim.objects[upper]["parent"], hip)
            self.assertEqual(sim.objects[knee]["parent"], upper)
            self.assertEqual(sim.objects[lower]["parent"], knee)
            self.assertEqual(
                sim.objects[upper]["position"],
                ([UPPER_LEG_LENGTH / 2, 0.0, 0.0], hip),
            )
            self.assertEqual(
                sim.objects[knee]["position"],
                ([UPPER_LEG_LENGTH / 2, 0.0, 0.0], upper),
            )
            self.assertEqual(
                sim.objects[lower]["position"],
                ([0.0, LOWER_LEG_LENGTH / 2, 0.0], knee),
            )
            self.assertEqual(sim.objects[upper]["orientation"], ([0.0, 0.0, 0.0], hip))
            self.assertEqual(sim.objects[knee]["orientation"], ([0.0, 0.0, 0.0], upper))
            self.assertEqual(sim.objects[lower]["orientation"], ([0.0, 0.0, 0.0], knee))
            self.assertEqual(sim.objects[hip]["alias"], f"HipJoint_{index}")
            self.assertEqual(sim.objects[knee]["alias"], f"KneeJoint_{index}")
            geometry = self.legs[index - 1]
            self.assertEqual(
                sim.objects[hip]["position"],
                (list(geometry.hip_position), sim.handle_world),
            )
            self.assertEqual(
                sim.objects[hip]["orientation"],
                (list(geometry.hip_orientation), sim.handle_world),
            )

    def test_invalid_leg_indices_are_rejected(self):
        for index in (0, LEG_COUNT + 1, -1):
            with self.subTest(index=index), self.assertRaises(ValueError):
                leg_geometry(index)

    def test_tripod_phase_and_left_right_radial_motion_are_mirrored(self):
        odd_tripod_time = CYCLE_SECONDS * 0.825
        odd_lift = [foot_target(index, odd_tripod_time)[1] for index in (1, 3, 5)]
        even_lift = [foot_target(index, odd_tripod_time)[1] for index in (2, 4, 6)]
        self.assertTrue(all(value > 0 for value in odd_lift))
        self.assertTrue(all(value == 0 for value in even_lift))

        even_tripod_time = CYCLE_SECONDS * 0.325
        odd_lift = [foot_target(index, even_tripod_time)[1] for index in (1, 3, 5)]
        even_lift = [foot_target(index, even_tripod_time)[1] for index in (2, 4, 6)]
        self.assertTrue(all(value == 0 for value in odd_lift))
        self.assertTrue(all(value > 0 for value in even_lift))

        for left_index, right_index in ((1, 2), (3, 4), (5, 6)):
            for phase in (0.0, 0.3, 0.7, 0.9):
                left = foot_target(left_index, CYCLE_SECONDS * phase)
                right = foot_target(
                    right_index,
                    CYCLE_SECONDS * (phase + 0.5),
                )
                self.assertAlmostEqual(left[0], right[0])
                self.assertAlmostEqual(left[1], right[1])
                left_leg = self.legs[left_index - 1]
                right_leg = self.legs[right_index - 1]
                left_world_y = (
                    left_leg.hip_position[1]
                    + math.sin(left_leg.angle) * left[0]
                )
                right_world_y = (
                    right_leg.hip_position[1]
                    + math.sin(right_leg.angle) * right[0]
                )
                self.assertAlmostEqual(left_world_y, -right_world_y)

    def test_tripod_foot_path_has_stance_swing_and_forward_projection(self):
        front_reach, front_lift = foot_target(1, 0.0)
        middle_reach, middle_lift = foot_target(3, 0.0)
        rear_reach, rear_lift = foot_target(5, 0.0)
        self.assertGreater(front_reach, UPPER_LEG_LENGTH)
        self.assertAlmostEqual(middle_reach, UPPER_LEG_LENGTH)
        self.assertLess(rear_reach, UPPER_LEG_LENGTH)
        self.assertEqual(front_lift, 0.0)
        self.assertEqual(middle_lift, 0.0)
        self.assertEqual(rear_lift, 0.0)

        swing_time = CYCLE_SECONDS * (
            STANCE_DUTY_FACTOR + (1.0 - STANCE_DUTY_FACTOR) / 2
        )
        _, swing_lift = foot_target(1, swing_time)
        self.assertAlmostEqual(swing_lift, SWING_HEIGHT)

    def test_inverse_kinematics_round_trips_and_respects_joint_limits(self):
        for index in range(1, LEG_COUNT + 1):
            for cycle_fraction in (0.0, 0.3, 0.65, 0.82, 0.95):
                reach, lift = foot_target(
                    index,
                    CYCLE_SECONDS * cycle_fraction,
                )
                angles = solve_planar_ik(reach, BODY_CENTER_Z - lift)
                reconstructed = forward_kinematics(*angles)
                with self.subTest(index=index, cycle_fraction=cycle_fraction):
                    self.assertGreaterEqual(angles[0], HIP_LIMITS[0])
                    self.assertLessEqual(angles[0], HIP_LIMITS[1])
                    self.assertGreaterEqual(angles[1], KNEE_LIMITS[0])
                    self.assertLessEqual(angles[1], KNEE_LIMITS[1])
                    self.assertAlmostEqual(reconstructed[0], reach, places=10)
                    self.assertAlmostEqual(
                        reconstructed[1],
                        BODY_CENTER_Z - lift,
                        places=10,
                    )

    def test_reported_leg_one_ik_failure_requires_collapsed_stance_height(self):
        elapsed = 4.35682
        with self.assertRaisesRegex(RuntimeError, "hip IK angle .* exceeds limits"):
            leg_joint_targets(1, elapsed, 0.26)

        nominal_angles = leg_joint_targets(1, elapsed, BODY_CENTER_Z)
        self.assertGreaterEqual(nominal_angles[0], HIP_LIMITS[0])
        self.assertLessEqual(nominal_angles[0], HIP_LIMITS[1])
        self.assertGreaterEqual(nominal_angles[1], KNEE_LIMITS[0])
        self.assertLessEqual(nominal_angles[1], KNEE_LIMITS[1])

    def test_full_tripod_cycle_stays_within_limits_at_supported_height(self):
        for index in range(1, LEG_COUNT + 1):
            for body_height in (BODY_CENTER_Z - 0.02, BODY_CENTER_Z, BODY_CENTER_Z + 0.02):
                for sample in range(201):
                    elapsed = CYCLE_SECONDS * sample / 201
                    with self.subTest(
                        index=index,
                        body_height=body_height,
                        sample=sample,
                    ):
                        hip_angle, knee_angle = leg_joint_targets(
                            index,
                            elapsed,
                            body_height,
                        )
                        self.assertGreaterEqual(hip_angle, HIP_LIMITS[0])
                        self.assertLessEqual(hip_angle, HIP_LIMITS[1])
                        self.assertGreaterEqual(knee_angle, KNEE_LIMITS[0])
                        self.assertLessEqual(knee_angle, KNEE_LIMITS[1])

    def test_inverse_kinematics_rejects_unreachable_targets(self):
        with self.assertRaises(ValueError):
            solve_planar_ik(UPPER_LEG_LENGTH + LOWER_LEG_LENGTH + 0.1, 0.1)


if __name__ == "__main__":
    unittest.main()
