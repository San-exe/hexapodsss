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
    IN_PLACE_STRIDE_SCALE,
    STANCE_DUTY_FACTOR,
    STRIDE_LENGTH,
    SWING_HEIGHT,
    TRIPOD_A,
    TRIPOD_B,
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
        tripod_a_time = CYCLE_SECONDS * 0.825
        tripod_a_lift = [foot_target(index, tripod_a_time)[1] for index in TRIPOD_A]
        tripod_b_lift = [foot_target(index, tripod_a_time)[1] for index in TRIPOD_B]
        self.assertTrue(all(value > 0 for value in tripod_a_lift))
        self.assertTrue(all(value == 0 for value in tripod_b_lift))

        tripod_b_time = CYCLE_SECONDS * 0.325
        tripod_a_lift = [foot_target(index, tripod_b_time)[1] for index in TRIPOD_A]
        tripod_b_lift = [foot_target(index, tripod_b_time)[1] for index in TRIPOD_B]
        self.assertTrue(all(value == 0 for value in tripod_a_lift))
        self.assertTrue(all(value > 0 for value in tripod_b_lift))

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

    def test_tripod_support_polygons_contain_center_of_mass(self):
        """Verify that both Tripod A and B enclose the body center of mass (0, 0)."""
        def foot_xy(index):
            leg = self.legs[index - 1]
            reach, _ = foot_target(index, 0.0)
            return (
                leg.hip_position[0] + reach * math.cos(leg.angle),
                leg.hip_position[1] + reach * math.sin(leg.angle),
            )

        def point_in_triangle(p, a, b, c):
            cp1 = _cross((b[0] - a[0], b[1] - a[1]), (p[0] - a[0], p[1] - a[1]))
            cp2 = _cross((c[0] - b[0], c[1] - b[1]), (p[0] - b[0], p[1] - b[1]))
            cp3 = _cross((a[0] - c[0], a[1] - c[1]), (p[0] - c[0], p[1] - c[1]))
            return (cp1 > 0 and cp2 > 0 and cp3 > 0) or (cp1 < 0 and cp2 < 0 and cp3 < 0)

        com = (0.0, 0.0)
        for name, tripod in (("Tripod A", TRIPOD_A), ("Tripod B", TRIPOD_B)):
            pts = [foot_xy(idx) for idx in tripod]
            self.assertTrue(
                point_in_triangle(com, pts[0], pts[1], pts[2]),
                f"{name} support polygon does not enclose the center of mass (0, 0)",
            )

    def test_tripod_duty_factor_maintains_continuous_support(self):
        """Verify that at every time step, at least 3 legs provide ground support."""
        for step in range(200):
            t = CYCLE_SECONDS * step / 200
            grounded_legs = sum(
                1 for index in range(1, LEG_COUNT + 1)
                if foot_target(index, t)[1] == 0.0
            )
            self.assertGreaterEqual(
                grounded_legs,
                3,
                f"Less than 3 legs grounded at t={t:.3f} s",
            )

    def test_all_legs_ik_within_limits_at_nominal_height_with_safety_margin(self):
        """Verify all legs stay well within limits across a full cycle at nominal height."""
        for index in range(1, LEG_COUNT + 1):
            for step in range(200):
                t = CYCLE_SECONDS * step / 200
                hip, knee = leg_joint_targets(index, t, BODY_CENTER_Z)
                self.assertLess(abs(hip), 0.35, f"Leg {index} hip {hip} exceeds margin")
                self.assertGreater(knee, -0.10)
                self.assertLess(knee, 0.40)

    def test_turning_parameter_validation(self):
        """Verify that out-of-range turn rates are rejected."""
        with self.assertRaises(ValueError):
            foot_target(1, 0.0, turn_rate=1.5)
        with self.assertRaises(ValueError):
            foot_target(1, 0.0, turn_rate=-1.1)

    def test_differential_turning_preserves_straight_baseline_at_zero_turn(self):
        """Verify turn_rate=0.0 produces identical trajectories to default baseline."""
        for index in range(1, LEG_COUNT + 1):
            for step in range(50):
                t = CYCLE_SECONDS * step / 50
                r_base, l_base = foot_target(index, t)
                r_turn, l_turn = foot_target(index, t, turn_rate=0.0, in_place=False)
                self.assertEqual(r_base, r_turn)
                self.assertEqual(l_base, l_turn)

    def test_differential_turning_stride_asymmetry(self):
        """Verify right legs take longer strides during left turns and vice versa."""
        # Left turn: right legs take 1.5x nominal, left legs take 0.5x nominal
        reach_l1_start, _ = foot_target(1, 0.0, turn_rate=0.5)
        reach_l1_mid, _ = foot_target(1, CYCLE_SECONDS * 0.65, turn_rate=0.5)
        reach_l2_start, _ = foot_target(2, 0.5 * CYCLE_SECONDS, turn_rate=0.5)
        reach_l2_mid, _ = foot_target(2, (0.5 + 0.65) * CYCLE_SECONDS, turn_rate=0.5)

        stride_l1 = abs(reach_l1_start - reach_l1_mid)
        stride_l2 = abs(reach_l2_start - reach_l2_mid)
        self.assertGreater(stride_l2, stride_l1, "Right leg stride should exceed left leg stride in left turn")

        # Middle legs (3 and 4) have zero radial displacement in body frame
        for t_step in range(20):
            t = CYCLE_SECONDS * t_step / 20
            r3, _ = foot_target(3, t, turn_rate=0.8)
            r4, _ = foot_target(4, t, turn_rate=0.8)
            self.assertAlmostEqual(r3, UPPER_LEG_LENGTH, places=7)
            self.assertAlmostEqual(r4, UPPER_LEG_LENGTH, places=7)

    def test_all_legs_ik_within_limits_across_all_turning_modes(self):
        """Verify all legs satisfy joint limits across curved and in-place turning modes."""
        modes = [
            ("Arc Left", 0.5, False),
            ("Arc Right", -0.5, False),
            ("Tight Left", 1.0, False),
            ("Tight Right", -1.0, False),
            ("In-Place Left", 1.0, True),
            ("In-Place Right", -1.0, True),
        ]
        for mode_name, tr, ip in modes:
            for index in range(1, LEG_COUNT + 1):
                for step in range(50):
                    t = CYCLE_SECONDS * step / 50
                    hip, knee = leg_joint_targets(
                        index,
                        t,
                        BODY_CENTER_Z,
                        turn_rate=tr,
                        in_place=ip,
                    )
                    self.assertGreaterEqual(
                        hip,
                        HIP_LIMITS[0],
                        f"{mode_name} Leg {index} hip {hip} < min",
                    )
                    self.assertLessEqual(
                        hip,
                        HIP_LIMITS[1],
                        f"{mode_name} Leg {index} hip {hip} > max",
                    )
                    self.assertGreaterEqual(
                        knee,
                        KNEE_LIMITS[0],
                        f"{mode_name} Leg {index} knee {knee} < min",
                    )
                    self.assertLessEqual(
                        knee,
                        KNEE_LIMITS[1],
                        f"{mode_name} Leg {index} knee {knee} > max",
                    )
                    # Verify safe margins: hip peak < 0.35 rad (limit 0.60), knee < 0.35 rad (limit 1.20)
                    self.assertLess(abs(hip), 0.35, f"{mode_name} Leg {index} hip exceeds margin")
                    self.assertLess(knee, 0.35, f"{mode_name} Leg {index} knee exceeds margin")

    def test_positive_negative_turn_stride_symmetry(self):
        """Verify exact stride and kinematic symmetry between +turn_rate and -turn_rate on mirrored legs."""
        for tr in (0.25, 0.50, 0.75, 1.0):
            for left_idx, right_idx in ((1, 2), (3, 4), (5, 6)):
                # Evaluate each leg at its actual stance start and end based on its tripod phase
                t_l_start = (0.0 if left_idx in TRIPOD_A else 0.5) * CYCLE_SECONDS
                t_l_end = t_l_start + STANCE_DUTY_FACTOR * CYCLE_SECONDS
                reach_l_start, _ = foot_target(left_idx, t_l_start, turn_rate=tr)
                reach_l_end, _ = foot_target(left_idx, t_l_end, turn_rate=tr)
                disp_left = abs(reach_l_start - reach_l_end)

                t_r_start = (0.0 if right_idx in TRIPOD_A else 0.5) * CYCLE_SECONDS
                t_r_end = t_r_start + STANCE_DUTY_FACTOR * CYCLE_SECONDS
                reach_r_start, _ = foot_target(right_idx, t_r_start, turn_rate=-tr)
                reach_r_end, _ = foot_target(right_idx, t_r_end, turn_rate=-tr)
                disp_right = abs(reach_r_start - reach_r_end)

                self.assertAlmostEqual(
                    disp_left,
                    disp_right,
                    places=7,
                    msg=f"Asymmetric displacement between leg {left_idx} (+{tr}) and leg {right_idx} (-{tr})",
                )

    def test_turning_kinematic_targets_are_finite_and_continuous(self):
        """Verify no NaN, Inf, or discontinuous jumps occur in turning IK targets."""
        for tr in (-0.5, -0.25, 0.0, 0.25, 0.5):
            for ip in (False, True):
                for index in range(1, LEG_COUNT + 1):
                    prev_hip, prev_knee = None, None
                    for step in range(120):
                        t = CYCLE_SECONDS * step / 120
                        hip, knee = leg_joint_targets(
                            index,
                            t,
                            BODY_CENTER_Z,
                            turn_rate=tr,
                            in_place=ip,
                        )
                        self.assertTrue(math.isfinite(hip), f"Non-finite hip on leg {index}")
                        self.assertTrue(math.isfinite(knee), f"Non-finite knee on leg {index}")
                        if prev_hip is not None:
                            # Step delta should be smooth (< 0.15 rad per 50ms step)
                            self.assertLess(
                                abs(hip - prev_hip),
                                0.15,
                                f"Discontinuous hip jump on leg {index} at t={t}",
                            )
                            self.assertLess(
                                abs(knee - prev_knee),
                                0.15,
                                f"Discontinuous knee jump on leg {index} at t={t}",
                            )
                        prev_hip, prev_knee = hip, knee

    def test_in_place_stride_scale_workspace_limits(self):
        """Verify that scaled in-place strides respect workspace and joint limits."""
        self.assertGreaterEqual(IN_PLACE_STRIDE_SCALE, 1.0)
        self.assertLessEqual(IN_PLACE_STRIDE_SCALE, 2.0)
        for tr in (-1.0, 1.0):
            for index in range(1, LEG_COUNT + 1):
                for step in range(60):
                    t = CYCLE_SECONDS * step / 60
                    hip, knee = leg_joint_targets(
                        index,
                        t,
                        BODY_CENTER_Z,
                        turn_rate=tr,
                        in_place=True,
                    )
                    self.assertGreaterEqual(hip, HIP_LIMITS[0])
                    self.assertLessEqual(hip, HIP_LIMITS[1])
                    self.assertGreaterEqual(knee, KNEE_LIMITS[0])
                    self.assertLessEqual(knee, KNEE_LIMITS[1])
                    # Safe margin verification (< 0.40 rad for hip, < 0.40 rad for knee)
                    self.assertLess(abs(hip), 0.40, f"Hip angle {hip} on leg {index} exceeds margin")
                    self.assertLess(knee, 0.40, f"Knee angle {knee} on leg {index} exceeds margin")

    def test_in_place_turning_direction_and_mirrored_symmetry(self):
        """Verify that positive and negative in-place turns produce opposite, mirrored foot sweeps."""
        for left_idx, right_idx in ((1, 2), (5, 6)):
            t_l_start = (0.0 if left_idx in TRIPOD_A else 0.5) * CYCLE_SECONDS
            t_l_end = t_l_start + STANCE_DUTY_FACTOR * CYCLE_SECONDS
            r_pos_start, _ = foot_target(left_idx, t_l_start, turn_rate=1.0, in_place=True)
            r_pos_end, _ = foot_target(left_idx, t_l_end, turn_rate=1.0, in_place=True)
            delta_pos = r_pos_end - r_pos_start

            r_neg_start, _ = foot_target(left_idx, t_l_start, turn_rate=-1.0, in_place=True)
            r_neg_end, _ = foot_target(left_idx, t_l_end, turn_rate=-1.0, in_place=True)
            delta_neg = r_neg_end - r_neg_start

            # Opposite signs
            self.assertAlmostEqual(delta_pos, -delta_neg, places=7)
            # Scaled displacement should match base * IN_PLACE_STRIDE_SCALE
            expected_stride = STRIDE_LENGTH * IN_PLACE_STRIDE_SCALE
            leg_proj = abs(math.cos(leg_geometry(left_idx).angle))
            self.assertAlmostEqual(abs(delta_pos), expected_stride * leg_proj, places=7)

    def test_in_place_stride_scale_preserves_straight_walking(self):
        """Verify that straight walking (in_place=False) remains 100% unaffected by in-place scaling."""
        for leg_idx in range(1, LEG_COUNT + 1):
            for step in range(30):
                t = CYCLE_SECONDS * step / 30
                r_straight, lift_straight = foot_target(
                    leg_idx, t, turn_rate=0.0, in_place=False
                )
                # Stride in straight walking must use unscaled STRIDE_LENGTH
                leg = leg_geometry(leg_idx)
                phase_offset = 0.0 if leg_idx in TRIPOD_A else 0.5
                phase = (t / CYCLE_SECONDS + phase_offset) % 1.0
                if phase < STANCE_DUTY_FACTOR:
                    prog = phase / STANCE_DUTY_FACTOR
                    expected_disp = STRIDE_LENGTH / 2 - STRIDE_LENGTH * prog
                else:
                    prog = (phase - STANCE_DUTY_FACTOR) / (1.0 - STANCE_DUTY_FACTOR)
                    expected_disp = -STRIDE_LENGTH / 2 + STRIDE_LENGTH * prog
                expected_r = UPPER_LEG_LENGTH + math.cos(leg.angle) * expected_disp
                self.assertAlmostEqual(r_straight, expected_r, places=7)

    def test_backward_walking_stride_direction_is_opposite_of_forward(self):
        """Verify that backward walking produces exact opposite radial stride displacement from forward walking."""
        for leg_idx in range(1, LEG_COUNT + 1):
            for step in range(60):
                t = CYCLE_SECONDS * step / 60
                r_fwd, lift_fwd = foot_target(leg_idx, t, direction="forward")
                r_bwd, lift_bwd = foot_target(leg_idx, t, direction="backward")

                # Vertical lift profile must be identical
                self.assertAlmostEqual(lift_fwd, lift_bwd, places=7)

                # Radial displacement relative to neutral reach must be exact opposite
                disp_fwd = r_fwd - UPPER_LEG_LENGTH
                disp_bwd = r_bwd - UPPER_LEG_LENGTH
                self.assertAlmostEqual(
                    disp_fwd,
                    -disp_bwd,
                    places=7,
                    msg=f"Displacement not opposite on leg {leg_idx} at t={t}",
                )

        # Check stance sweep direction for front legs (1 and 2):
        # Forward: foot moves backward during stance (r_start > r_end for positive forward_projection)
        # Backward: foot moves forward during stance (r_start < r_end for positive forward_projection)
        for front_idx in (1, 2):
            t_start = (0.0 if front_idx in TRIPOD_A else 0.5) * CYCLE_SECONDS
            t_end = t_start + STANCE_DUTY_FACTOR * CYCLE_SECONDS
            rf_start, _ = foot_target(front_idx, t_start, direction="forward")
            rf_end, _ = foot_target(front_idx, t_end, direction="forward")
            rb_start, _ = foot_target(front_idx, t_start, direction="backward")
            rb_end, _ = foot_target(front_idx, t_end, direction="backward")

            delta_fwd = rf_end - rf_start
            delta_bwd = rb_end - rb_start
            self.assertLess(delta_fwd, 0.0, "Forward stance should sweep leg inward/backward")
            self.assertGreater(delta_bwd, 0.0, "Backward stance should sweep leg outward/forward")
            self.assertAlmostEqual(delta_fwd, -delta_bwd, places=7)

    def test_backward_walking_all_legs_ik_within_limits_at_nominal_height(self):
        """Verify all legs stay well within limits across a full cycle in backward walking."""
        for index in range(1, LEG_COUNT + 1):
            for step in range(200):
                t = CYCLE_SECONDS * step / 200
                hip, knee = leg_joint_targets(
                    index, t, BODY_CENTER_Z, direction="backward"
                )
                self.assertLess(abs(hip), 0.35, f"Leg {index} hip {hip} exceeds margin in backward")
                self.assertGreater(knee, -0.10)
                self.assertLess(knee, 0.40)

    def test_backward_walking_with_turning_modes_ik_within_limits(self):
        """Verify all legs satisfy joint limits across curved turning modes in backward walking."""
        modes = [
            ("Backward Arc Left", 0.5),
            ("Backward Arc Right", -0.5),
            ("Backward Tight Left", 1.0),
            ("Backward Tight Right", -1.0),
        ]
        for mode_name, tr in modes:
            for index in range(1, LEG_COUNT + 1):
                for step in range(50):
                    t = CYCLE_SECONDS * step / 50
                    hip, knee = leg_joint_targets(
                        index,
                        t,
                        BODY_CENTER_Z,
                        turn_rate=tr,
                        in_place=False,
                        direction="backward",
                    )
                    self.assertGreaterEqual(
                        hip,
                        HIP_LIMITS[0],
                        f"{mode_name} Leg {index} hip {hip} < min",
                    )
                    self.assertLessEqual(
                        hip,
                        HIP_LIMITS[1],
                        f"{mode_name} Leg {index} hip {hip} > max",
                    )
                    self.assertGreaterEqual(
                        knee,
                        KNEE_LIMITS[0],
                        f"{mode_name} Leg {index} knee {knee} < min",
                    )
                    self.assertLessEqual(
                        knee,
                        KNEE_LIMITS[1],
                        f"{mode_name} Leg {index} knee {knee} > max",
                    )
                    self.assertLess(abs(hip), 0.35, f"{mode_name} Leg {index} hip exceeds margin")
                    self.assertLess(knee, 0.35, f"{mode_name} Leg {index} knee exceeds margin")

    def test_backward_walking_preserves_in_place_behavior(self):
        """Verify that in_place=True behavior is identical regardless of direction parameter."""
        for tr in (-1.0, 0.0, 1.0):
            for index in range(1, LEG_COUNT + 1):
                for step in range(30):
                    t = CYCLE_SECONDS * step / 30
                    rf, lf = foot_target(index, t, turn_rate=tr, in_place=True, direction="forward")
                    rb, lb = foot_target(index, t, turn_rate=tr, in_place=True, direction="backward")
                    self.assertEqual(rf, rb)
                    self.assertEqual(lf, lb)

    def test_direction_parameter_validation(self):
        """Verify that invalid direction arguments are rejected."""
        with self.assertRaises(ValueError):
            foot_target(1, 0.0, direction="sideways")
        with self.assertRaises(ValueError):
            foot_target(1, 0.0, direction="")
        with self.assertRaises(ValueError):
            leg_joint_targets(1, 0.0, direction="invalid")


if __name__ == "__main__":
    unittest.main()

