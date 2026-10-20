
import math
import logging
import argparse
from typing import NamedTuple

LOGGER = logging.getLogger("hexapod.geometry")

# Robot body and ground
BODY_SIZE = (0.30, 0.20, 0.06)
BODY_CENTER_Z = 0.34

GROUND_SIZE = (2.0, 2.0, 0.02)
GROUND_CENTER_Z = -0.01

# Leg dimensions
UPPER_LEG_LENGTH = 0.16
LOWER_LEG_LENGTH = BODY_CENTER_Z
LEG_COUNT = 6
HIP_ATTACHMENT_CLEARANCE = 0.03
UPPER_LEG_WIDTH = 0.035
LOWER_LEG_WIDTH = 0.035
FRONT_REAR_SPLAY_ANGLE = math.radians(35.0)

# Upper-leg heading in the body XY plane. The order keeps tripods 1/3/5
# and 2/4/6 on alternating sides of each pair.
LEG_HEADINGS = (
    FRONT_REAR_SPLAY_ANGLE,       # Front-left
    -FRONT_REAR_SPLAY_ANGLE,      # Front-right
    math.pi / 2,                  # Middle-left
    -math.pi / 2,                 # Middle-right
    math.pi - FRONT_REAR_SPLAY_ANGLE,  # Rear-left
    -math.pi + FRONT_REAR_SPLAY_ANGLE, # Rear-right
)
LEG_NAMES = (
    "front-left",
    "front-right",
    "middle-left",
    "middle-right",
    "rear-left",
    "rear-right",
)

# Joint and physics settings
HIP_LIMITS = (-0.60, 0.60)
KNEE_LIMITS = (-0.20, 1.20)

HIP_TORQUE_LIMIT = 1.2
KNEE_TORQUE_LIMIT = 1.2

ROBOT_BODY_MASS = 0.8
UPPER_LEG_MASS = 0.05
LOWER_LEG_MASS = 0.10

RESPONDABLE_MASK = 0xFF00


class LegGeometry(NamedTuple):
    index: int
    angle: float
    direction: int
    hip_position: tuple[float, float, float]
    upper_center: tuple[float, float, float]
    knee_position: tuple[float, float, float]
    lower_center: tuple[float, float, float]
    hip_orientation: tuple[float, float, float]
    knee_orientation: tuple[float, float, float]


def leg_geometry(index: int) -> LegGeometry:
    """Return a mirrored radial leg pose with its hinge axis tangent to the plane."""
    if not 1 <= index <= LEG_COUNT:
        raise ValueError(
            f"leg index must be between 1 and {LEG_COUNT}"
        )

    angle = LEG_HEADINGS[index - 1]
    radial_x = math.cos(angle)
    radial_y = math.sin(angle)

    body_edge_radii = []
    if abs(radial_x) > 1e-12:
        body_edge_radii.append((BODY_SIZE[0] / 2) / abs(radial_x))
    if abs(radial_y) > 1e-12:
        body_edge_radii.append((BODY_SIZE[1] / 2) / abs(radial_y))
    hip_radius = min(body_edge_radii) + HIP_ATTACHMENT_CLEARANCE
    direction = 1 if radial_x > 1e-12 else -1 if radial_x < -1e-12 else 0

    hip_x = hip_radius * radial_x
    hip_y = hip_radius * radial_y
    knee_x = hip_x + UPPER_LEG_LENGTH * radial_x
    knee_y = hip_y + UPPER_LEG_LENGTH * radial_y
    hip_position = (hip_x, hip_y, BODY_CENTER_Z)
    knee_position = (knee_x, knee_y, BODY_CENTER_Z)
    upper_center = (
        (hip_x + knee_x) / 2,
        (hip_y + knee_y) / 2,
        BODY_CENTER_Z,
    )
    lower_center = (
        knee_x,
        knee_y,
        BODY_CENTER_Z - LOWER_LEG_LENGTH / 2,
    )

    return LegGeometry(
        index=index,
        angle=angle,
        direction=direction,
        hip_position=hip_position,
        upper_center=upper_center,
        knee_position=knee_position,
        lower_center=lower_center,
        # CoppeliaSim uses alpha-beta-gamma rotations about the object's
        # axes. This equivalent Euler pose maps local X outward, local Y down,
        # and local Z tangent to the leg's vertical motion plane.
        hip_orientation=(-math.pi / 2, -angle, 0.0),
        knee_orientation=(-math.pi / 2, -angle, 0.0),
    )


def create_box(
    sim,
    name,
    size,
    position,
    orientation=None,
):
    """Create and pose a cuboid in world coordinates."""
    obj = sim.createPrimitiveShape(
        sim.primitiveshape_cuboid,
        list(size),
        0,
    )

    sim.setObjectAlias(obj, name)
    sim.setObjectPosition(
        obj,
        list(position),
        sim.handle_world,
    )

    if orientation is not None:
        sim.setObjectOrientation(
            obj,
            list(orientation),
            sim.handle_world,
        )

    return obj


def create_revolute_joint(
    sim,
    name,
    position,
    orientation,
    joint_mode,
    limits,
):
    """Create a limited revolute joint at its world-space frame."""
    joint = sim.createJoint(
        sim.joint_revolute_subtype,
        joint_mode,
        0,
        [0.04, 0.02],
    )

    sim.setObjectAlias(joint, name)
    sim.setObjectPosition(
        joint,
        list(position),
        sim.handle_world,
    )
    sim.setObjectOrientation(
        joint,
        list(orientation),
        sim.handle_world,
    )

    lower_limit, upper_limit = limits

    sim.setJointInterval(
        joint,
        False,
        [lower_limit, upper_limit - lower_limit],
    )

    return joint


def _remove_previous_generated_objects(sim):
    """Remove the previously generated robot and ground."""
    body = sim.getObject(
        "/HexapodBody",
        {"noError": True},
    )

    if body != -1:
        sim.removeObjects(
            sim.getObjectsInTree(
                body,
                sim.handle_all,
                0,
            )
        )

    ground = sim.getObject(
        "/Ground",
        {"noError": True},
    )

    if ground != -1:
        sim.removeObjects([ground])


def configure_physics(sim, scene):
    """Enable dynamic links, collision response, and position motors."""
    sim.setBoolProperty(scene["ground"], "dynamic", False)
    sim.setBoolProperty(scene["ground"], "respondable", True)
    sim.setIntProperty(
        scene["ground"],
        "respondableMask",
        RESPONDABLE_MASK,
    )

    sim.setBoolProperty(scene["body"], "dynamic", True)
    sim.setBoolProperty(scene["body"], "respondable", True)
    sim.setIntProperty(
        scene["body"],
        "respondableMask",
        RESPONDABLE_MASK,
    )
    sim.setShapeMass(
        scene["body"],
        ROBOT_BODY_MASS,
    )

    for upper, lower in scene["leg_shapes"]:
        for shape, mass in (
            (upper, UPPER_LEG_MASS),
            (lower, LOWER_LEG_MASS),
        ):
            sim.setBoolProperty(shape, "dynamic", True)
            sim.setBoolProperty(shape, "respondable", True)
            sim.setIntProperty(
                shape,
                "respondableMask",
                RESPONDABLE_MASK,
            )
            sim.setShapeMass(shape, mass)

    for hip, knee in scene["legs"]:
        for joint, torque_limit in (
            (hip, HIP_TORQUE_LIMIT),
            (knee, KNEE_TORQUE_LIMIT),
        ):
            sim.setObjectInt32Param(
                joint,
                sim.jointintparam_motor_enabled,
                1,
            )
            sim.setObjectInt32Param(
                joint,
                sim.jointintparam_ctrl_enabled,
                1,
            )
            sim.setJointTargetForce(joint, torque_limit)
            sim.setJointTargetPosition(joint, 0.0)

    LOGGER.info(
        "Enabled dynamic links (body %.2f kg; upper %.2f kg; "
        "lower %.2f kg)",
        ROBOT_BODY_MASS,
        UPPER_LEG_MASS,
        LOWER_LEG_MASS,
    )


def build_scene(sim, mode="kinematic"):
    """Build the hexapod in kinematic or physics mode."""
    if mode not in ("kinematic", "physics"):
        raise ValueError(
            "mode must be 'kinematic' or 'physics'"
        )

    if sim.getSimulationState() != sim.simulation_stopped:
        raise RuntimeError(
            "Stop the CoppeliaSim simulation before rebuilding."
        )

    _remove_previous_generated_objects(sim)

    ground = create_box(
        sim,
        "Ground",
        GROUND_SIZE,
        (0.0, 0.0, GROUND_CENTER_Z),
    )

    body = create_box(
        sim,
        "HexapodBody",
        BODY_SIZE,
        (0.0, 0.0, BODY_CENTER_Z),
    )

    legs = []
    leg_shapes = []

    joint_mode = (
        sim.jointmode_kinematic
        if mode == "kinematic"
        else sim.jointmode_dynamic
    )

    for index in range(1, LEG_COUNT + 1):
        geometry = leg_geometry(index)
        suffix = str(index)

        hip = create_revolute_joint(
            sim,
            f"HipJoint_{suffix}",
            geometry.hip_position,
            geometry.hip_orientation,
            joint_mode,
            HIP_LIMITS,
        )

        upper = create_box(
            sim,
            f"UpperLeg_{suffix}",
            (UPPER_LEG_LENGTH, UPPER_LEG_WIDTH, UPPER_LEG_WIDTH),
            geometry.upper_center,
            geometry.hip_orientation,
        )

        knee = create_revolute_joint(
            sim,
            f"KneeJoint_{suffix}",
            geometry.knee_position,
            geometry.knee_orientation,
            joint_mode,
            KNEE_LIMITS,
        )

        lower = create_box(
            sim,
            f"LowerLeg_{suffix}",
            (LOWER_LEG_WIDTH, LOWER_LEG_LENGTH, LOWER_LEG_WIDTH),
            geometry.lower_center,
            geometry.hip_orientation,
        )

        # Keep each segment in its joint's local frame so rendered geometry,
        # collision geometry, and the revolute axes use the same transforms.
        sim.setObjectParent(hip, body, True)
        sim.setObjectParent(upper, hip, False)
        sim.setObjectPosition(
            upper,
            [UPPER_LEG_LENGTH / 2, 0.0, 0.0],
            hip,
        )
        sim.setObjectOrientation(upper, [0.0, 0.0, 0.0], hip)

        sim.setObjectParent(knee, upper, False)
        sim.setObjectPosition(
            knee,
            [UPPER_LEG_LENGTH / 2, 0.0, 0.0],
            upper,
        )
        sim.setObjectOrientation(knee, [0.0, 0.0, 0.0], upper)

        sim.setObjectParent(lower, knee, False)
        sim.setObjectPosition(
            lower,
            [0.0, LOWER_LEG_LENGTH / 2, 0.0],
            knee,
        )
        sim.setObjectOrientation(lower, [0.0, 0.0, 0.0], knee)

        legs.append((hip, knee))
        leg_shapes.append((upper, lower))

        LOGGER.info(
            "Created leg %d (%s, heading=%.1f degrees)",
            index,
            LEG_NAMES[index - 1],
            math.degrees(geometry.angle),
        )

    scene = {
        "ground": ground,
        "body": body,
        "legs": legs,
        "leg_shapes": leg_shapes,
        "mode": mode,
    }

    if mode == "physics":
        configure_physics(sim, scene)

    return scene


def main():
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    parser = argparse.ArgumentParser(
        description="Build the CoppeliaSim hexapod."
    )

    parser.add_argument(
        "--mode",
        choices=("kinematic", "physics"),
        default="kinematic",
        help=(
            "create kinematic test joints or physics-enabled joints"
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    LOGGER.info(
        "Connecting to CoppeliaSim ZeroMQ Remote API (%s mode)",
        args.mode,
    )

    client = RemoteAPIClient()
    sim = client.require("sim")

    scene = build_scene(sim, args.mode)

    LOGGER.info(
        "Generated %d legs and %d %s revolute joints",
        len(scene["legs"]),
        2 * len(scene["legs"]),
        args.mode,
    )

    if args.mode == "kinematic":
        LOGGER.warning(
            "Kinematic test scene: physics-based walking is not configured."
        )
    else:
        LOGGER.warning(
            "Physics scene configured; run the settling/contact test "
            "before using a gait."
        )


if __name__ == "__main__":
    main()
