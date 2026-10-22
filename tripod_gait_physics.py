import argparse
import logging
import math
import time

from build_hexapod import (
    BODY_CENTER_Z,
    HIP_LIMITS,
    KNEE_LIMITS,
    LEG_COUNT,
    LOWER_LEG_LENGTH,
    UPPER_LEG_LENGTH,
    leg_geometry,
)


LOGGER = logging.getLogger("hexapod.physics_gait")
CYCLE_SECONDS = 6.0
STANCE_DUTY_FACTOR = 0.65
STRIDE_LENGTH = 0.05
SWING_HEIGHT = 0.04
STOP_TIMEOUT_SECONDS = 5.0
TRIPOD_A = (1, 4, 5)  # Front-Left, Middle-Right, Rear-Left
TRIPOD_B = (2, 3, 6)  # Front-Right, Middle-Left, Rear-Right


def foot_target(leg_index, elapsed, cycle_seconds=CYCLE_SECONDS):
    """Return the leg-plane forward reach and lift height for a tripod cycle."""
    if not 1 <= leg_index <= LEG_COUNT:
        raise ValueError(f"leg index must be between 1 and {LEG_COUNT}")
    if elapsed < 0:
        raise ValueError("elapsed time cannot be negative")
    if cycle_seconds <= 0:
        raise ValueError("cycle_seconds must be positive")

    leg = leg_geometry(leg_index)
    forward_projection = math.cos(leg.angle)
    phase_offset = 0.0 if leg_index in TRIPOD_A else 0.5
    phase = (elapsed / cycle_seconds + phase_offset) % 1.0

    if phase < STANCE_DUTY_FACTOR:
        progress = phase / STANCE_DUTY_FACTOR
        world_forward_displacement = (
            STRIDE_LENGTH / 2 - STRIDE_LENGTH * progress
        )
        lift = 0.0
    else:
        progress = (phase - STANCE_DUTY_FACTOR) / (1.0 - STANCE_DUTY_FACTOR)
        world_forward_displacement = (
            -STRIDE_LENGTH / 2 + STRIDE_LENGTH * progress
        )
        lift = SWING_HEIGHT * math.sin(math.pi * progress)

    # Project the requested body-forward foot sweep into this leg's radial
    # sagittal plane. Mirrored legs therefore receive mirrored world poses
    # without pretending their local reach axes all point along world X.
    radial_displacement = forward_projection * world_forward_displacement
    return UPPER_LEG_LENGTH + radial_displacement, lift


def solve_planar_ik(reach, down):
    """Solve hip and knee angles for a two-link sagittal-plane foot target."""
    if reach <= 0 or down <= 0:
        raise ValueError("reach and downward distance must be positive")

    cosine = (
        reach * reach
        + down * down
        - UPPER_LEG_LENGTH * UPPER_LEG_LENGTH
        - LOWER_LEG_LENGTH * LOWER_LEG_LENGTH
    ) / (2 * UPPER_LEG_LENGTH * LOWER_LEG_LENGTH)
    if not -1.0 <= cosine <= 1.0:
        raise ValueError(f"foot target is outside the leg workspace: {reach=}, {down=}")

    relative_angle = math.acos(cosine)
    hip_angle = math.atan2(down, reach) - math.atan2(
        LOWER_LEG_LENGTH * math.sin(relative_angle),
        UPPER_LEG_LENGTH + LOWER_LEG_LENGTH * math.cos(relative_angle),
    )
    knee_angle = relative_angle - math.pi / 2
    return hip_angle, knee_angle


def forward_kinematics(hip_angle, knee_angle):
    """Return local reach and downward distance for a pair of joint angles."""
    combined_angle = hip_angle + knee_angle
    reach = UPPER_LEG_LENGTH * math.cos(hip_angle) - (
        LOWER_LEG_LENGTH * math.sin(combined_angle)
    )
    down = UPPER_LEG_LENGTH * math.sin(hip_angle) + (
        LOWER_LEG_LENGTH * math.cos(combined_angle)
    )
    return reach, down


def leg_joint_targets(
    leg_index,
    elapsed,
    body_height=BODY_CENTER_Z,
    cycle_seconds=CYCLE_SECONDS,
):
    """Convert a tripod foot target to joint angles and enforce safe limits."""
    reach, lift = foot_target(leg_index, elapsed, cycle_seconds)
    hip_angle, knee_angle = solve_planar_ik(reach, body_height - lift)
    if not HIP_LIMITS[0] <= hip_angle <= HIP_LIMITS[1]:
        raise RuntimeError(
            f"Leg {leg_index} hip IK angle {hip_angle:.3f} rad exceeds limits "
            f"at body height {body_height:.4f} m "
            f"(elapsed={elapsed:.4f} s, reach={reach:.4f} m, lift={lift:.4f} m)."
        )
    if not KNEE_LIMITS[0] <= knee_angle <= KNEE_LIMITS[1]:
        raise RuntimeError(
            f"Leg {leg_index} knee IK angle {knee_angle:.3f} rad exceeds limits "
            f"at body height {body_height:.4f} m "
            f"(elapsed={elapsed:.4f} s, reach={reach:.4f} m, lift={lift:.4f} m)."
        )
    return hip_angle, knee_angle


def _load_physics_legs(sim):
    body = sim.getObject("/HexapodBody", {"noError": True})
    if body == -1:
        raise RuntimeError("No /HexapodBody found; build with --mode physics first.")

    legs = []
    for index in range(1, LEG_COUNT + 1):
        hip = sim.getObject(f"/HexapodBody/HipJoint_{index}", {"noError": True})
        knee = sim.getObject(
            f"/HexapodBody/HipJoint_{index}/UpperLeg_{index}/KneeJoint_{index}",
            {"noError": True},
        )
        if hip == -1 or knee == -1:
            raise RuntimeError(f"Missing hip or knee joint for leg {index}.")

        for joint in (hip, knee):
            mode, _ = sim.getJointMode(joint)
            if mode != sim.jointmode_dynamic:
                raise RuntimeError(
                    f"Leg {index} is not in dynamic mode; rebuild with --mode physics."
                )
            if sim.getObjectInt32Param(
                joint,
                sim.jointintparam_motor_enabled,
            ) != 1 or sim.getObjectInt32Param(
                joint,
                sim.jointintparam_ctrl_enabled,
            ) != 1:
                raise RuntimeError(f"Leg {index} has a disabled position motor.")
        legs.append((hip, knee))
    return body, legs


def _stop_simulation(sim):
    if sim.getSimulationState() == sim.simulation_stopped:
        return
    sim.stopSimulation()
    deadline = time.monotonic() + STOP_TIMEOUT_SECONDS
    while sim.getSimulationState() != sim.simulation_stopped:
        if time.monotonic() >= deadline:
            raise RuntimeError("CoppeliaSim did not stop within the safe-stop timeout.")
        time.sleep(0.02)


def run_gait(cycles=3, cycle_seconds=CYCLE_SECONDS):
    if cycles <= 0:
        raise ValueError("cycles must be a positive integer")
    if cycle_seconds <= 0:
        raise ValueError("cycle_seconds must be positive")

    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    client = RemoteAPIClient()
    sim = client.require("sim")
    if sim.getSimulationState() != sim.simulation_stopped:
        raise RuntimeError("Stop CoppeliaSim before starting the physics gait.")

    body, legs = _load_physics_legs(sim)
    client.setStepping(True)
    start_position = sim.getObjectPosition(body, sim.handle_world)
    end_position = start_position
    last_logged_cycle = -1
    interrupted = False

    try:
        sim.startSimulation()
        start_time = sim.getSimulationTime()
        while sim.getSimulationTime() - start_time < cycles * cycle_seconds:
            elapsed = sim.getSimulationTime() - start_time
            for index, (hip, knee) in enumerate(legs, start=1):
                hip_angle, knee_angle = leg_joint_targets(
                    index,
                    elapsed,
                    BODY_CENTER_Z,
                    cycle_seconds,
                )
                sim.setJointTargetPosition(hip, hip_angle)
                sim.setJointTargetPosition(knee, knee_angle)

            client.step()
            cycle_number = int(elapsed / cycle_seconds)
            if cycle_number > last_logged_cycle and cycle_number > 0:
                last_logged_cycle = cycle_number
                LOGGER.info("Completed tripod gait cycle %d of %d", cycle_number, cycles)

        if last_logged_cycle < cycles:
            LOGGER.info("Completed tripod gait cycle %d of %d", cycles, cycles)

        end_position = sim.getObjectPosition(body, sim.handle_world)
    except KeyboardInterrupt:
        interrupted = True
        LOGGER.info("Gait interrupted; stopping simulation safely.")
    finally:
        _stop_simulation(sim)

    forward_distance = end_position[0] - start_position[0]
    LOGGER.info(
        "Measured body displacement: dx=%.4f m, dy=%.4f m, dz=%.4f m",
        forward_distance,
        end_position[1] - start_position[1],
        end_position[2] - start_position[2],
    )
    if not interrupted and forward_distance <= 0.01:
        raise RuntimeError(
            "Forward progress was not measurable (> 0.01 m); inspect contacts and tune gait."
        )
    return forward_distance


def main():
    parser = argparse.ArgumentParser(
        description="Run a slow, physics-based tripod gait with planar IK."
    )
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--cycle-seconds", type=float, default=CYCLE_SECONDS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_gait(args.cycles, args.cycle_seconds)


if __name__ == "__main__":
    main()
