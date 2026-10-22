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
IN_PLACE_STRIDE_SCALE = 1.25
SWING_HEIGHT = 0.04
STOP_TIMEOUT_SECONDS = 5.0
TRIPOD_A = (1, 4, 5)  # Front-Left, Middle-Right, Rear-Left
TRIPOD_B = (2, 3, 6)  # Front-Right, Middle-Left, Rear-Right


def foot_target(
    leg_index,
    elapsed,
    cycle_seconds=CYCLE_SECONDS,
    turn_rate=0.0,
    in_place=False,
):
    """Return the leg-plane forward reach and lift height for a tripod cycle.

    Parameters
    ----------
    leg_index : int
        Leg identifier (1 to 6).
    elapsed : float
        Elapsed simulation time in seconds.
    cycle_seconds : float
        Duration of a full tripod cycle.
    turn_rate : float
        Steering rate in [-1.0, 1.0]. Positive turns left (right legs take larger
        strides); negative turns right (left legs take larger strides); 0.0 is straight.
    in_place : bool
        If True, left and right tripods take equal and opposite strides to rotate
        in place about the robot's center of mass.
    """
    if not 1 <= leg_index <= LEG_COUNT:
        raise ValueError(f"leg index must be between 1 and {LEG_COUNT}")
    if elapsed < 0:
        raise ValueError("elapsed time cannot be negative")
    if cycle_seconds <= 0:
        raise ValueError("cycle_seconds must be positive")
    if not -1.0 <= turn_rate <= 1.0:
        raise ValueError("turn_rate must be between -1.0 and 1.0")

    leg = leg_geometry(leg_index)
    forward_projection = math.cos(leg.angle)
    phase_offset = 0.0 if leg_index in TRIPOD_A else 0.5
    phase = (elapsed / cycle_seconds + phase_offset) % 1.0

    # Compute leg-specific effective stride based on turning mode
    is_left = leg_index in (1, 3, 5)
    if in_place:
        base_stride = STRIDE_LENGTH * IN_PLACE_STRIDE_SCALE
        leg_stride = (base_stride if is_left else -base_stride) * turn_rate
    else:
        stride_scale = (1.0 - turn_rate) if is_left else (1.0 + turn_rate)
        leg_stride = STRIDE_LENGTH * stride_scale

    if phase < STANCE_DUTY_FACTOR:
        progress = phase / STANCE_DUTY_FACTOR
        world_forward_displacement = (
            leg_stride / 2 - leg_stride * progress
        )
        lift = 0.0
    else:
        progress = (phase - STANCE_DUTY_FACTOR) / (1.0 - STANCE_DUTY_FACTOR)
        world_forward_displacement = (
            -leg_stride / 2 + leg_stride * progress
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
    turn_rate=0.0,
    in_place=False,
):
    """Convert a tripod foot target to joint angles and enforce safe limits."""
    reach, lift = foot_target(
        leg_index,
        elapsed,
        cycle_seconds,
        turn_rate=turn_rate,
        in_place=in_place,
    )
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


def run_gait(
    cycles=3,
    cycle_seconds=CYCLE_SECONDS,
    turn_rate=0.0,
    in_place=False,
):
    if cycles <= 0:
        raise ValueError("cycles must be a positive integer")
    if cycle_seconds <= 0:
        raise ValueError("cycle_seconds must be positive")
    if not -1.0 <= turn_rate <= 1.0:
        raise ValueError("turn_rate must be between -1.0 and 1.0")

    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    client = RemoteAPIClient()
    sim = client.require("sim")
    if sim.getSimulationState() != sim.simulation_stopped:
        raise RuntimeError("Stop CoppeliaSim before starting the physics gait.")

    body, legs = _load_physics_legs(sim)
    lower_legs = [
        sim.getObject(f"/HexapodBody/HipJoint_{i}/UpperLeg_{i}/KneeJoint_{i}/LowerLeg_{i}")
        for i in range(1, 7)
    ]
    start_position = sim.getObjectPosition(body, sim.handle_world)
    start_orientation = sim.getObjectOrientation(body, sim.handle_world)
    end_position = start_position
    end_orientation = start_orientation
    last_logged_cycle = -1
    interrupted = False
    stepping_enabled = False

    prev_tips = {}
    cum_path_slip = {i: 0.0 for i in range(1, 7)}
    net_stance_scrub = {i: 0.0 for i in range(1, 7)}
    touchdown_pos = {}
    was_in_stance = {i: False for i in range(1, 7)}
    stance_counts = {i: 0 for i in range(1, 7)}

    try:
        client.setStepping(True)
        stepping_enabled = True
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
                    turn_rate=turn_rate,
                    in_place=in_place,
                )
                sim.setJointTargetPosition(hip, hip_angle)
                sim.setJointTargetPosition(knee, knee_angle)

            client.step()

            # Measure foot tip position and physical slip during stance (tip z < 5mm)
            for index, lower in enumerate(lower_legs, start=1):
                mat = sim.getObjectMatrix(lower, sim.handle_world)
                tip = sim.multiplyVector(mat, [0, 0.17, 0])
                in_stance = (tip[2] < 0.005)
                if in_stance:
                    stance_counts[index] += 1
                    if not was_in_stance[index]:
                        touchdown_pos[index] = tip
                    elif index in prev_tips:
                        step_dx = tip[0] - prev_tips[index][0]
                        step_dy = tip[1] - prev_tips[index][1]
                        cum_path_slip[index] += math.sqrt(step_dx * step_dx + step_dy * step_dy)
                    prev_tips[index] = tip
                else:
                    if was_in_stance[index] and index in touchdown_pos:
                        td = touchdown_pos[index]
                        scrub_dx = tip[0] - td[0]
                        scrub_dy = tip[1] - td[1]
                        net_stance_scrub[index] += math.sqrt(scrub_dx * scrub_dx + scrub_dy * scrub_dy)
                was_in_stance[index] = in_stance

            cycle_number = int(elapsed / cycle_seconds)
            if cycle_number > last_logged_cycle and cycle_number > 0:
                last_logged_cycle = cycle_number
                cur_pos = sim.getObjectPosition(body, sim.handle_world)
                cur_ori = sim.getObjectOrientation(body, sim.handle_world)
                cur_yaw_deg = math.degrees(cur_ori[2] - start_orientation[2])
                LOGGER.info(
                    "Completed tripod gait cycle %d of %d (dx=%.4f m, dyaw=%.2f deg)",
                    cycle_number,
                    cycles,
                    cur_pos[0] - start_position[0],
                    cur_yaw_deg,
                )

        if last_logged_cycle < cycles:
            LOGGER.info("Completed tripod gait cycle %d of %d", cycles, cycles)

        end_position = sim.getObjectPosition(body, sim.handle_world)
        end_orientation = sim.getObjectOrientation(body, sim.handle_world)
    except KeyboardInterrupt:
        interrupted = True
        LOGGER.info("Gait interrupted; stopping simulation safely.")
    finally:
        if stepping_enabled:
            try:
                client.setStepping(False)
            except Exception as exc:
                LOGGER.warning("Failed to restore stepping mode: %s", exc)
        _stop_simulation(sim)

    def _wrap_deg(d):
        return (d + 180.0) % 360.0 - 180.0

    forward_distance = end_position[0] - start_position[0]
    lateral_distance = end_position[1] - start_position[1]
    vertical_distance = end_position[2] - start_position[2]
    droll = _wrap_deg(math.degrees(end_orientation[0] - start_orientation[0]))
    dpitch = _wrap_deg(math.degrees(end_orientation[1] - start_orientation[1]))
    dyaw = _wrap_deg(math.degrees(end_orientation[2] - start_orientation[2]))

    LOGGER.info(
        "Initial pose: pos=[%.4f, %.4f, %.4f] m, ori=[%.2f, %.2f, %.2f] deg",
        start_position[0],
        start_position[1],
        start_position[2],
        math.degrees(start_orientation[0]),
        math.degrees(start_orientation[1]),
        math.degrees(start_orientation[2]),
    )
    LOGGER.info(
        "Final pose:   pos=[%.4f, %.4f, %.4f] m, ori=[%.2f, %.2f, %.2f] deg",
        end_position[0],
        end_position[1],
        end_position[2],
        math.degrees(end_orientation[0]),
        math.degrees(end_orientation[1]),
        math.degrees(end_orientation[2]),
    )
    LOGGER.info(
        "Measured body displacement: dx=%.4f m, dy=%.4f m, dz=%.4f m",
        forward_distance,
        lateral_distance,
        vertical_distance,
    )
    LOGGER.info(
        "Measured body attitude change: droll=%.2f deg, dpitch=%.2f deg, dyaw=%.2f deg",
        droll,
        dpitch,
        dyaw,
    )
    # Account for final stance phase of legs still on the ground
    for index in range(1, 7):
        if was_in_stance[index] and index in touchdown_pos and index in prev_tips:
            td = touchdown_pos[index]
            cur = prev_tips[index]
            net_stance_scrub[index] += math.sqrt(
                (cur[0] - td[0]) ** 2 + (cur[1] - td[1]) ** 2
            )

    LOGGER.info(
        "Measured stance scrub (net touchdown-liftoff displacement): L1=%.1fmm, L2=%.1fmm, L3(mid)=%.1fmm, L4(mid)=%.1fmm, L5=%.1fmm, L6=%.1fmm",
        net_stance_scrub[1] * 1000.0,
        net_stance_scrub[2] * 1000.0,
        net_stance_scrub[3] * 1000.0,
        net_stance_scrub[4] * 1000.0,
        net_stance_scrub[5] * 1000.0,
        net_stance_scrub[6] * 1000.0,
    )
    LOGGER.info(
        "Measured foot slip (cumulative path length): L1=%.1fmm, L2=%.1fmm, L3(mid)=%.1fmm, L4(mid)=%.1fmm, L5=%.1fmm, L6=%.1fmm",
        cum_path_slip[1] * 1000.0,
        cum_path_slip[2] * 1000.0,
        cum_path_slip[3] * 1000.0,
        cum_path_slip[4] * 1000.0,
        cum_path_slip[5] * 1000.0,
        cum_path_slip[6] * 1000.0,
    )

    if not interrupted:
        if in_place:
            if abs(dyaw) <= 0.5:
                raise RuntimeError(
                    "In-place yaw rotation was not measurable (> 0.5 deg); inspect steering."
                )
        elif abs(turn_rate) < 0.9:
            if forward_distance <= 0.01:
                raise RuntimeError(
                    "Forward progress was not measurable (> 0.01 m); inspect contacts and tune gait."
                )
    return forward_distance, dyaw


def main():
    parser = argparse.ArgumentParser(
        description="Run a slow, physics-based tripod gait with planar IK."
    )
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--cycle-seconds", type=float, default=CYCLE_SECONDS)
    parser.add_argument(
        "--turn-rate",
        type=float,
        default=0.0,
        help="Turn rate in [-1.0, 1.0]: positive turns left, negative turns right.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rotate in place around the body center instead of translating forward.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_gait(
        cycles=args.cycles,
        cycle_seconds=args.cycle_seconds,
        turn_rate=args.turn_rate,
        in_place=args.in_place,
    )


if __name__ == "__main__":
    main()
