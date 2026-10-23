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
    direction="forward",
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
    direction : str
        Walking direction: 'forward' or 'backward'. Defaults to 'forward'.
    """
    if not 1 <= leg_index <= LEG_COUNT:
        raise ValueError(f"leg index must be between 1 and {LEG_COUNT}")
    if elapsed < 0:
        raise ValueError("elapsed time cannot be negative")
    if cycle_seconds <= 0:
        raise ValueError("cycle_seconds must be positive")
    if not -1.0 <= turn_rate <= 1.0:
        raise ValueError("turn_rate must be between -1.0 and 1.0")
    if direction not in ("forward", "backward"):
        raise ValueError(
            f"direction must be 'forward' or 'backward', got {direction!r}"
        )

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

    dir_sign = -1.0 if (direction == "backward" and not in_place) else 1.0

    if phase < STANCE_DUTY_FACTOR:
        progress = phase / STANCE_DUTY_FACTOR
        world_forward_displacement = dir_sign * (
            leg_stride / 2 - leg_stride * progress
        )
        lift = 0.0
    else:
        progress = (phase - STANCE_DUTY_FACTOR) / (1.0 - STANCE_DUTY_FACTOR)
        world_forward_displacement = dir_sign * (
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
    direction="forward",
):
    """Convert a tripod foot target to joint angles and enforce safe limits."""
    reach, lift = foot_target(
        leg_index,
        elapsed,
        cycle_seconds,
        turn_rate=turn_rate,
        in_place=in_place,
        direction=direction,
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


def _get_remote_client(port=None):
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    import socket
    if port is not None:
        return RemoteAPIClient(port=port)
    for p in (23000, 23001):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.2)
        try:
            s.connect(("127.0.0.1", p))
            s.close()
            return RemoteAPIClient(port=p)
        except Exception:
            pass
    return RemoteAPIClient()


def _stop_simulation(sim):
    if sim.getSimulationState() == sim.simulation_stopped:
        return
    sim.stopSimulation()
    deadline = time.monotonic() + STOP_TIMEOUT_SECONDS
    while sim.getSimulationState() != sim.simulation_stopped:
        if time.monotonic() >= deadline:
            raise RuntimeError("CoppeliaSim did not stop within the safe-stop timeout.")
        time.sleep(0.02)



class GaitResult(tuple):
    """2-tuple (forward_distance, dyaw) with full telemetry attributes."""

    def __new__(
        cls,
        dx,
        dyaw,
        dy=0.0,
        dz=0.0,
        final_z=BODY_CENTER_Z,
        droll=0.0,
        dpitch=0.0,
        cycles_completed=0,
        target_reached=True,
        foot_slip=None,
        foot_scrub=None,
        wall_seconds=0.0,
        start_pos=None,
        end_pos=None,
        start_ori=None,
        end_ori=None,
    ):
        inst = super().__new__(cls, (dx, dyaw))
        inst.dx = dx
        inst.dyaw = dyaw
        inst.dy = dy
        inst.dz = dz
        inst.final_z = final_z
        inst.droll = droll
        inst.dpitch = dpitch
        inst.cycles_completed = cycles_completed
        inst.target_reached = target_reached
        inst.foot_slip = foot_slip or {}
        inst.foot_scrub = foot_scrub or {}
        inst.wall_seconds = wall_seconds
        inst.start_pos = start_pos or (0.0, 0.0, 0.0)
        inst.end_pos = end_pos or (0.0, 0.0, 0.0)
        inst.start_ori = start_ori or (0.0, 0.0, 0.0)
        inst.end_ori = end_ori or (0.0, 0.0, 0.0)
        return inst


def _wrap_deg(d):
    """Wrap angle to [-180.0, 180.0) degrees."""
    return (d + 180.0) % 360.0 - 180.0


def run_gait(
    cycles=5,
    cycle_seconds=CYCLE_SECONDS,
    turn_rate=0.0,
    in_place=False,
    direction="forward",
    target_yaw=None,
    target_yaw_tol=2.0,
    max_cycles=60,
    port=None,
):
    if target_yaw is not None:
        in_place = True
        if turn_rate == 0.0:
            turn_rate = 1.0 if target_yaw > 0 else -1.0
        if max_cycles <= 0:
            raise ValueError("max_cycles must be a positive integer")
        if target_yaw_tol <= 0:
            raise ValueError("target_yaw_tol must be positive")
    else:
        if cycles <= 0:
            raise ValueError("cycles must be a positive integer")
    if cycle_seconds <= 0:
        raise ValueError("cycle_seconds must be positive")
    if not -1.0 <= turn_rate <= 1.0:
        raise ValueError("turn_rate must be between -1.0 and 1.0")
    if direction not in ("forward", "backward"):
        raise ValueError(
            f"direction must be 'forward' or 'backward', got {direction!r}"
        )

    client = _get_remote_client(port=port)
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
        start_yaw_rad = start_orientation[2]
        cos_start_yaw = math.cos(start_yaw_rad)
        sin_start_yaw = math.sin(start_yaw_rad)
        cycle_limit = max_cycles if target_yaw is not None else cycles
        target_reached = False if target_yaw is not None else True
        active_turn_rate = turn_rate
        elapsed = 0.0

        if target_yaw is not None:
            LOGGER.info(
                "Starting target-based rotation: target=%+.2f deg, tolerance=±%.2f deg, max_cycles=%d",
                target_yaw,
                target_yaw_tol,
                max_cycles,
            )

        while sim.getSimulationTime() - start_time < cycle_limit * cycle_seconds:
            elapsed = sim.getSimulationTime() - start_time

            if target_yaw is not None:
                cur_ori = sim.getObjectOrientation(body, sim.handle_world)
                cur_yaw_diff = _wrap_deg(
                    math.degrees(
                        math.atan2(
                            math.sin(cur_ori[2] - start_yaw_rad),
                            math.cos(cur_ori[2] - start_yaw_rad),
                        )
                    )
                )
                yaw_error = _wrap_deg(target_yaw - cur_yaw_diff)

                if abs(yaw_error) <= target_yaw_tol:
                    target_reached = True
                    LOGGER.info(
                        "Target yaw reached: target=%+.2f deg, actual=%+.2f deg (error=%+.2f deg within ±%.2f deg tolerance at elapsed=%.2f s).",
                        target_yaw,
                        cur_yaw_diff,
                        yaw_error,
                        target_yaw_tol,
                        elapsed,
                    )
                    break

                # Scale turn_rate when near target to avoid overshoot
                if abs(yaw_error) < 6.0:
                    active_turn_rate = math.copysign(
                        max(0.25, abs(yaw_error) / 6.0), yaw_error
                    )
                else:
                    active_turn_rate = 1.0 if yaw_error > 0 else -1.0

            for index, (hip, knee) in enumerate(legs, start=1):
                hip_angle, knee_angle = leg_joint_targets(
                    index,
                    elapsed,
                    BODY_CENTER_Z,
                    cycle_seconds,
                    turn_rate=active_turn_rate,
                    in_place=in_place,
                    direction=direction,
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
                cur_dx = (
                    (cur_pos[0] - start_position[0]) * cos_start_yaw
                    + (cur_pos[1] - start_position[1]) * sin_start_yaw
                )
                cur_yaw_deg = _wrap_deg(
                    math.degrees(
                        math.atan2(
                            math.sin(cur_ori[2] - start_yaw_rad),
                            math.cos(cur_ori[2] - start_yaw_rad),
                        )
                    )
                )
                if target_yaw is not None:
                    LOGGER.info(
                        "Tripod cycle %d (target=%+.1f deg, actual=%+.2f deg, error=%+.2f deg, dx=%.4f m)",
                        cycle_number,
                        target_yaw,
                        cur_yaw_deg,
                        _wrap_deg(target_yaw - cur_yaw_deg),
                        cur_dx,
                    )
                else:
                    LOGGER.info(
                        "Completed tripod gait cycle %d of %d (dx=%.4f m, dyaw=%.2f deg)",
                        cycle_number,
                        cycles,
                        cur_dx,
                        cur_yaw_deg,
                    )

        if target_yaw is not None and not target_reached and not interrupted:
            cur_ori = sim.getObjectOrientation(body, sim.handle_world)
            cur_yaw_deg = _wrap_deg(
                math.degrees(
                    math.atan2(
                        math.sin(cur_ori[2] - start_yaw_rad),
                        math.cos(cur_ori[2] - start_yaw_rad),
                    )
                )
            )
            LOGGER.warning(
                "Target yaw NOT reached within %d cycles (target=%+.2f deg, actual=%+.2f deg, error=%+.2f deg)",
                max_cycles,
                target_yaw,
                cur_yaw_deg,
                _wrap_deg(target_yaw - cur_yaw_deg),
            )
        elif target_yaw is None and last_logged_cycle < cycles:
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

    world_dx = end_position[0] - start_position[0]
    world_dy = end_position[1] - start_position[1]
    cos_yaw = math.cos(start_orientation[2])
    sin_yaw = math.sin(start_orientation[2])
    forward_distance = world_dx * cos_yaw + world_dy * sin_yaw
    lateral_distance = -world_dx * sin_yaw + world_dy * cos_yaw
    vertical_distance = end_position[2] - start_position[2]
    droll = _wrap_deg(math.degrees(end_orientation[0] - start_orientation[0]))
    dpitch = _wrap_deg(math.degrees(end_orientation[1] - start_orientation[1]))
    dyaw = _wrap_deg(
        math.degrees(
            math.atan2(
                math.sin(end_orientation[2] - start_orientation[2]),
                math.cos(end_orientation[2] - start_orientation[2]),
            )
        )
    )

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

    completed_cycles = (
        last_logged_cycle if last_logged_cycle > 0
        else (int(elapsed / cycle_seconds) if cycle_seconds > 0 else 0)
    )

    if not interrupted:
        if target_yaw is not None:
            yaw_err = abs(_wrap_deg(target_yaw - dyaw))
            if yaw_err > target_yaw_tol:
                raise RuntimeError(
                    f"Target yaw {target_yaw:+.1f} deg was not reached within tolerance ±{target_yaw_tol:.1f} deg (actual dyaw={dyaw:+.2f} deg, error={yaw_err:.2f} deg)."
                )
        elif in_place:
            if abs(dyaw) <= 0.5:
                raise RuntimeError(
                    "In-place yaw rotation was not measurable (> 0.5 deg); inspect steering."
                )
        elif abs(turn_rate) < 0.9:
            if direction == "backward":
                if forward_distance >= -0.01:
                    raise RuntimeError(
                        "Backward progress was not measurable (< -0.01 m); inspect contacts and tune gait."
                    )
            else:
                if forward_distance <= 0.01:
                    raise RuntimeError(
                        "Forward progress was not measurable (> 0.01 m); inspect contacts and tune gait."
                    )

    return GaitResult(
        forward_distance,
        dyaw,
        dy=lateral_distance,
        dz=vertical_distance,
        final_z=end_position[2],
        droll=droll,
        dpitch=dpitch,
        cycles_completed=completed_cycles,
        target_reached=target_reached,
        foot_slip=cum_path_slip,
        foot_scrub=net_stance_scrub,
        wall_seconds=elapsed,
        start_pos=tuple(start_position),
        end_pos=tuple(end_position),
        start_ori=tuple(start_orientation),
        end_ori=tuple(end_orientation),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run a slow, physics-based tripod gait with planar IK."
    )
    parser.add_argument("--cycles", type=int, default=5, help="Number of gait cycles (default: 5).")
    parser.add_argument("--cycle-seconds", type=float, default=CYCLE_SECONDS, help="Cycle duration in seconds (default: 6.0).")
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
    parser.add_argument(
        "--direction",
        choices=["forward", "backward"],
        default="forward",
        help="Walking direction: forward or backward (default: forward).",
    )
    parser.add_argument(
        "--target-yaw",
        type=float,
        default=None,
        help="Target yaw angle in degrees (e.g. 30 for left, -30 for right). Automatically enables in-place rotation.",
    )
    parser.add_argument(
        "--target-yaw-tol",
        type=float,
        default=2.0,
        help="Target yaw tolerance in degrees (default: 2.0).",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=60,
        help="Maximum gait cycles allowed when reaching a target yaw (default: 60).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="ZeroMQ Remote API port (default: auto-detect 23000 then 23001).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_gait(
        cycles=args.cycles,
        cycle_seconds=args.cycle_seconds,
        turn_rate=args.turn_rate,
        in_place=args.in_place,
        direction=args.direction,
        target_yaw=args.target_yaw,
        target_yaw_tol=args.target_yaw_tol,
        max_cycles=args.max_cycles,
        port=args.port,
    )


if __name__ == "__main__":
    main()
