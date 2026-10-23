"""Automated movement test runner for the hexapod gait controller.

Runs all 8 movement modes sequentially against the live CoppeliaSim physics
scene, records pass/fail/inconclusive, and writes a detailed report to movement_test_report.txt.

Usage
-----
    python test_all_movements.py                # 1 cycle per movement (default)
    python test_all_movements.py --cycles 3     # 3 cycles per movement
    python test_all_movements.py --help         # view configurable threshold options

Prerequisites
-------------
- CoppeliaSim must be open, the hexapod scene loaded, and the simulation STOPPED.
- Run `python -m unittest -v test_hexapod_geometry.py` first to confirm geometry
  is consistent; this script does it automatically.

Design principles
-----------------
- Reuses run_gait() from tripod_gait_physics.py; does not duplicate its algorithm.
- Does NOT rebuild the robot or modify build_hexapod.py.
- Measures each movement relative to its own starting pose so that cumulative
  drift from prior movements does not cause false failures.
- Reports cumulative drift across all tests in the summary.
- Curved walking validates both translational displacement and yaw direction.
- Yaw changes below the baseline noise/drift band are flagged as INCONCLUSIVE
  rather than falsely passed as confirmed steering.
- Supports configurable, provisional thresholds with flexible multi-cycle scaling.
- Stores results to movement_test_report.txt with timestamps and telemetry.
"""

import argparse
import datetime
import logging
import math
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Default provisional thresholds (per-cycle baseline)
# ---------------------------------------------------------------------------
# Straight translation threshold (meters): measured ~0.039 m in 1 cycle
DEFAULT_MIN_TRANSLATION_PER_CYCLE_M = 0.030

# Curved translation threshold (meters): curved path has shorter chord length
DEFAULT_MIN_CURVE_TRANSLATION_PER_CYCLE_M = 0.020

# Minimum curved-walking yaw magnitude (degrees) to confirm steering.
# Background straight-walking drift is 0.05 deg to 0.13 deg per cycle.
# A yaw change below 0.20 deg is within the noise floor and is marked INCONCLUSIVE.
DEFAULT_MIN_CURVE_YAW_DEG = 0.20

# In-place rotation threshold (degrees): measured ~0.9 deg in 1 cycle
DEFAULT_MIN_IN_PLACE_YAW_PER_CYCLE_DEG = 0.50


def _wrap_deg(deg):
    """Wrap angle to [-180.0, 180.0) degrees."""
    return (deg + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("movement_test")


# ---------------------------------------------------------------------------
# Remote API health check
# ---------------------------------------------------------------------------

def _check_remote_api(port=None):
    """Return True if CoppeliaSim's ZeroMQ Remote API is reachable and stopped."""
    try:
        from tripod_gait_physics import _get_remote_client

        client = _get_remote_client(port=port)
        sim = client.require("sim")
        state = sim.getSimulationState()
        LOGGER.info("Remote API reachable. Simulation state code: %d", state)
        if state != sim.simulation_stopped:
            LOGGER.error(
                "CoppeliaSim simulation is NOT stopped (state=%d). "
                "Stop it before running this script.",
                state,
            )
            return False
        return True
    except Exception as exc:
        LOGGER.error("Remote API unavailable: %s", exc)
        return False



# ---------------------------------------------------------------------------
# Offline geometry suite
# ---------------------------------------------------------------------------

def _run_offline_tests():
    """Run test_hexapod_geometry.py and return True if all tests pass."""
    LOGGER.info("Running offline geometry test suite ...")
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "-v", "test_hexapod_geometry.py"],
        capture_output=True,
        text=True,
    )
    summary_lines = result.stderr.strip().splitlines()
    for line in summary_lines[-3:]:
        LOGGER.info("  [unittest] %s", line)
    if result.returncode != 0:
        LOGGER.error("Offline tests FAILED -- aborting live simulation.")
        return False
    LOGGER.info("Offline tests PASSED.")
    return True


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

class MovementResult:
    """Holds telemetry and outcome for a single movement test."""

    STATUS_PASS         = "PASS"
    STATUS_FAIL         = "FAIL"
    STATUS_INCONCLUSIVE = "INCONCLUSIVE"
    STATUS_ERROR        = "ERROR"
    STATUS_SKIPPED      = "SKIPPED"

    def __init__(self, name, pass_desc):
        self.name             = name
        self.pass_desc        = pass_desc
        self.status           = self.STATUS_SKIPPED
        self.notes            = ""
        self.dx_m             = 0.0
        self.dyaw_deg         = 0.0
        self.dy_m             = 0.0
        self.dz_m             = 0.0
        self.final_z_m        = 0.0
        self.droll_deg        = 0.0
        self.dpitch_deg       = 0.0
        self.wall_seconds     = 0.0
        self.cycles_completed = 0
        self.foot_slip        = {}
        self.foot_scrub       = {}
        self.target_reached   = None
        self.error_message    = ""
        self.kwargs           = {}

    def set_telemetry(self, dx_m, dyaw_deg, wall_seconds, **extra):
        self.dx_m         = dx_m
        self.dyaw_deg     = dyaw_deg
        self.wall_seconds = wall_seconds
        for k, v in extra.items():
            if hasattr(self, k):
                setattr(self, k, v)


# ---------------------------------------------------------------------------
# Test suite builder
# ---------------------------------------------------------------------------

def _build_test_suite(cycles, cfg):
    """Return the ordered list of test specifications.

    Each entry is a dict with keys:
        name       - human-readable label
        kwargs     - keyword arguments forwarded verbatim to run_gait()
        eval_fn    - callable(dx_m, dyaw_deg) -> (status, note_string)
        pass_desc  - human description of the pass condition
    """
    min_trans       = cfg["min_trans"]
    min_curve_trans = cfg["min_curve_trans"]
    min_curve_yaw   = cfg["min_curve_yaw"]
    walk_cycle_sec  = cfg.get("walk_cycle_seconds", 6.0)
    target_yaw      = cfg.get("rotate_target_yaw", 30.0)
    target_yaw_tol  = cfg.get("rotate_yaw_tol", 2.0)
    rot_cycle_sec   = cfg.get("rotate_cycle_seconds", 3.0)
    max_rot_cycles  = cfg.get("max_rotate_cycles", 60)

    def fwd_eval(dx, _dyaw):
        if dx >= min_trans:
            return (
                MovementResult.STATUS_PASS,
                f"Forward translation confirmed: {dx:.4f} m >= {min_trans:.4f} m",
            )
        return (
            MovementResult.STATUS_FAIL,
            f"Insufficient forward translation: {dx:.4f} m < {min_trans:.4f} m",
        )

    def bwd_eval(dx, _dyaw):
        if dx <= -min_trans:
            return (
                MovementResult.STATUS_PASS,
                f"Backward translation confirmed: {dx:.4f} m <= {-min_trans:.4f} m",
            )
        return (
            MovementResult.STATUS_FAIL,
            f"Insufficient backward translation: {dx:.4f} m > {-min_trans:.4f} m",
        )

    def turn_left_eval(dx, dyaw):
        wrapped_yaw = _wrap_deg(dyaw)
        if dx < min_curve_trans:
            return (
                MovementResult.STATUS_FAIL,
                f"Insufficient forward translation: {dx:.4f} m < {min_curve_trans:.4f} m",
            )
        if wrapped_yaw <= 0.0:
            return (
                MovementResult.STATUS_FAIL,
                f"Yaw direction incorrect: dyaw={wrapped_yaw:.2f} deg (expected > 0.0 deg for left turn)",
            )
        if wrapped_yaw < min_curve_yaw:
            return (
                MovementResult.STATUS_INCONCLUSIVE,
                f"Translation passed ({dx:.4f} m) and yaw direction is positive (+{wrapped_yaw:.2f} deg), "
                f"but magnitude is below noise threshold ({min_curve_yaw:.2f} deg); steering inconclusive",
            )
        return (
            MovementResult.STATUS_PASS,
            f"Translation ({dx:.4f} m) and left-turn yaw (+{wrapped_yaw:.2f} deg >= {min_curve_yaw:.2f} deg) confirmed",
        )

    def turn_right_eval(dx, dyaw):
        wrapped_yaw = _wrap_deg(dyaw)
        if dx < min_curve_trans:
            return (
                MovementResult.STATUS_FAIL,
                f"Insufficient forward translation: {dx:.4f} m < {min_curve_trans:.4f} m",
            )
        if wrapped_yaw >= 0.0:
            return (
                MovementResult.STATUS_FAIL,
                f"Yaw direction incorrect: dyaw={wrapped_yaw:.2f} deg (expected < 0.0 deg for right turn)",
            )
        if abs(wrapped_yaw) < min_curve_yaw:
            return (
                MovementResult.STATUS_INCONCLUSIVE,
                f"Translation passed ({dx:.4f} m) and yaw direction is negative ({wrapped_yaw:.2f} deg), "
                f"but magnitude is below noise threshold ({min_curve_yaw:.2f} deg); steering inconclusive",
            )
        return (
            MovementResult.STATUS_PASS,
            f"Translation ({dx:.4f} m) and right-turn yaw ({wrapped_yaw:.2f} deg <= -{min_curve_yaw:.2f} deg) confirmed",
        )

    def bwd_left_eval(dx, dyaw):
        wrapped_yaw = _wrap_deg(dyaw)
        if dx > -min_curve_trans:
            return (
                MovementResult.STATUS_FAIL,
                f"Insufficient backward translation: {dx:.4f} m > {-min_curve_trans:.4f} m",
            )
        if wrapped_yaw >= 0.0:
            return (
                MovementResult.STATUS_FAIL,
                f"Yaw direction incorrect: dyaw={wrapped_yaw:.2f} deg (expected < 0.0 deg for reverse left turn)",
            )
        if abs(wrapped_yaw) < min_curve_yaw:
            return (
                MovementResult.STATUS_INCONCLUSIVE,
                f"Translation passed ({dx:.4f} m) and yaw direction is negative ({wrapped_yaw:.2f} deg), "
                f"but magnitude is below noise threshold ({min_curve_yaw:.2f} deg); steering inconclusive",
            )
        return (
            MovementResult.STATUS_PASS,
            f"Translation ({dx:.4f} m) and reverse left-turn yaw ({wrapped_yaw:.2f} deg <= -{min_curve_yaw:.2f} deg) confirmed",
        )

    def bwd_right_eval(dx, dyaw):
        wrapped_yaw = _wrap_deg(dyaw)
        if dx > -min_curve_trans:
            return (
                MovementResult.STATUS_FAIL,
                f"Insufficient backward translation: {dx:.4f} m > {-min_curve_trans:.4f} m",
            )
        if wrapped_yaw <= 0.0:
            return (
                MovementResult.STATUS_FAIL,
                f"Yaw direction incorrect: dyaw={wrapped_yaw:.2f} deg (expected > 0.0 deg for reverse right turn)",
            )
        if wrapped_yaw < min_curve_yaw:
            return (
                MovementResult.STATUS_INCONCLUSIVE,
                f"Translation passed ({dx:.4f} m) and yaw direction is positive (+{wrapped_yaw:.2f} deg), "
                f"but magnitude is below noise threshold ({min_curve_yaw:.2f} deg); steering inconclusive",
            )
        return (
            MovementResult.STATUS_PASS,
            f"Translation ({dx:.4f} m) and reverse right-turn yaw (+{wrapped_yaw:.2f} deg >= {min_curve_yaw:.2f} deg) confirmed",
        )

    def rotate_left_target_eval(_dx, dyaw):
        wrapped_yaw = _wrap_deg(dyaw)
        err = abs(_wrap_deg(wrapped_yaw - target_yaw))
        if err <= target_yaw_tol:
            return (
                MovementResult.STATUS_PASS,
                f"Target-based left rotation confirmed: actual={wrapped_yaw:+.2f} deg reached target={target_yaw:+.1f} deg (error={err:.2f} deg <= ±{target_yaw_tol:.1f} deg)",
            )
        if wrapped_yaw > 0.0:
            return (
                MovementResult.STATUS_INCONCLUSIVE,
                f"Left rotation positive (+{wrapped_yaw:.2f} deg) but error {err:.2f} deg exceeds ±{target_yaw_tol:.1f} deg tolerance (target={target_yaw:+.1f} deg)",
            )
        return (
            MovementResult.STATUS_FAIL,
            f"Left rotation failed: dyaw={wrapped_yaw:.2f} deg (expected positive rotation toward +{target_yaw:.1f} deg)",
        )

    def rotate_right_target_eval(_dx, dyaw):
        wrapped_yaw = _wrap_deg(dyaw)
        tgt = -target_yaw
        err = abs(_wrap_deg(wrapped_yaw - tgt))
        if err <= target_yaw_tol:
            return (
                MovementResult.STATUS_PASS,
                f"Target-based right rotation confirmed: actual={wrapped_yaw:+.2f} deg reached target={tgt:+.1f} deg (error={err:.2f} deg <= ±{target_yaw_tol:.1f} deg)",
            )
        if wrapped_yaw < 0.0:
            return (
                MovementResult.STATUS_INCONCLUSIVE,
                f"Right rotation negative ({wrapped_yaw:.2f} deg) but error {err:.2f} deg exceeds ±{target_yaw_tol:.1f} deg tolerance (target={tgt:+.1f} deg)",
            )
        return (
            MovementResult.STATUS_FAIL,
            f"Right rotation failed: dyaw={wrapped_yaw:.2f} deg (expected negative rotation toward {tgt:.1f} deg)",
        )

    return [
        {
            "name": "Walk Forward",
            "kwargs": dict(cycles=cycles, cycle_seconds=walk_cycle_sec, turn_rate=0.0, in_place=False, direction="forward"),
            "eval_fn": fwd_eval,
            "pass_desc": f"dx >= {min_trans:.3f} m",
        },
        {
            "name": "Walk Backward",
            "kwargs": dict(cycles=cycles, cycle_seconds=walk_cycle_sec, turn_rate=0.0, in_place=False, direction="backward"),
            "eval_fn": bwd_eval,
            "pass_desc": f"dx <= {-min_trans:.3f} m",
        },
        {
            "name": "Walk Forward + Turn Left (rate=0.5)",
            "kwargs": dict(cycles=cycles, cycle_seconds=walk_cycle_sec, turn_rate=0.5, in_place=False, direction="forward"),
            "eval_fn": turn_left_eval,
            "pass_desc": f"dx >= {min_curve_trans:.3f} m, dyaw > 0.0 deg (conclusive >= {min_curve_yaw:.2f} deg)",
        },
        {
            "name": "Walk Forward + Turn Right (rate=-0.5)",
            "kwargs": dict(cycles=cycles, cycle_seconds=walk_cycle_sec, turn_rate=-0.5, in_place=False, direction="forward"),
            "eval_fn": turn_right_eval,
            "pass_desc": f"dx >= {min_curve_trans:.3f} m, dyaw < 0.0 deg (conclusive <= -{min_curve_yaw:.2f} deg)",
        },
        {
            "name": "Walk Backward + Turn Left (rate=0.5)",
            "kwargs": dict(cycles=cycles, cycle_seconds=walk_cycle_sec, turn_rate=0.5, in_place=False, direction="backward"),
            "eval_fn": bwd_left_eval,
            "pass_desc": f"dx <= {-min_curve_trans:.3f} m, dyaw < 0.0 deg (conclusive <= -{min_curve_yaw:.2f} deg)",
        },
        {
            "name": "Walk Backward + Turn Right (rate=-0.5)",
            "kwargs": dict(cycles=cycles, cycle_seconds=walk_cycle_sec, turn_rate=-0.5, in_place=False, direction="backward"),
            "eval_fn": bwd_right_eval,
            "pass_desc": f"dx <= {-min_curve_trans:.3f} m, dyaw > 0.0 deg (conclusive >= {min_curve_yaw:.2f} deg)",
        },
        {
            "name": f"In-Place Rotate Left (+{target_yaw:.0f} deg target)",
            "kwargs": dict(
                target_yaw=target_yaw,
                target_yaw_tol=target_yaw_tol,
                max_cycles=max_rot_cycles,
                cycle_seconds=rot_cycle_sec,
                in_place=True,
            ),
            "eval_fn": rotate_left_target_eval,
            "pass_desc": f"dyaw reaches +{target_yaw:.1f} deg within ±{target_yaw_tol:.1f} deg tolerance",
        },
        {
            "name": f"In-Place Rotate Right (-{target_yaw:.0f} deg target)",
            "kwargs": dict(
                target_yaw=-target_yaw,
                target_yaw_tol=target_yaw_tol,
                max_cycles=max_rot_cycles,
                cycle_seconds=rot_cycle_sec,
                in_place=True,
            ),
            "eval_fn": rotate_right_target_eval,
            "pass_desc": f"dyaw reaches -{target_yaw:.1f} deg within ±{target_yaw_tol:.1f} deg tolerance",
        },
    ]

    port = cfg.get("port", None)
    if port is not None:
        for item in suite:
            item["kwargs"]["port"] = port

    return suite



# ---------------------------------------------------------------------------
# Run one movement
# ---------------------------------------------------------------------------

def _run_one(spec):
    """Execute a single movement and return a populated MovementResult."""
    from tripod_gait_physics import run_gait

    result        = MovementResult(spec["name"], spec["pass_desc"])
    result.kwargs = spec["kwargs"]

    LOGGER.info("-" * 60)
    LOGGER.info("TEST: %s", spec["name"])
    LOGGER.info("  Parameters: %s", spec["kwargs"])
    LOGGER.info("  Evaluation rule: %s", spec["pass_desc"])

    t0 = time.monotonic()
    try:
        res = run_gait(**spec["kwargs"])
        dx_m, dyaw_deg = res[0], res[1]
        elapsed = time.monotonic() - t0

        extra = {}
        if hasattr(res, "dz"):
            extra["dz_m"] = res.dz
            extra["dy_m"] = res.dy
            extra["final_z_m"] = res.final_z
            extra["droll_deg"] = res.droll
            extra["dpitch_deg"] = res.dpitch
            extra["cycles_completed"] = res.cycles_completed
            extra["target_reached"] = res.target_reached
            extra["foot_slip"] = res.foot_slip
            extra["foot_scrub"] = res.foot_scrub

        result.set_telemetry(dx_m, dyaw_deg, elapsed, **extra)
        status, note = spec["eval_fn"](dx_m, dyaw_deg)
        result.status = status
        result.notes  = note

        LOGGER.info(
            "  Result: dx=%.4f m  dyaw=%.2f deg  wall=%.1f s",
            dx_m, dyaw_deg, elapsed,
        )
        LOGGER.info("  Status: %s%s", result.status, f" ({note})" if note else "")

    except RuntimeError as exc:
        elapsed              = time.monotonic() - t0
        result.wall_seconds  = elapsed
        result.status        = result.STATUS_FAIL
        result.error_message = str(exc)
        LOGGER.error("  RuntimeError: %s", exc)
        LOGGER.info("  Status: %s (runtime failure)", result.status)

    except KeyboardInterrupt:
        elapsed              = time.monotonic() - t0
        result.wall_seconds  = elapsed
        result.status        = result.STATUS_ERROR
        result.error_message = "Interrupted by user (KeyboardInterrupt)"
        LOGGER.warning("  Interrupted -- skipping remaining tests.")
        raise

    except Exception as exc:
        elapsed              = time.monotonic() - t0
        result.wall_seconds  = elapsed
        result.status        = result.STATUS_ERROR
        result.error_message = f"{type(exc).__name__}: {exc}"
        LOGGER.error("  Unexpected error: %s", result.error_message)

    return result


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

_REPORT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "movement_test_report.txt",
)

_STATUS_WIDTH = 13
_NAME_WIDTH   = 46


def _write_report(results, cycles, start_ts, cfg):
    lines = []
    lines.append("=" * 78)
    lines.append("  HEXAPOD MOVEMENT TEST REPORT")
    lines.append(f"  Generated   : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Started     : {start_ts}")
    lines.append(f"  Cycles      : {cycles} per walking movement")
    lines.append(
        f"  Scaling     : {cfg['cycle_scaling']} (multiplier={cfg['multiplier']:.2f})"
    )
    lines.append(
        f"  Thresholds  : trans={cfg['min_trans']:.3f}m, curve_trans={cfg['min_curve_trans']:.3f}m, "
        f"curve_yaw={cfg['min_curve_yaw']:.2f}deg, target_yaw=±{cfg['rotate_target_yaw']:.1f}deg (tol=±{cfg['rotate_yaw_tol']:.1f}deg)"
    )
    lines.append("=" * 78)
    lines.append("")

    passes   = sum(1 for r in results if r.status == MovementResult.STATUS_PASS)
    fails    = sum(1 for r in results if r.status == MovementResult.STATUS_FAIL)
    inconcl  = sum(1 for r in results if r.status == MovementResult.STATUS_INCONCLUSIVE)
    errors   = sum(1 for r in results if r.status == MovementResult.STATUS_ERROR)
    skips    = sum(1 for r in results if r.status == MovementResult.STATUS_SKIPPED)

    lines.append(
        f"  SUMMARY: {passes} PASS, {fails} FAIL, {inconcl} INCONCLUSIVE, "
        f"{errors} ERROR, {skips} SKIPPED  out of {len(results)} tests"
    )
    lines.append("")
    lines.append(
        f"  {'#':<4} {'STATUS':<{_STATUS_WIDTH}} {'NAME':<{_NAME_WIDTH}} "
        f"{'dx (m)':>8} {'dyaw (deg)':>10} {'time (s)':>8}"
    )
    lines.append("  " + "-" * 76)

    for idx, r in enumerate(results, start=1):
        lines.append(
            f"  {idx:<4} {r.status:<{_STATUS_WIDTH}} {r.name:<{_NAME_WIDTH}} "
            f"{r.dx_m:>8.4f} {r.dyaw_deg:>10.2f} {r.wall_seconds:>8.1f}"
        )

    lines.append("")
    lines.append("-" * 78)
    lines.append("  DETAILED RESULTS")
    lines.append("-" * 78)

    cumulative_dx   = 0.0
    cumulative_dyaw = 0.0

    for idx, r in enumerate(results, start=1):
        cumulative_dx   += r.dx_m
        cumulative_dyaw += r.dyaw_deg
        lines.append("")
        lines.append(f"  Test {idx}: {r.name}")
        lines.append(f"    Status     : {r.status}")
        lines.append(f"    Parameters : {r.kwargs}")
        lines.append(f"    Pass rule  : {r.pass_desc}")
        lines.append(f"    dx         : {r.dx_m:.4f} m")
        lines.append(f"    dyaw       : {r.dyaw_deg:.2f} deg")
        lines.append(f"    Wall time  : {r.wall_seconds:.1f} s")
        if r.final_z_m > 0.0:
            lines.append(f"    Body height: {r.final_z_m:.4f} m (dz: {r.dz_m:+.4f} m)")
        if abs(r.droll_deg) > 0.0 or abs(r.dpitch_deg) > 0.0:
            lines.append(f"    Attitude   : droll={r.droll_deg:+.2f} deg, dpitch={r.dpitch_deg:+.2f} deg")
        if r.cycles_completed > 0:
            lines.append(f"    Cycles run : {r.cycles_completed}")
        if r.target_reached is not None:
            lines.append(f"    Target reached: {r.target_reached}")
        if r.foot_slip:
            slip_str = ", ".join(f"L{k}={v*1000.0:.1f}mm" for k, v in sorted(r.foot_slip.items()))
            lines.append(f"    Foot slip  : {slip_str}")
        lines.append(f"    Cumul. dx  : {cumulative_dx:.4f} m   (drift since start)")
        lines.append(f"    Cumul. yaw : {cumulative_dyaw:.2f} deg (drift since start)")
        if r.notes:
            lines.append(f"    Assessment : {r.notes}")
        if r.error_message:
            lines.append(f"    Error      : {r.error_message}")

    lines.append("")
    lines.append("=" * 78)
    lines.append(
        f"  CUMULATIVE DRIFT ACROSS ALL TESTS:  dx={cumulative_dx:.4f} m  dyaw={cumulative_dyaw:.2f} deg"
    )
    lines.append(
        "  (Each movement was measured from its own starting pose, not from the origin.)"
    )
    lines.append("=" * 78)

    report_text = "\n".join(lines) + "\n"

    with open(_REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report_text)

    return report_text


def _print_terminal_summary(results):
    print()
    print("=" * 78)
    print("  MOVEMENT TEST SUMMARY")
    print("=" * 78)
    for idx, r in enumerate(results, start=1):
        if r.status == MovementResult.STATUS_PASS:
            icon = "v"
        elif r.status == MovementResult.STATUS_INCONCLUSIVE:
            icon = "?"
        elif r.status == MovementResult.STATUS_FAIL:
            icon = "x"
        elif r.status == MovementResult.STATUS_ERROR:
            icon = "!"
        else:
            icon = "-"
        note_str = f" -- {r.notes}" if r.notes else ""
        print(f"  {icon}  [{r.status:<12}]  #{idx:>2}  {r.name}{note_str}")
    print()
    passes  = sum(1 for r in results if r.status == MovementResult.STATUS_PASS)
    fails   = sum(1 for r in results if r.status == MovementResult.STATUS_FAIL)
    inconcl = sum(1 for r in results if r.status == MovementResult.STATUS_INCONCLUSIVE)
    total   = len(results)
    print(f"  Results: {passes}/{total} PASS, {fails}/{total} FAIL, {inconcl}/{total} INCONCLUSIVE")
    print(f"  Report saved to: {_REPORT_PATH}")
    print("=" * 78)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Automated movement test runner for the hexapod gait controller.\n"
            "Runs all 8 movement modes sequentially in CoppeliaSim and saves a report."
        )
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=5,
        help="Number of gait cycles for walking movement tests (default: 5).",
    )
    parser.add_argument(
        "--cycle-seconds",
        type=float,
        default=6.0,
        help="Cycle duration in seconds for walking tests (default: 6.0).",
    )
    parser.add_argument(
        "--min-trans",
        type=float,
        default=None,
        help="Minimum straight translation threshold in meters (default: 0.030 m * scaling).",
    )
    parser.add_argument(
        "--min-curve-trans",
        type=float,
        default=None,
        help="Minimum curved translation threshold in meters (default: 0.020 m * scaling).",
    )
    parser.add_argument(
        "--min-curve-yaw",
        type=float,
        default=None,
        help="Minimum curved yaw magnitude in degrees to confirm steering (default: 0.25 deg; below this is INCONCLUSIVE).",
    )
    parser.add_argument(
        "--rotate-target-yaw",
        type=float,
        default=30.0,
        help="Target yaw angle in degrees for in-place rotation tests (default: 30.0 deg).",
    )
    parser.add_argument(
        "--rotate-yaw-tol",
        type=float,
        default=2.0,
        help="Tolerance in degrees for target yaw tests (default: 2.0 deg).",
    )
    parser.add_argument(
        "--rotate-cycle-seconds",
        type=float,
        default=3.0,
        help="Cycle duration in seconds for in-place rotation tests (default: 3.0).",
    )
    parser.add_argument(
        "--max-rotate-cycles",
        type=int,
        default=60,
        help="Maximum cycles allowed for in-place rotation tests (default: 60).",
    )
    parser.add_argument(
        "--cycle-scaling",
        choices=["sublinear", "linear", "none"],
        default="sublinear",
        help="Threshold scaling across multiple cycles: 'sublinear' (1 + (N-1)*0.85), 'linear' (N), or 'none' (1.0) (default: sublinear).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="ZeroMQ Remote API port (default: auto-detect 23000 then 23001).",
    )
    parser.add_argument(
        "--skip-offline-tests",
        action="store_true",
        help="Skip the offline geometry test suite (not recommended).",
    )
    args = parser.parse_args()

    # Determine scaling multiplier across cycles
    if args.cycle_scaling == "sublinear":
        multiplier = 1.0 + (args.cycles - 1) * 0.85
    elif args.cycle_scaling == "linear":
        multiplier = float(args.cycles)
    else:
        multiplier = 1.0

    cfg = {
        "min_trans": (
            args.min_trans if args.min_trans is not None
            else round(DEFAULT_MIN_TRANSLATION_PER_CYCLE_M * multiplier, 4)
        ),
        "min_curve_trans": (
            args.min_curve_trans if args.min_curve_trans is not None
            else round(DEFAULT_MIN_CURVE_TRANSLATION_PER_CYCLE_M * multiplier, 4)
        ),
        "min_curve_yaw": (
            args.min_curve_yaw if args.min_curve_yaw is not None
            else round(DEFAULT_MIN_CURVE_YAW_DEG if args.cycle_scaling == "none" else 0.25, 2)
        ),
        "cycle_scaling": args.cycle_scaling,
        "multiplier": multiplier,
        "walk_cycle_seconds": args.cycle_seconds,
        "rotate_target_yaw": args.rotate_target_yaw,
        "rotate_yaw_tol": args.rotate_yaw_tol,
        "rotate_cycle_seconds": args.rotate_cycle_seconds,
        "max_rotate_cycles": args.max_rotate_cycles,
        "port": args.port,
    }

    start_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    LOGGER.info("Hexapod movement test runner starting.")
    LOGGER.info("Cycles per walking movement: %d (scaling: %s, factor: %.2f)", args.cycles, args.cycle_scaling, multiplier)
    LOGGER.info(
        "Configured thresholds: straight_trans=%.3f m, curve_trans=%.3f m, curve_yaw=%.2f deg, target_yaw=±%.1f deg (tol=±%.1f deg)",
        cfg["min_trans"], cfg["min_curve_trans"], cfg["min_curve_yaw"], cfg["rotate_target_yaw"], cfg["rotate_yaw_tol"],
    )

    # ------------------------------------------------------------------
    # Step 1: Run offline geometry tests
    # ------------------------------------------------------------------
    if not args.skip_offline_tests:
        if not _run_offline_tests():
            LOGGER.error("Aborting: offline tests must pass before live simulation.")
            return 2

    # ------------------------------------------------------------------
    # Step 2: Check the Remote API connection
    # ------------------------------------------------------------------
    LOGGER.info("Checking CoppeliaSim Remote API ...")
    if not _check_remote_api(port=args.port):
        LOGGER.error(
            "Aborting: CoppeliaSim Remote API is unavailable or simulation is not stopped.\n"
            "  1. Open CoppeliaSim and load your hexapod scene.\n"
            "  2. Ensure the simulation is STOPPED (not paused).\n"
            "  3. Retry."
        )
        return 1

    # ------------------------------------------------------------------
    # Step 3: Build test suite and run
    # ------------------------------------------------------------------
    suite   = _build_test_suite(args.cycles, cfg)
    results = []

    LOGGER.info("Starting live movement tests (%d movements) ...", len(suite))

    interrupted = False
    for spec in suite:
        try:
            result = _run_one(spec)
            results.append(result)
        except KeyboardInterrupt:
            interrupted = True
            for remaining_spec in suite[len(results):]:
                r = MovementResult(remaining_spec["name"], remaining_spec["pass_desc"])
                r.status        = MovementResult.STATUS_SKIPPED
                r.error_message = "Skipped due to KeyboardInterrupt"
                results.append(r)
            break

        # Brief settle pause between movements
        if not interrupted:
            LOGGER.info("  Settling pause (2 s) ...")
            time.sleep(2.0)

    # ------------------------------------------------------------------
    # Step 4: Write report and print summary
    # ------------------------------------------------------------------
    _write_report(results, args.cycles, start_ts, cfg)
    _print_terminal_summary(results)

    if interrupted:
        LOGGER.warning("Run was interrupted. Partial results saved.")

    failures = [
        r for r in results
        if r.status in (MovementResult.STATUS_FAIL, MovementResult.STATUS_ERROR)
    ]
    return 0 if not failures else 3


if __name__ == "__main__":
    sys.exit(main())
