# CoppeliaSim Hexapod

The project provides separate kinematic and physics modes. Physics mode uses
six radial upper-leg frames, two revolute joints per leg, and a tripod gait
with two-link inverse kinematics.

## Leg layout and frames

The previous layout placed all legs at the same left/right Y offsets and aimed
the front and middle upper links in the same +X direction. On each side, the
front and middle 160 mm boxes therefore overlapped by 55 mm in world space.
This was actual shape intersection, not only a camera-projection artifact.
Additionally, those upper shapes were independently posed in world
coordinates before parenting, so their visible axes were not defined by the
joint-local transforms.

The generator now places front and rear hips at splayed body-corner rays and
middle hips at the lateral body edges. Hip/knee axes are tangent to each leg's
radial sagittal plane. Upper segments are aligned to the hip's local X axis;
lower segments extend down the hip/knee local Y axis. Their local transforms
are set through the hierarchy rather than rotated separately. The gait
projects body-forward stride into each leg's own radial plane; left/right
pairs retain mirrored world poses and the 1/3/5 versus 2/4/6 tripod phasing.

## Requirements

- Windows and Python
- CoppeliaSim with the ZeroMQ Remote API server enabled
- The `coppeliasim-zmqremoteapi-client` package installed in the Python
  interpreter used to run the scripts

Install the client package if needed:

```powershell
python -m pip install coppeliasim-zmqremoteapi-client
```

## Stage 1: build and inspect geometry

1. Start CoppeliaSim and open the scene in which the robot should be generated.
2. Stop the simulation.
3. From this project directory, run:

   ```powershell
   python build_hexapod.py
   ```

   The generator replaces the previously generated `/HexapodBody` subtree and
   `/Ground`, then creates three mirrored left/right leg pairs. Front and rear
   upper legs splay diagonally; middle legs extend laterally. Each hip/knee
   axis is tangent to its leg's radial plane. Legs 1, 3, 5 and 2, 4, 6 remain
   the alternating tripod groups. Upper and lower segments are positioned in
   their parent joint frames, and neutral feet reach the ground plane.
4. Inspect the leg layout and joint directions in CoppeliaSim before starting
   further tests.

To test API connectivity or a single kinematic leg:

```powershell
python hexapod_test.py
python test_joints.py
```

The default build intentionally creates **kinematic** joints for independent
joint testing. Build the physics-configured scene separately:

```powershell
python build_hexapod.py --mode physics
```

Physics mode sets dynamic and respondable shape properties, per-link masses,
global-only collision masks (to avoid internal self-collision), dynamic
position-controlled joints, bounded joint intervals, and 1.2 N m motor limits.
The physics scene starts in a straight-leg stance. Run the bounded, synchronous
settling/contact check before gait testing:

```powershell
python test_physics_stance.py
```

Run the slow physics-based tripod gait with two-link inverse kinematics:

```powershell
python tripod_gait_physics.py --cycles 3 --cycle-seconds 6
```

The gait uses a 65% stance phase, a 50 mm backward body-forward sweep projected
into each leg's radial plane, a lifted swing return, and alternating leg groups
1/3/5 and 2/4/6. The shorter sweep keeps the requested IK poses within the
existing joint limits during the body-height variation observed in physics. It
uses synchronous stepping and stops the simulation on normal completion or
`Ctrl+C`. It also reports measured body displacement; that output, not the
joint targets alone, is the movement check.

The existing `test_joints.py`, `tripod_gait.py`, and `tripod_gait_v2.py` scripts
remain kinematic demonstrations and must not be treated as physics gait tests.
In particular, joint animation in those scripts is not evidence of physical
walking.

The geometry, segment-clearance checks, generator hierarchy, gait trajectory,
and inverse-kinematics math can be checked without CoppeliaSim:

```powershell
python -m unittest -v test_hexapod_geometry.py
python -m py_compile build_hexapod.py tripod_gait_physics.py test_hexapod_geometry.py test_physics_stance.py
```

The tests do not by themselves verify collision, ground contact, physics
stability, or forward movement.

The build script sets finite joint limits in both modes but never starts the
simulator. The old kinematic demos remain available, including
`test_joints.py`, `tripod_gait.py`, and `tripod_gait_v2.py`; they remain
kinematic demonstrations, not physics gait tests.

## Troubleshooting

- **`ModuleNotFoundError: coppeliasim_zmqremoteapi_client`**: install the package
  into the same Python interpreter used by the command above.
- **Remote API connection refused or timed out**: start CoppeliaSim and confirm
  that its ZeroMQ Remote API server is enabled and listening.
- **Generator reports that simulation must be stopped**: stop the current
  simulation in CoppeliaSim, then rerun the generator.
- **Robot geometry appears duplicated**: the generator removes the generated
  `/HexapodBody` subtree and `/Ground` before rebuilding. If those objects were
  renamed or reparented outside that subtree, remove the renamed leftovers
  manually before rerunning.

## Verification stages

1. **Geometry:** run the unit tests, then `python build_hexapod.py` and inspect
   the kinematic frames.
2. **Physics/contact:** run `python build_hexapod.py --mode physics`, then
   `python test_physics_stance.py`. The check verifies dynamics, motor limits,
   stable body height, and an actual ground contact/reaction force before gait
   testing.
3. **Gait/movement:** run the physics gait command above and inspect the
   reported body displacement as well as the simulation.

The previous walking versions are preserved in
`build_hexapod.py.pre_overlap_fix.bak` and
`tripod_gait_physics.py.pre_overlap_fix.bak`. The current simulator was not
available during this geometry update, so no live rebuild, contact, or walking
result is claimed. The automated tests cover the neutral geometric frames and
the IK path; run the three simulator commands above to verify dynamic behavior
in your CoppeliaSim scene.
