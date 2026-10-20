
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math

client = RemoteAPIClient()
sim = client.require("sim")

if sim.getSimulationState() == sim.simulation_stopped:
    sim.startSimulation()
    time.sleep(0.5)

legs = []

for i in range(1, 7):
    hip = sim.getObject(f"/HexapodBody/HipJoint_{i}")
    knee = sim.getObject(
        f"/HexapodBody/HipJoint_{i}/UpperLeg_{i}/KneeJoint_{i}"
    )
    legs.append((hip, knee))

cycle_seconds = 8.0
hip_range = 0.20
knee_bend = 0.35

print("Tripod lift-and-swing test started. Press Ctrl+C to stop.")

start = time.time()

try:
    while True:
        t = (time.time() - start) % cycle_seconds

        for i, (hip, knee) in enumerate(legs):
            # Tripod B starts half a cycle after tripod A.
            phase = (t / cycle_seconds + (0.5 if i % 2 else 0.0)) % 1.0

            if phase < 0.5:
                # Stance: sweep the hip while keeping the knee straighter.
                u = phase / 0.5
                hip_angle = hip_range * (1.0 - 2.0 * u)
                knee_angle = 0.05
            else:
                # Swing: return the hip while bending the knee.
                u = (phase - 0.5) / 0.5
                hip_angle = hip_range * (-1.0 + 2.0 * u)
                knee_angle = 0.05 + knee_bend * math.sin(math.pi * u)

            sim.setJointPosition(hip, hip_angle)
            sim.setJointPosition(knee, knee_angle)

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nGait test stopped.")
