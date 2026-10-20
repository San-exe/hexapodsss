
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math

client = RemoteAPIClient()
sim = client.require("sim")

# Start simulation if necessary
if sim.getSimulationState() == sim.simulation_stopped:
    sim.startSimulation()
    time.sleep(0.5)

legs = []

# Find the joints for each leg
for i in range(1, 7):
    hip = sim.getObject(f"/HexapodBody/HipJoint_{i}")
    knee = sim.getObject(
        f"/HexapodBody/HipJoint_{i}/UpperLeg_{i}/KneeJoint_{i}"
    )
    legs.append((hip, knee))

print("Found all 12 joints.")
print("Tripod gait running; press Ctrl+C to stop.")

frequency = 0.125  # 8-second cycle
hip_amplitude = 0.20
knee_amplitude = 0.25

start = time.time()

try:
    while True:
        t = time.time() - start

        for i, (hip, knee) in enumerate(legs):
            # Legs 1, 3, 5 vs. legs 2, 4, 6
            phase = 0 if i % 2 == 0 else math.pi
            wave = math.sin(2 * math.pi * frequency * t + phase)

            sim.setJointPosition(
                hip, hip_amplitude * wave
            )
            sim.setJointPosition(
                knee, 0.10 + knee_amplitude * max(0, wave)
            )

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nTripod gait stopped.")
