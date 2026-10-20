
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math

client = RemoteAPIClient()
sim = client.require("sim")

# Start the simulation if it is stopped.
if sim.getSimulationState() == sim.simulation_stopped:
    sim.startSimulation()
    time.sleep(0.5)

# Find the first hip and knee by their scene aliases.
hip = sim.getObject("/HexapodBody/HipJoint_1")
knee = sim.getObject("/HexapodBody/HipJoint_1/UpperLeg_1/KneeJoint_1")

print("Hip handle:", hip)
print("Knee handle:", knee)
print("Moving leg 1 slowly. Stop with Ctrl+C.")

try:
    start = time.time()

    while True:
        t = time.time() - start

        # Small, slow test movements in radians.
        sim.setJointPosition(hip, 0.20 * math.sin(t))
        sim.setJointPosition(knee, 0.25 * math.sin(t))

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Joint test stopped.")
