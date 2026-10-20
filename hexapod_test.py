
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Connect to the running CoppeliaSim application
client = RemoteAPIClient()
sim = client.require("sim")

print("Connected to CoppeliaSim!")
print("Simulation time:", sim.getSimulationTime())
print("Connection test successful.")
