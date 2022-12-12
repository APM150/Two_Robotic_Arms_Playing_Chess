import numpy as np

import pydrake.all
from pydrake.all import *

from scenarios import *
from controllers import *
from trajectory_planning import *


# Starts the simulation of Chess Playing
def SimulateWithMeshcat():
    total_timesteps = 3
    builder = DiagramBuilder()
    scenario = Scenario(time_step=0.002)
    station = builder.AddSystem(scenario.get_diagram())

    # Set up the initial chess status
    InitEndGame2(scenario)

    # Set up trajectory given pick and place grid
    temp_context = station.CreateDefaultContext()
    X1_G, times1, X2_G, times2 = make_endgame2_frames(station, scenario, temp_context, meshcat=meshcat)

    # Set up controller given trajectory
    integrator1 = SetupController(builder, scenario, station, "iiwa1", X1_G, times1, total_timesteps=total_timesteps)
    integrator2 = SetupController(builder, scenario, station, "iiwa2", X2_G, times2, total_timesteps=total_timesteps)

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), meshcat)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    diagram.Publish(context)

    # Set up simulator
    simulator = Simulator(diagram, context)
    simulator_context = simulator.get_mutable_context()

    SetupIntegrator(integrator1, simulator_context, scenario, "iiwa1")
    SetupIntegrator(integrator2, simulator_context, scenario, "iiwa2")

    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(0.01)

    return simulator, visualizer


if __name__ == '__main__':
    # Start the visualizer.
    meshcat = StartMeshcat()
    simulator, visualizer = SimulateWithMeshcat()
    # Run simulation of system
    visualizer.StartRecording()
    simulator.AdvanceTo(82)
    visualizer.StopRecording()
    visualizer.PublishRecording()
