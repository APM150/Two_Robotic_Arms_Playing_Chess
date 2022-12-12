import numpy as np

import pydrake.all
from pydrake.all import *

from trajectory_planning import *


class PseudoInverseController(LeafSystem):
    def __init__(self, plant, iiwa, wsg, name):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = iiwa
        self._wsg = wsg
        self.player_num = name[-1]
        self._G = plant.GetBodyByName("body", wsg).body_frame()
        self._W = plant.world_frame()

        self.V_G_port = self.DeclareVectorInputPort(f"V{self.player_num}_WG", 6)
        self.q_port = self.DeclareVectorInputPort(f"iiwa{self.player_num}_position", 7)
        self.DeclareVectorOutputPort(f"iiwa{self.player_num}_velocity", 7, self.CalcOutput)
        self.iiwa_start = plant.GetJointByName("iiwa_joint_1", iiwa).velocity_start()
        self.iiwa_end = plant.GetJointByName("iiwa_joint_7", iiwa).velocity_start()

    def CalcOutput(self, context, output):
        V_G = self.V_G_port.Eval(context)
        q = self.q_port.Eval(context)
        self._plant.SetPositions(self._plant_context, self._iiwa, q)
        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context, JacobianWrtVariable.kV,
            self._G, [0,0,0], self._W, self._W)
        J_G = J_G[:,self.iiwa_start:self.iiwa_end+1] # Only iiwa terms.
        v = np.linalg.pinv(J_G).dot(V_G)
        output.SetFromVector(v)


# Sets up pseudo inverse controllers given trajectories and wires them up
def SetupController(builder, scenario, station, name, X_G, times, total_timesteps=1):
    plant = scenario._plant
    iiwa, wsg = scenario.get_iiwa(name)
    player_num = name[-1]

    # Make the trajectories
    traj = make_gripper_trajectory(X_G, times, total_timesteps=total_timesteps)
    traj_V_G = traj.MakeDerivative()

    V_G_source = builder.AddSystem(TrajectorySource(traj_V_G))
    V_G_source.set_name(f"v{player_num}_WG")
    # Set up controller
    controller = builder.AddSystem(PseudoInverseController(plant, iiwa, wsg, name))
    controller.set_name(f"PseudoInverseController{player_num}")
    builder.Connect(V_G_source.get_output_port(), controller.GetInputPort(f"V{player_num}_WG"))
    integrator = builder.AddSystem(Integrator(7))
    integrator.set_name(f"integrator{player_num}")
    builder.Connect(controller.get_output_port(),
                    integrator.get_input_port())
    builder.Connect(integrator.get_output_port(),
                    station.GetInputPort(f"iiwa{player_num}_position"))
    builder.Connect(station.GetOutputPort(f"iiwa{player_num}_position_measured"),
                    controller.GetInputPort(f"iiwa{player_num}_position"))

    traj_wsg_command = make_wsg_command_trajectory(times, total_timesteps=total_timesteps)
    wsg_source = builder.AddSystem(TrajectorySource(traj_wsg_command))
    wsg_source.set_name(f"wsg{player_num}_command")
    builder.Connect(wsg_source.get_output_port(), station.GetInputPort(f"wsg{player_num}_position"))

    return integrator


# Sets up integrator with initial pose
def SetupIntegrator(integrator, context, scenario, name):
    plant = scenario._plant
    iiwa, wsg = scenario.get_iiwa(name)
    integrator.set_integral_value(
        integrator.GetMyContextFromRoot(context),
        plant.GetPositions(plant.GetMyContextFromRoot(context), iiwa))
