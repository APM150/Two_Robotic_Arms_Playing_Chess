import numpy as np

import pydrake.all
from pydrake.all import *

from manipulation import FindResource
from manipulation.scenarios import AddIiwaDifferentialIK, AddRgbdSensors
from manipulation.meshcat_utils import MeshcatPoseSliders, WsgButton

IIWA_DEFAULT_Q = [0, 0.1, 0.0, -1.2, 0.0, 1.6, 0.0]

# Adds an iiwa robot to the plant in the given pose, with given name
def AddIiwa(plant, collision_model="no_collision", pose=RigidTransform(), name="iiwa"):
    sdf_path = pydrake.common.FindResourceOrThrow(
        "drake/manipulation/models/iiwa_description/iiwa7/"
        f"iiwa7_{collision_model}.sdf")

    parser = Parser(plant)
    iiwa = parser.AddModelFromFile(sdf_path, name)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0", iiwa), pose)

    # Set default positions
    q0 = IIWA_DEFAULT_Q
    index = 0
    for joint_index in plant.GetJointIndices(iiwa):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, pydrake.multibody.tree.RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1
    return iiwa

# Adds wsg welded to the given iiwa
def AddWsg(plant, iiwa, name="wsg", roll=np.pi/2.0, welded=False, internal=False):
    parser = Parser(plant)
    if welded:
        gripper = parser.AddModelFromFile(
            FindResource("models/schunk_wsg_50_welded_fingers.sdf"), name)
    else:
        gripper = parser.AddModelFromFile(
            FindResourceOrThrow(
                "drake/manipulation/models/"
                "wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf"), name)
    if not internal:
        X_7G = RigidTransform(RollPitchYaw(np.pi/2.0, 0, roll), [0, 0, 0.09])
        plant.WeldFrames(plant.GetFrameByName("iiwa_link_7", iiwa),
                         plant.GetFrameByName("body", gripper), X_7G)
    return gripper

# Adds a table welded at the given height
def AddChessBoard(plant, name='chessBoard'):
    parser = pydrake.multibody.parsing.Parser(plant)
    chessBoard = parser.AddModelFromFile("models/Chess_Board_OBJ/Wooden_Chess_Board.sdf", name)

    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("Wooden_Chess_Board_body_link"),
        RigidTransform(RollPitchYaw(np.pi/2.0, 0, np.pi/2.0), p=[0, 0, 0])
    )

    return chessBoard

# Adds bishop given color
def AddBishop(plant, color=None, name='bishop'):
    parser = Parser(plant)
    if color == "black":
        bishop = parser.AddModelFromFile("models/Bishop_Chess_OBJ/Bishop_Chess_black.sdf", name)
    else:
        bishop = parser.AddModelFromFile("models/Bishop_Chess_OBJ/Bishop_Chess.sdf", name)
    return bishop

# Adds king given color
def AddKing(plant, color=None, name='king'):
    parser = Parser(plant)
    if color == "black":
        king = parser.AddModelFromFile("models/King_Chess_OBJ/King_Chess_black.sdf", name)
    else:
        king = parser.AddModelFromFile("models/King_Chess_OBJ/King_Chess.sdf", name)
    return king

# Adds knight given color
def AddKnight(plant, color=None, name='knight'):
    parser = Parser(plant)
    if color == "black":
        knight = parser.AddModelFromFile("models/Knight_Chess_OBJ/Knight_Chess_black.sdf", name)
    else:
        knight = parser.AddModelFromFile("models/Knight_Chess_OBJ/Knight_Chess.sdf", name)
    return knight

# Adds pawn given color
def AddPawn(plant, color=None, name='pawn'):
    parser = Parser(plant)
    if color == "black":
        pawn = parser.AddModelFromFile("models/Pawn_Chess_OBJ/Pawn_Chess_black.sdf", name)
    else:
        pawn = parser.AddModelFromFile("models/Pawn_Chess_OBJ/Pawn_Chess.sdf", name)
    return pawn

# Adds queen given color
def AddQueen(plant, color=None, name='queen'):
    parser = Parser(plant)
    if color == "black":
        queen = parser.AddModelFromFile("models/Queen_Chess_OBJ/Queen_Chess_black.sdf", name)
    else:
        queen = parser.AddModelFromFile("models/Queen_Chess_OBJ/Queen_Chess.sdf", name)
    return queen

# Adds rook given color
def AddRook(plant, color=None, name='rook'):
    parser = Parser(plant)
    if color == "black":
        rook = parser.AddModelFromFile("models/Rook_Chess_OBJ/Rook_Chess_black.sdf", name)
    else:
        rook = parser.AddModelFromFile("models/Rook_Chess_OBJ/Rook_Chess.sdf", name)
    return rook

# Adds a bin for removal chess piece
def AddBin(plant, name="bin"):
    parser = pydrake.multibody.parsing.Parser(plant)
    bin = parser.AddModelFromFile("models/bin.sdf", name)

    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("bin_base"),
        RigidTransform(RollPitchYaw(0, 0, np.pi/2.0), p=[0, -0.75, 0])
    )
    return bin

# Adds cameras
def AddCamera(plant, pose=RigidTransform(), name="camera"):
    parser = pydrake.multibody.parsing.Parser(plant)
    camera = parser.AddModelFromFile("models/camera_box.sdf", name)

    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("base", camera), pose)
    return camera

# Sets up an iiwa robot with input and output ports
def ConfigureIiwa(builder, plant, iiwa, name='iiwa', time_step=0.002):
    num_iiwa_positions = plant.num_positions(iiwa)

    # Need a PassThrough system to export the input port
    iiwa_position = builder.AddSystem(PassThrough(num_iiwa_positions))
    builder.ExportInput(iiwa_position.get_input_port(), f"{name}_position")
    builder.ExportOutput(iiwa_position.get_output_port(), f"{name}_position_commanded")

    # Export the iiwa "state" outputs
    demux = builder.AddSystem(Demultiplexer(
        2 * num_iiwa_positions, num_iiwa_positions))
    builder.Connect(plant.get_state_output_port(iiwa), demux.get_input_port())
    builder.ExportOutput(demux.get_output_port(0), f"{name}_position_measured")
    builder.ExportOutput(demux.get_output_port(1), f"{name}_velocity_estimated")
    builder.ExportOutput(plant.get_state_output_port(iiwa), f"{name}_state_estimated")

    # Make the plant for the iiwa controller to use
    controller_plant = MultibodyPlant(time_step=time_step)
    controller_iiwa = AddIiwa(controller_plant)
    AddWsg(controller_plant, controller_iiwa, welded=True)
    controller_plant.Finalize()

    # Add the iiwa controller
    iiwa_controller = builder.AddSystem(
        InverseDynamicsController(
            controller_plant,
            kp=[100] * num_iiwa_positions,
            ki=[1] * num_iiwa_positions,
            kd=[20] * num_iiwa_positions,
            has_reference_acceleration=False))
    iiwa_controller.set_name(f"{name}_controller")
    builder.Connect(plant.get_state_output_port(iiwa),
                    iiwa_controller.get_input_port_estimated_state())

    # Add in the feed-forward torque
    adder = builder.AddSystem(Adder(2, num_iiwa_positions))
    builder.Connect(iiwa_controller.get_output_port_control(),
                    adder.get_input_port(0))
    # Use a PassThrough to make the port optional
    torque_passthrough = builder.AddSystem(PassThrough([0]*num_iiwa_positions))
    builder.Connect(torque_passthrough.get_output_port(),
                    adder.get_input_port(1))
    builder.ExportInput(torque_passthrough.get_input_port(),
                        f"{name}_feedforward_torque")
    builder.Connect(adder.get_output_port(),
                    plant.get_actuation_input_port(iiwa))

    # Add discrete derivative to command velocities
    desired_state_from_position = builder.AddSystem(
        StateInterpolatorWithDiscreteDerivative(
            num_iiwa_positions, time_step, suppress_initial_transient=True))
    desired_state_from_position.set_name(f"{name}_desired_state_from_position")
    builder.Connect(desired_state_from_position.get_output_port(),
                    iiwa_controller.get_input_port_desired_state())
    builder.Connect(iiwa_position.get_output_port(),
                    desired_state_from_position.get_input_port())

    # Export commanded torques.
    builder.ExportOutput(adder.get_output_port(),
                         f"{name}_torque_commanded")
    builder.ExportOutput(adder.get_output_port(),
                         f"{name}_torque_measured")
    builder.ExportOutput(plant.get_generalized_contact_forces_output_port(iiwa),
                         f"{name}_torque_external")

# Sets up a wsg with input and output ports
def ConfigureWsg(builder, plant, wsg, name='wsg', time_step=0.002):
    # Add wsg controller
    wsg_controller = builder.AddSystem(SchunkWsgPositionController(time_step=time_step))
    wsg_controller.set_name(f"{name}_controller")
    builder.Connect(wsg_controller.get_generalized_force_output_port(),
                    plant.get_actuation_input_port(wsg))
    builder.Connect(plant.get_state_output_port(wsg),
                    wsg_controller.get_state_input_port())
    builder.ExportInput(wsg_controller.get_desired_position_input_port(),
                        f"{name}_position")
    builder.ExportInput(wsg_controller.get_force_limit_input_port(),
                        f"{name}_force_limit")
    wsg_mbp_state_to_wsg_state = builder.AddSystem(
        MakeMultibodyStateToWsgStateSystem())
    builder.Connect(plant.get_state_output_port(wsg),
                    wsg_mbp_state_to_wsg_state.get_input_port())
    builder.ExportOutput(wsg_mbp_state_to_wsg_state.get_output_port(),
                         f"{name}_state_measured")
    builder.ExportOutput(wsg_controller.get_grip_force_output_port(),
                         f"{name}_force_measured")

# Starts up the chess scenario
class Scenario:

    def __init__(self, time_step=0.002, internal_model=False):
        self._builder = DiagramBuilder()

        self._plant, self._scene_graph = AddMultibodyPlantSceneGraph(
            self._builder, time_step=time_step)

        if not internal_model:
            # Add two iiwas to the scene
            self.iiwa1 = AddIiwa(self._plant, pose=RigidTransform(p=[-0.6,0,0]), name="iiwa1")
            self.iiwa2 = AddIiwa(self._plant, pose=RigidTransform(RollPitchYaw(0, 0, np.pi), [0.6,0,0]), name="iiwa2")

            # Give the iiwas wsg
            self.wsg1 = AddWsg(self._plant, self.iiwa1, name="wsg1")
            self.wsg2 = AddWsg(self._plant, self.iiwa2, name="wsg2")
        else:
            self.wsg = AddWsg(self._plant, None, name="wsg", internal=True)

        # Add bishops
        self.bishops_white = []
        self.bishops_black = []
        for i in range(2):
            self.bishops_white.append(AddBishop(self._plant, name=f"bishop_white{i}"))
            self.bishops_black.append(AddBishop(self._plant, color="black", name=f"bishop_black{i}"))

        # Add knight
        self.knights_white = []
        self.knights_black = []
        for i in range(2):
            self.knights_white.append(AddKnight(self._plant, name=f"knight_white{i}"))
            self.knights_black.append(AddKnight(self._plant, color="black", name=f"knight_black{i}"))

        # Add rook
        self.rooks_white = []
        self.rooks_black = []
        for i in range(2):
            self.rooks_white.append(AddRook(self._plant, name=f"rook_white{i}"))
            self.rooks_black.append(AddRook(self._plant, color="black", name=f"rook_black{i}"))

        # Add king
        self.king_white = AddKing(self._plant, name="king_white")
        self.king_black = AddKing(self._plant, color="black", name="king_black")

        # Add queen
        self.queen_white = AddQueen(self._plant, name="queen_white")
        self.queen_black = AddQueen(self._plant, color="black", name="queen_black")

        # Add pawns
        self.pawns_white = []
        self.pawns_black = []
        for i in range(8):
            self.pawns_white.append(AddPawn(self._plant, name=f"pawn_white{i}"))
            self.pawns_black.append(AddPawn(self._plant, color="black", name=f"pawn_black{i}"))

        # Add chessboard
        self.chessBoard = AddChessBoard(self._plant)

        # Add bin
        self.bin = AddBin(self._plant)

        # Add Cameras
        self.camera0 = AddCamera(self._plant, pose=RigidTransform(RollPitchYaw(0,np.pi,0), p=[0,0,1]), name="camera0")
        self.camera1 = AddCamera(self._plant, pose=RigidTransform(RollPitchYaw(np.pi/4,np.pi,np.pi), p=[0,-0.4,1]), name="camera1")
        self.camera2 = AddCamera(self._plant, pose=RigidTransform(RollPitchYaw(np.pi/4,np.pi,0), p=[0,0.4,1]), name="camera2")
        self.camera3 = AddCamera(self._plant, pose=RigidTransform(RollPitchYaw(np.pi/2,np.pi,3*np.pi/4),p=[-0.5,-0.5,0.1]), name="camera3")
        self.camera4 = AddCamera(self._plant, pose=RigidTransform(RollPitchYaw(np.pi/2,np.pi,np.pi/4),p=[-0.5,0.5,0.1]), name="camera4")
        self.camera5 = AddCamera(self._plant, pose=RigidTransform(RollPitchYaw(np.pi/2,np.pi,-np.pi/4),p=[0.5,0.5,0.1]), name="camera5")
        self.camera6 = AddCamera(self._plant, pose=RigidTransform(RollPitchYaw(np.pi/2,np.pi,-3*np.pi/4),p=[0.5,-0.5,0.1]), name="camera6")
        self.camera7 = AddCamera(self._plant, pose=RigidTransform(RollPitchYaw(np.pi/2,np.pi,0),p=[0.0,0.5,0.1]), name="camera7")

        self._plant.Finalize()

        if not internal_model:
            # Configure the ports for the iiwas
            ConfigureIiwa(self._builder, self._plant, self.iiwa1, name="iiwa1", time_step=time_step)
            ConfigureIiwa(self._builder, self._plant, self.iiwa2, name="iiwa2", time_step=time_step)

            # Configure the ports for the wsgs
            ConfigureWsg(self._builder, self._plant, self.wsg1, name="wsg1", time_step=time_step)
            ConfigureWsg(self._builder, self._plant, self.wsg2, name="wsg2", time_step=time_step)

        # Configure the ports for the cameras
        AddRgbdSensors(self._builder,
                       self._plant,
                       self._scene_graph,
                       model_instance_prefix="camera")

        # Export "cheat" ports
        self._builder.ExportOutput(self._scene_graph.get_query_output_port(), "query_object")
        self._builder.ExportOutput(self._plant.get_contact_results_output_port(), "contact_results")
        self._builder.ExportOutput(self._plant.get_state_output_port(), "plant_continuous_state")
        self._builder.ExportOutput(self._plant.get_body_poses_output_port(), "body_poses")

    # Builds diagram
    def get_diagram(self):
        diagram = self._builder.Build()
        return diagram

    # returns iiwa associated with name
    def get_iiwa(self, name):
        if name == "iiwa1":
            return self.iiwa1, self.wsg1
        elif name == "iiwa2":
            return self.iiwa2, self.wsg2


def InitFullGame(scenario):
    plant = scenario._plant
    # bishop white
    for i, piece in enumerate(scenario.bishops_white):
        pos = [0.35, -0.15+i*0.3, 0.04]
        plant.SetDefaultPositions(piece, np.hstack((0,0,0.7071068,0.7071068,pos)))
    # bishop black
    for i, piece in enumerate(scenario.bishops_black):
        pos = [-0.35, -0.15+i*0.3, 0.04]
        plant.SetDefaultPositions(piece, np.hstack((0,0,0.7071068,0.7071068,pos)))
    # knight white
    for i, piece in enumerate(scenario.knights_white):
        pos = [0.35, -0.25+i*0.5, 0.04]
        plant.SetDefaultPositions(piece, np.hstack((0,0,0.7071068,0.7071068,pos)))
    # knight black
    for i, piece in enumerate(scenario.knights_black):
        pos = [-0.35, -0.25+i*0.5, 0.04]
        plant.SetDefaultPositions(piece, np.hstack((0,0,0.7071068,0.7071068,pos)))
    # rook white
    for i, piece in enumerate(scenario.rooks_white):
        pos = [0.35, -0.35+i*0.7, 0.04]
        plant.SetDefaultPositions(piece, np.hstack((0,0,0.7071068,0.7071068,pos)))
    # rook black
    for i, piece in enumerate(scenario.rooks_black):
        pos = [-0.35, -0.35+i*0.7, 0.04]
        plant.SetDefaultPositions(piece, np.hstack((0,0,0.7071068,0.7071068,pos)))
    # pawn white
    for i, piece in enumerate(scenario.pawns_white):
        pos = [0.25, -0.35+i*0.1, 0.04]
        plant.SetDefaultPositions(piece, np.hstack((0,0,0.7071068,0.7071068,pos)))
    # pawn black
    for i, piece in enumerate(scenario.pawns_black):
        pos = [-0.25, -0.35+i*0.1, 0.04]
        plant.SetDefaultPositions(piece, np.hstack((0,0,0.7071068,0.7071068,pos)))
    # king white
    pos = [0.35, 0.05, 0.04]
    plant.SetDefaultPositions(scenario.king_white, np.hstack((0,0,0.7071068,0.7071068,pos)))
    # king black
    pos = [-0.35, 0.05, 0.04]
    plant.SetDefaultPositions(scenario.king_black, np.hstack((0,0,0.7071068,0.7071068,pos)))
    # queen white
    pos = [0.35, -0.05, 0.04]
    plant.SetDefaultPositions(scenario.queen_white, np.hstack((0,0,0.7071068,0.7071068,pos)))
    # queen black
    pos = [-0.35, -0.05, 0.04]
    plant.SetDefaultPositions(scenario.queen_black, np.hstack((0,0,0.7071068,0.7071068,pos)))


def InitEndGame1(scenario):
    plant = scenario._plant
    # bishop white
    for i, piece in enumerate(scenario.bishops_white):
        pos = [0.35, -0.15+i*0.3, 0.04]
        plant.SetDefaultPositions(piece, np.hstack((0,0,0.7071068,0.7071068,pos)))
    # bishop black
    for i, piece in enumerate(scenario.bishops_black):
        pos = [-0.35, -0.15+i*0.3, 0.04]
        plant.SetDefaultPositions(piece, np.hstack((0,0,0.7071068,0.7071068,pos)))
    # knight white
    for i, piece in enumerate(scenario.knights_white):
        pos = [0.35, -0.25+i*0.5, 0.04]
        plant.SetDefaultPositions(piece, np.hstack((0,0,0.7071068,0.7071068,pos)))
    # knight black
    for i, piece in enumerate(scenario.knights_black):
        pos = [-0.35, -0.25+i*0.5, 0.04]
        plant.SetDefaultPositions(piece, np.hstack((0,0,0.7071068,0.7071068,pos)))
    # rook white
    for i, piece in enumerate(scenario.rooks_white):
        pos = [-0.1+i*0.3, -0.75, 0.3]
        plant.SetDefaultPositions(piece, np.hstack((0,0,0.7071068,0.7071068,pos)))
    # rook black
    for i, piece in enumerate(scenario.rooks_black):
        pos = [-0.35, -0.35+i*0.7, 0.04]
        plant.SetDefaultPositions(piece, np.hstack((0,0,0.7071068,0.7071068,pos)))
    # pawn white
    for i, piece in enumerate(scenario.pawns_white):
        pos = [-0.25+i*0.05, -0.7, 0.3]
        plant.SetDefaultPositions(piece, np.hstack((0,0,0.7071068,0.7071068,pos)))
    # pawn black
    for i, piece in enumerate(scenario.pawns_black):
        pos = [-0.25, -0.35+i*0.1, 0.04]
        plant.SetDefaultPositions(piece, np.hstack((0,0,0.7071068,0.7071068,pos)))
    # king white
    pos = [0.35, 0.05, 0.04]
    plant.SetDefaultPositions(scenario.king_white, np.hstack((0,0,0.7071068,0.7071068,pos)))
    # king black
    pos = [-0.35, 0.05, 0.04]
    plant.SetDefaultPositions(scenario.king_black, np.hstack((0,0,0.7071068,0.7071068,pos)))
    # queen white
    pos = [0.35, -0.05, 0.04]
    plant.SetDefaultPositions(scenario.queen_white, np.hstack((0,0,0.7071068,0.7071068,pos)))
    # queen black
    pos = [-0.35, -0.05, 0.04]
    plant.SetDefaultPositions(scenario.queen_black, np.hstack((0,0,0.7071068,0.7071068,pos)))


def InitEndGame2(scenario):
    """ Anatoly Karpov v.s. Garry Kasparov (Mar-21-2017) last 2 steps """
    plant = scenario._plant
    # bishop white
    plant.SetDefaultPositions(scenario.bishops_white[0],
                              np.hstack((0,0,0.7071068,0.7071068,-0.15,-0.15,0.04)))
    plant.SetDefaultPositions(scenario.bishops_white[1],
                              np.hstack((0,0,0.7071068,0.7071068,-0.1,-0.75,0.3)))
    # bishop black
    plant.SetDefaultPositions(scenario.bishops_black[0],
                              np.hstack((0,0,0.7071068,0.7071068,0.25,-0.35,0.04)))
    plant.SetDefaultPositions(scenario.bishops_black[1],
                              np.hstack((0,0,0.7071068,0.7071068,-0.15,-0.75,0.3)))
    # knight white
    for i, piece in enumerate(scenario.knights_white):
        pos = [-0.1+i*0.05, -0.7, 0.3]
        plant.SetDefaultPositions(piece, np.hstack((0,0,0.7071068,0.7071068,pos)))
    # knight black
    for i, piece in enumerate(scenario.knights_black):
        pos = [-0.2+i*0.05, -0.7, 0.3]
        plant.SetDefaultPositions(piece, np.hstack((0,0,0.7071068,0.7071068,pos)))
    # rook white
    plant.SetDefaultPositions(scenario.rooks_white[0],
                              np.hstack((0,0,0.7071068,0.7071068,-0.05,0.05,0.04)))
    plant.SetDefaultPositions(scenario.rooks_white[1],
                              np.hstack((0,0,0.7071068,0.7071068,-0.1,-0.8,0.3)))
    # rook black
    plant.SetDefaultPositions(scenario.rooks_black[0],
                              np.hstack((0,0,0.7071068,0.7071068,-0.35,0.35,0.04)))
    plant.SetDefaultPositions(scenario.rooks_black[1],
                              np.hstack((0,0,0.7071068,0.7071068,-0.15,-0.8,0.3)))
    # pawn white
    plant.SetDefaultPositions(scenario.pawns_white[-1],
                              np.hstack((0,0,0.7071068,0.7071068,-0.25,-0.15,0.04)))
    plant.SetDefaultPositions(scenario.pawns_white[-2],
                              np.hstack((0,0,0.7071068,0.7071068,0.15,-0.35,0.04)))
    plant.SetDefaultPositions(scenario.pawns_white[-3],
                              np.hstack((0,0,0.7071068,0.7071068,0.05,0.25,0.04)))
    for i in range(5):
        pos = [-0.1+i*0.05, -0.85, 0.3]
        plant.SetDefaultPositions(scenario.pawns_white[i], np.hstack((0,0,0.7071068,0.7071068,pos)))
    # pawn black
    plant.SetDefaultPositions(scenario.pawns_black[-1],
                              np.hstack((0,0,0.7071068,0.7071068,0.25,0.35,0.04)))
    for i in range(7):
        pos = [-0.1+i*0.05, -0.9, 0.3]
        plant.SetDefaultPositions(scenario.pawns_black[i], np.hstack((0,0,0.7071068,0.7071068,pos)))
    # king white
    plant.SetDefaultPositions(scenario.king_white,
                              np.hstack((0,0,0.7071068,0.7071068,-0.15,-0.25,0.04)))
    # king black
    plant.SetDefaultPositions(scenario.king_black,
                              np.hstack((0,0,0.7071068,0.7071068,0.15,0.25,0.04)))
    # queen white
    plant.SetDefaultPositions(scenario.queen_white,
                              np.hstack((0,0,0.7071068,0.7071068,-0.2,-0.75,0.3)))
    # queen black
    plant.SetDefaultPositions(scenario.queen_black,
                              np.hstack((0,0,0.7071068,0.7071068,-0.25,-0.75,0.3)))


if __name__ == "__main__":
    ###################################################
    ## Teleop, comment one iiwa in Scenario to work. ##
    ###################################################
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    scenario = Scenario(time_step=0.002)
    station = builder.AddSystem(scenario.get_diagram())
    InitFullGame(scenario)

    # Visualize
    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), meshcat)
    meshcat.ResetRenderMode()
    meshcat.DeleteAddedControls()

    # Try the teleop
    # Set up differential inverse kinematics.
    robot1 = station.GetSubsystemByName("iiwa1_controller").get_multibody_plant_for_control()
    differential_ik1 = AddIiwaDifferentialIK(builder, robot1,
                                             robot1.GetFrameByName("iiwa_link_7"))
    builder.Connect(differential_ik1.get_output_port(),
                    station.GetInputPort("iiwa1_position"))
    builder.Connect(station.GetOutputPort("iiwa1_state_estimated"),
                    differential_ik1.GetInputPort("robot_state"))
    # Set up teleop widgets.
    teleop1 = builder.AddSystem(
        MeshcatPoseSliders(
            meshcat,
            min_range=MeshcatPoseSliders.MinRange(roll=0,
                                                    pitch=-0.5,
                                                    yaw=-np.pi,
                                                    x=-0.6,
                                                    y=-0.8,
                                                    z=0.0),
            max_range=MeshcatPoseSliders.MaxRange(roll=2 * np.pi,
                                                    pitch=np.pi,
                                                    yaw=np.pi,
                                                    x=0.8,
                                                    y=0.3,
                                                    z=1.1),
            body_index=scenario._plant.GetBodyByName("iiwa_link_7", scenario.iiwa1).index()))
    builder.Connect(teleop1.get_output_port(0),
                    differential_ik1.get_input_port(0))
    builder.Connect(station.GetOutputPort("body_poses"),
                    teleop1.GetInputPort("body_poses"))
    wsg_teleop1 = builder.AddSystem(WsgButton(meshcat))
    builder.Connect(wsg_teleop1.get_output_port(0),
                    station.GetInputPort("wsg1_position"))
    # Simulate
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1.0)
    # simulator.AdvanceTo(0.1)
    meshcat.AddButton("Stop Simulation", "Escape")
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
    meshcat.DeleteButton("Stop Simulation")
