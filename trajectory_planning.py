import numpy as np

import pydrake.all
from pydrake.all import *
from grasp_selection import *

opened = np.array([0.107])
closed = np.array([0.0])

def make_gripper_move_frames(X_G, X_O, timestep=0, far_pick=False,
                             far_place=False, last_return=0):
    """
    Takes a partial specification with X_G["initial"] and X_O["initial"] and X_0["goal"], and
    returns a X_G and times with all of the pick and place frames populated.
    """
    # Define (again) the gripper pose relative to the object when in grasp.
    p_GgraspO = [0, 0.11, 0]
    R_GgraspO = RotationMatrix.MakeXRotation(np.pi/2.0).multiply(
        RotationMatrix.MakeZRotation(np.pi/2.0))
    X_GgraspO = RigidTransform(R_GgraspO, p_GgraspO)
    X_OGgrasp = X_GgraspO.inverse()
    # Define the gripper pose for far pick object
    p_GgraspOfarpick = [0, 0.11, 0]
    R_GgraspOfarpick = RotationMatrix.MakeXRotation(3*np.pi/4.0).multiply(
        RotationMatrix.MakeZRotation(0))
    X_GgraspOfarpick = RigidTransform(R_GgraspOfarpick, p_GgraspOfarpick)
    X_OGgraspfarpick = X_GgraspOfarpick.inverse()
    # Define the gripper pose for far place object
    p_GgraspOfarplace = [0, 0.11, 0]
    R_GgraspOfarplace = RotationMatrix.MakeXRotation(-np.pi).multiply(
        RotationMatrix.MakeZRotation(np.pi/2.0))
    X_GgraspOfarplace = RigidTransform(R_GgraspOfarplace, p_GgraspOfarplace)
    X_OGgraspfarplace = X_GgraspOfarplace.inverse()

    # pregrasp is negative y in the gripper frame (see the figure!).
    X_GgraspGpregrasp = RigidTransform([0, -0.08, 0])
    X_GgraspGpregraspfarpick = RigidTransform([0, -0.08, 0])
    X_GgraspGpregraspfarplace = RigidTransform([0, 0, -0.12])

    if not far_pick:
        X_G[f"pick_{timestep}"] = X_O["initial"].multiply(X_OGgrasp)
        X_G[f"prepick_{timestep}"] = X_G[f"pick_{timestep}"].multiply(X_GgraspGpregrasp)
    else:
        X_G[f"pick_{timestep}"] = X_O["initial"].multiply(X_OGgraspfarpick)
        X_G[f"prepick_{timestep}"] = X_G[f"pick_{timestep}"].multiply(X_GgraspGpregraspfarpick)
    if not far_place:
        X_G[f"place_{timestep}"] = X_O["goal"].multiply(X_OGgrasp)
        X_G[f"preplace_{timestep}"] = X_G[f"place_{timestep}"].multiply(X_GgraspGpregrasp)
    else:
        X_G[f"place_{timestep}"] = X_O["goal"].multiply(X_OGgraspfarplace)
        X_G[f"preplace_{timestep}"] = X_G[f"place_{timestep}"].multiply(X_GgraspGpregraspfarplace)

    # I'll interpolate a halfway orientation by converting to axis angle and halving the angle.
    X_GprepickGpreplace = X_G[f"prepick_{timestep}"].inverse().multiply(X_G[f"preplace_{timestep}"])
    angle_axis = X_GprepickGpreplace.rotation().ToAngleAxis()
    X_GprepickGclearance = RigidTransform(AngleAxis(angle=angle_axis.angle()/2.0, axis=angle_axis.axis()),
                                          X_GprepickGpreplace.translation()/2.0 + np.array([0, -0.3, 0]))
    X_G[f"clearance_{timestep}"] = X_G[f"prepick_{timestep}"].multiply(X_GprepickGclearance)

    # Now let's set the timing
    times = {f"initial_{timestep}": last_return+0.01}
    X_GinitialGprepick = X_G[f"initial_{timestep}"].inverse().multiply(X_G[f"prepick_{timestep}"])
    times[f"prepick_{timestep}"] = times[f"initial_{timestep}"] + 10.0*np.linalg.norm(X_GinitialGprepick.translation())
    # Allow some time for the gripper to close.
    times[f"pick_start_{timestep}"] = times[f"prepick_{timestep}"] + 2.0
    times[f"pick_end_{timestep}"] = times[f"pick_start_{timestep}"] + 2.0
    X_G[f"pick_start_{timestep}"] = X_G[f"pick_{timestep}"]
    X_G[f"pick_end_{timestep}"] = X_G[f"pick_{timestep}"]
    times[f"postpick_{timestep}"] = times[f"pick_end_{timestep}"] + 2.0
    X_G[f"postpick_{timestep}"] = X_G[f"prepick_{timestep}"]
    time_to_from_clearance = 10.0*np.linalg.norm(X_GprepickGclearance.translation())
    times[f"clearance_{timestep}"] = times[f"postpick_{timestep}"] + time_to_from_clearance
    times[f"preplace_{timestep}"] = times[f"clearance_{timestep}"] + time_to_from_clearance
    times[f"place_start_{timestep}"] = times[f"preplace_{timestep}"] + 2.0
    times[f"place_end_{timestep}"] = times[f"place_start_{timestep}"] + 2.0
    X_G[f"place_start_{timestep}"] = X_G[f"place_{timestep}"]
    X_G[f"place_end_{timestep}"] = X_G[f"place_{timestep}"]
    times[f"postplace_{timestep}"] = times[f"place_end_{timestep}"] + 2.0
    X_G[f"postplace_{timestep}"] = X_G[f"preplace_{timestep}"]
    times[f"return_{timestep}"] = times[f"postplace_{timestep}"] + 2.0 #0.7623
    X_G[f"return_{timestep}"] = X_G[f"initial_{timestep}"]

    return X_G, times


def make_gripper_stop_frames(X_G, timestep=0, last_return=0, time_spent=29.5):
    """
    Takes a partial specification with X_G["initial"] and X_O["initial"] and X_0["goal"], and
    returns a X_G and times with all of the pick and place frames populated.
    """
    times = {f"initial_{timestep}": last_return+0.01}
    X_G[f"return_{timestep}"] = X_G[f"initial_{timestep}"]
    times[f"return_{timestep}"] = times[f"initial_{timestep}"] + time_spent
    return X_G, times


def make_gripper_trajectory(X_G, times, total_timesteps=1):
    """
    Constructs a gripper position trajectory from the plan "sketch".
    """

    sample_times = []
    poses = []
    for timestep in range(total_timesteps):
        for name in [f"initial_{timestep}", f"prepick_{timestep}", f"pick_start_{timestep}",
                     f"pick_end_{timestep}", f"postpick_{timestep}", f"clearance_{timestep}",
                     f"preplace_{timestep}", f"place_start_{timestep}", f"place_end_{timestep}",
                     f"postplace_{timestep}", f"return_{timestep}"]:
            sample_time = times.get(name)
            pose = X_G.get(name)
            if sample_time is not None:
                sample_times.append(sample_time)
            if pose is not None:
                poses.append(pose)

    return PiecewisePose.MakeLinear(sample_times, poses)


def make_wsg_command_trajectory(times, total_timesteps=1):
    if times.get("pick_start_0") != None:
        traj_wsg_command = PiecewisePolynomial.FirstOrderHold(
            [times["initial_0"], times["pick_start_0"]], np.hstack([[opened], [opened]]))
        traj_wsg_command.AppendFirstOrderSegment(times["pick_end_0"], closed)
        traj_wsg_command.AppendFirstOrderSegment(times["place_start_0"], closed)
        traj_wsg_command.AppendFirstOrderSegment(times["place_end_0"], opened)
        traj_wsg_command.AppendFirstOrderSegment(times["postplace_0"], opened)
    else:
        traj_wsg_command = PiecewisePolynomial.FirstOrderHold(
            [times["initial_0"], times["return_0"]], np.hstack([[opened], [opened]]))
    for timestep in range(1, total_timesteps):
        if times.get(f"pick_start_{timestep}") != None:
            traj_wsg_command.AppendFirstOrderSegment(times[f"initial_{timestep}"], opened)
            traj_wsg_command.AppendFirstOrderSegment(times[f"pick_start_{timestep}"], opened)

            traj_wsg_command.AppendFirstOrderSegment(times[f"pick_end_{timestep}"], closed)
            traj_wsg_command.AppendFirstOrderSegment(times[f"place_start_{timestep}"], closed)
            traj_wsg_command.AppendFirstOrderSegment(times[f"place_end_{timestep}"], opened)
            traj_wsg_command.AppendFirstOrderSegment(times[f"postplace_{timestep}"], opened)
        else:
            traj_wsg_command.AppendFirstOrderSegment(times[f"initial_{timestep}"], opened)
            traj_wsg_command.AppendFirstOrderSegment(times[f"return_{timestep}"], opened)

    return traj_wsg_command


def make_endgame2_frames(station, scenario, temp_context, meshcat=None):
    plant = scenario._plant
    temp_plant_context = plant.GetMyContextFromRoot(temp_context)
    # step 0: p1 move g4 to bin
    X1_G_init = {"initial_0": plant.EvalBodyPoseInWorld(temp_plant_context,
                                plant.GetBodyByName("body", scenario.wsg1))}
    X2_G_init = {"initial_0": plant.EvalBodyPoseInWorld(temp_plant_context,
                                plant.GetBodyByName("body", scenario.wsg2))}
    X1_O = {"initial": RigidTransform(RotationMatrix.MakeZRotation(np.pi/2.0), [0.05, 0.25, 0.07]),
            "goal": RigidTransform(RotationMatrix.MakeZRotation(np.pi), [-0.2, -0.65, 0.4])}
    # X_Gapg = SelectGrasp(station, temp_context, camera_body_indices=[
    #     plant.GetBodyIndices(scenario.camera0)[0],
    #     plant.GetBodyIndices(scenario.camera1)[0],
    #     plant.GetBodyIndices(scenario.camera2)[0],
    #     plant.GetBodyIndices(scenario.camera3)[0],
    #     plant.GetBodyIndices(scenario.camera4)[0],
    #     plant.GetBodyIndices(scenario.camera5)[0],
    #     plant.GetBodyIndices(scenario.camera6)[0],
    #     plant.GetBodyIndices(scenario.camera7)[0]
    # ], crop_lower=[0, 0.2, 0.01], crop_upper=[0.1, 0.3, 1], meshcat=meshcat)
    # print("X_Gapg")
    # print(X_Gapg)
    # X1_O["initial"] = RigidTransform(RotationMatrix.MakeZRotation(np.pi/2.0),
    #                                  [0.05, 0.25, X_Gapg.translation()[-1]-0.11])
    X1_G, times1 = make_gripper_move_frames(X1_G_init, X1_O, timestep=0, far_place=True)
    time_spent = times1["return_0"] - times1["initial_0"]
    X2_G, times2 = make_gripper_stop_frames(X2_G_init, timestep=0, time_spent=time_spent)
    last_return = times1["return_0"]

    # step 1: p1 move g3 to g4
    X1_G_init = {"initial_1": X1_G_init["initial_0"]}
    X2_G_init = {"initial_1": X2_G_init["initial_0"]}
    X1_O = {"initial": RigidTransform(RotationMatrix.MakeZRotation(np.pi/2.0), [0.15, 0.25, 0.13]),
            "goal": RigidTransform(RotationMatrix.MakeZRotation(np.pi), [0.05, 0.25, 0.13])}
    # X_Gapg = SelectGrasp(station, temp_context, camera_body_indices=[
    #     plant.GetBodyIndices(scenario.camera0)[0],
    #     plant.GetBodyIndices(scenario.camera1)[0],
    #     plant.GetBodyIndices(scenario.camera2)[0],
    #     plant.GetBodyIndices(scenario.camera3)[0],
    #     plant.GetBodyIndices(scenario.camera4)[0],
    #     plant.GetBodyIndices(scenario.camera5)[0],
    #     plant.GetBodyIndices(scenario.camera6)[0],
    #     plant.GetBodyIndices(scenario.camera7)[0]
    # ], crop_lower=[0.1, 0.2, 0.01], crop_upper=[0.2, 0.3, 1], meshcat=meshcat)
    # X1_O["initial"] = RigidTransform(RotationMatrix.MakeZRotation(np.pi/2.0),
    #                                  [0.15, 0.25, X_Gapg.translation()[-1]-0.11])
    p1_result_frame = make_gripper_move_frames(X1_G_init, X1_O, timestep=1, far_pick=True,
                                               last_return=last_return)
    time_spent = p1_result_frame[1]["return_1"] - p1_result_frame[1]["initial_1"]
    p2_result_frame = make_gripper_stop_frames(X2_G_init, timestep=1,
                                               last_return=last_return, time_spent=time_spent)
    last_return = p1_result_frame[1]["return_1"]
    X1_G.update(p1_result_frame[0])
    times1.update(p1_result_frame[1])
    X2_G.update(p2_result_frame[0])
    times2.update(p2_result_frame[1])

    # step 2: p2 move e5 to e2
    X1_G_init = {"initial_2": X1_G_init["initial_1"]}
    X2_G_init = {"initial_2": X2_G_init["initial_1"]}
    X2_O = {"initial": RigidTransform(RotationMatrix.MakeZRotation(np.pi/2.0), [-0.05, 0.05, 0.1]),
            "goal": RigidTransform(RotationMatrix.MakeZRotation(np.pi), [0.25, 0.05, 0.1])}
    p2_result_frame = make_gripper_move_frames(X2_G_init, X2_O, timestep=2,
                                               last_return=last_return)
    time_spent = p2_result_frame[1]["return_2"] - p2_result_frame[1]["initial_2"]
    p1_result_frame = make_gripper_stop_frames(X1_G_init, timestep=2,
                                               last_return=last_return, time_spent=time_spent)
    X1_G.update(p1_result_frame[0])
    times1.update(p1_result_frame[1])
    X2_G.update(p2_result_frame[0])
    times2.update(p2_result_frame[1])

    return X1_G, times1, X2_G, times2


def make_test1_frames(station, scenario, temp_context, meshcat=None):
    plant = scenario._plant
    temp_plant_context = plant.GetMyContextFromRoot(temp_context)
    # step 0
    X1_G_init = {"initial_0": plant.EvalBodyPoseInWorld(temp_plant_context,
                                plant.GetBodyByName("body", scenario.wsg1))}
    X2_G_init = {"initial_0": plant.EvalBodyPoseInWorld(temp_plant_context,
                                plant.GetBodyByName("body", scenario.wsg2))}
    X1_O = {"initial": RigidTransform(RotationMatrix.MakeZRotation(np.pi/2.0), [-0.35, -0.35, 0.1]),
            "goal": RigidTransform(RotationMatrix.MakeZRotation(np.pi), [-0.05, -0.05, 0.1])}
    # X_Gapg = SelectGrasp(station, temp_context, camera_body_indices=[
    #     plant.GetBodyIndices(scenario.camera0)[0],
    #     plant.GetBodyIndices(scenario.camera1)[0],
    #     plant.GetBodyIndices(scenario.camera2)[0],
    #     plant.GetBodyIndices(scenario.camera3)[0],
    #     plant.GetBodyIndices(scenario.camera4)[0],
    #     plant.GetBodyIndices(scenario.camera5)[0],
    #     plant.GetBodyIndices(scenario.camera6)[0],
    #     plant.GetBodyIndices(scenario.camera7)[0]
    # ], crop_lower=[-0.4, -0.4, 0.01], crop_upper=[-0.3, -0.3, 1], meshcat=meshcat)
    # print("X_Gapg")
    # print(X_Gapg)
    # X1_O["initial"] = RigidTransform(RotationMatrix.MakeZRotation(np.pi/2.0),
    #                                  [-0.35, -0.35, X_Gapg.translation()[-1]-0.11])
    # X1_O["initial"] = X_Gapg

    X1_G, times1 = make_gripper_move_frames(X1_G_init, X1_O, timestep=0, far_place=False)
    time_spent = times1["return_0"] - times1["initial_0"]
    X2_G, times2 = make_gripper_stop_frames(X2_G_init, timestep=0, time_spent=time_spent)

    return X1_G, times1, X2_G, times2


def make_test2_frames(station, scenario, temp_context, meshcat=None):
    plant = scenario._plant
    temp_plant_context = plant.GetMyContextFromRoot(temp_context)
    # step 0
    X1_G_init = {"initial_0": plant.EvalBodyPoseInWorld(temp_plant_context,
                                plant.GetBodyByName("body", scenario.wsg1))}
    X2_G_init = {"initial_0": plant.EvalBodyPoseInWorld(temp_plant_context,
                                plant.GetBodyByName("body", scenario.wsg2))}
    X1_O = {"initial": RigidTransform(RotationMatrix.MakeZRotation(np.pi/2.0), [0.25, -0.35, 0.07]),
            "goal": RigidTransform(RotationMatrix.MakeZRotation(np.pi), [0.15, 0.35, 0.07])}

    X1_G, times1 = make_gripper_move_frames(X1_G_init, X1_O, timestep=0, far_pick=True, far_place=True)
    time_spent = times1["return_0"] - times1["initial_0"]
    X2_G, times2 = make_gripper_stop_frames(X2_G_init, timestep=0, time_spent=time_spent)

    return X1_G, times1, X2_G, times2
