import numpy as np

import pydrake.all
from pydrake.all import *
from manipulation.utils import AddPackagePaths, FindResource
from manipulation.scenarios import (AddFloatingRpyJoint, AddRgbdSensor,
                                    AddRgbdSensors, ycb)

from scenarios import *


def GraspCandidateCost(diagram,
                       context,
                       cloud,
                       wsg_body_index=None,
                       plant_system_name="plant",
                       scene_graph_system_name="scene_graph",
                       adjust_X_G=False,
                       verbose=False,
                       meshcat_path=None):
    """
    Args:
        diagram: A diagram containing a MultibodyPlant+SceneGraph that contains
            a free body gripper and any obstacles in the environment that we
            want to check collisions against. It should not include the objects
            in the point cloud; those are handled separately.
        context: The diagram context.  All positions in the context will be
            held fixed *except* the gripper free body pose.
        cloud: a PointCloud in world coordinates which represents candidate
            grasps.
        wsg_body_index: The body index of the gripper in plant.  If None, then
            a body named "body" will be searched for in the plant.

    Returns:
        cost: The grasp cost

    If adjust_X_G is True, then it also updates the gripper pose in the plant
    context.
    """
    plant = diagram.GetSubsystemByName(plant_system_name)
    plant_context = plant.GetMyMutableContextFromRoot(context)
    scene_graph = diagram.GetSubsystemByName(scene_graph_system_name)
    scene_graph_context = scene_graph.GetMyMutableContextFromRoot(context)
    if wsg_body_index:
        wsg = plant.get_body(wsg_body_index)
    else:
        wsg = plant.GetBodyByName("body")
        wsg_body_index = wsg.index()

    X_G = plant.GetFreeBodyPose(plant_context, wsg)

    # Transform cloud into gripper frame
    X_GW = X_G.inverse()
    p_GC = X_GW @ cloud.xyzs()

    # Crop to a region inside of the finger box.
    crop_min = [-.05, 0.1, -0.00625]
    crop_max = [.05, 0.1125, 0.00625]
    indices = np.all((crop_min[0] <= p_GC[0,:], p_GC[0,:] <= crop_max[0],
                      crop_min[1] <= p_GC[1,:], p_GC[1,:] <= crop_max[1],
                      crop_min[2] <= p_GC[2,:], p_GC[2,:] <= crop_max[2]),
                     axis=0)

    if meshcat_path:
        pc = PointCloud(np.sum(indices))
        pc.mutable_xyzs()[:] = cloud.xyzs()[:, indices]
        meshcat.SetObject("planning/points", pc, rgba=Rgba(1., 0, 0), point_size=0.01)

    if adjust_X_G and np.sum(indices)>0:
        p_GC_x = p_GC[0, indices]
        p_Gcenter_x = (p_GC_x.min() + p_GC_x.max())/2.0
        X_G.set_translation(X_G @ np.array([p_Gcenter_x, 0, 0]))
        plant.SetFreeBodyPose(plant_context, wsg, X_G)
        X_GW = X_G.inverse()

    query_object = scene_graph.get_query_output_port().Eval(
        scene_graph_context)

    # Check collisions between the gripper and the sink
    if query_object.HasCollisions():
        cost = np.inf
        if verbose:
            print("Gripper is colliding with the sink!\n")
            print(f"cost: {cost}")
        return cost

    # Check collisions between the gripper and the point cloud
    margin = 0.0  # must be smaller than the margin used in the point cloud preprocessing.
    for i in range(cloud.size()):
        distances = query_object.ComputeSignedDistanceToPoint(cloud.xyz(i),
                                                              threshold=margin)
        if distances:
            cost = np.inf
            if verbose:
                print("Gripper is colliding with the point cloud!\n")
                print(f"cost: {cost}")
            return cost

    n_GC = X_GW.rotation().multiply(cloud.normals()[:,indices])

    # Penalize deviation of the gripper from vertical.
    # weight * -dot([0, 0, -1], R_G * [0, 1, 0]) = weight * R_G[2,1]
    cost = 20.0*X_G.rotation().matrix()[2, 1]

    # Reward sum |dot product of normals with gripper x|^2
    cost -= np.sum(n_GC[0,:]**2)
    if verbose:
        print(f"cost: {cost}")
        print(f"normal terms: {n_GC[0,:]**2}")
    return cost


def GenerateAntipodalGraspCandidate(diagram,
                                    context,
                                    cloud,
                                    rng,
                                    wsg_model_instance,
                                    wsg_body_index=None,
                                    plant_system_name="plant",
                                    scene_graph_system_name="scene_graph"):
    """
    Picks a random point in the cloud, and aligns the robot finger with the normal of that pixel.
    The rotation around the normal axis is drawn from a uniform distribution over [min_roll, max_roll].
    Args:
        diagram: A diagram containing a MultibodyPlant+SceneGraph that contains
            a free body gripper and any obstacles in the environment that we
            want to check collisions against. It should not include the objects
            in the point cloud; those are handled separately.
        context: The diagram context.  All positions in the context will be
            held fixed *except* the gripper free body pose.
        cloud: a PointCloud in world coordinates which represents candidate
            grasps.
        rng: a np.random.default_rng()
        wsg_body_index: The body index of the gripper in plant.  If None, then
            a body named "body" will be searched for in the plant.

    Returns:
        cost: The grasp cost
        X_G: The grasp candidate
    """
    plant = diagram.GetSubsystemByName(plant_system_name)
    plant_context = plant.GetMyMutableContextFromRoot(context)
    scene_graph = diagram.GetSubsystemByName(scene_graph_system_name)
    scene_graph_context = scene_graph.GetMyMutableContextFromRoot(context)
    if wsg_body_index:
        wsg = plant.get_body(wsg_body_index)
    else:
        wsg = plant.GetBodyByName("body", wsg_model_instance)
        wsg_body_index = wsg.index()

    index = rng.integers(0, cloud.size() - 1)

    # Use S for sample point/frame.
    p_WS = cloud.xyz(index)
    n_WS = cloud.normal(index)

    assert np.isclose(np.linalg.norm(n_WS),
                      1.0), f"Normal has magnitude: {np.linalg.norm(n_WS)}"

    Gx = n_WS # gripper x axis aligns with normal
    # make orthonormal y axis, aligned with world down
    y = np.array([0.0, 0.0, -1.0])
    if np.abs(np.dot(y,Gx)) < 1e-6:
        # normal was pointing straight down.  reject this sample.
        return np.inf, None

    Gy = y - np.dot(y,Gx)*Gx
    Gz = np.cross(Gx, Gy)
    R_WG = RotationMatrix(np.vstack((Gx, Gy, Gz)).T)
    p_GS_G = [0.054 - 0.01, 0.10625, 0]

    # Try orientations from the center out
    min_roll=-np.pi/3.0
    max_roll=np.pi/3.0
    alpha = np.array([0.5, 0.65, 0.35, 0.8, 0.2, 1.0, 0.0])
    for theta in (min_roll + (max_roll - min_roll)*alpha):
        # Rotate the object in the hand by a random rotation (around the normal).
        R_WG2 = R_WG.multiply(RotationMatrix.MakeXRotation(theta))

        # Use G for gripper frame.
        p_SG_W = - R_WG2.multiply(p_GS_G)
        p_WG = p_WS + p_SG_W

        X_G = RigidTransform(R_WG2, p_WG)
        plant.SetFreeBodyPose(plant_context, wsg, X_G)
        cost = GraspCandidateCost(diagram, context, cloud, adjust_X_G=True)
        X_G = plant.GetFreeBodyPose(plant_context, wsg)
        if np.isfinite(cost):
            return cost, X_G

        #draw_grasp_candidate(X_G, f"collision/{theta:.1f}")

    return np.inf, None


def make_internal_model():
    internal_scenario = Scenario(internal_model=True)
    internal_station = internal_scenario.get_diagram()
    InitFullGame(internal_scenario)
    return internal_scenario, internal_station


def SelectGrasp(station, context, camera_body_indices, meshcat=None,
                crop_lower=[-0.4,-0.4,0.001], crop_upper=[0.4,0.4,3]):
    rng = np.random.default_rng()
    internal_scenario, internal_model = make_internal_model()
    internal_model_context = internal_model.CreateDefaultContext()
    body_poses = station.GetOutputPort("body_poses").Eval(context)
    pcd = []
    for i in range(len(camera_body_indices)):
        cloud = station.GetOutputPort(f"camera{i}_point_cloud").Eval(context)
        pcd.append(cloud.Crop(crop_lower, crop_upper))
        pcd[i].EstimateNormals(radius=0.1, num_closest=30)

        # Flip normals toward camera
        X_WC = body_poses[camera_body_indices[i]]
        pcd[i].FlipNormalsTowardPoint(X_WC.translation())
    merged_pcd = Concatenate(pcd)
    down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)
    if meshcat != None:
        meshcat.SetObject("cloud", down_sampled_pcd, point_size=0.003)

    costs = []
    X_Gs = []
    for i in range(500):
        cost, X_G = GenerateAntipodalGraspCandidate(
            internal_model, internal_model_context,
            down_sampled_pcd, rng, wsg_model_instance=internal_scenario.wsg)
        if np.isfinite(cost):
            costs.append(cost)
            X_Gs.append(X_G)

    if len(costs) == 0:
        # Didn't find a viable grasp candidate
        X_WG = RigidTransform(RollPitchYaw(-np.pi / 2, 0, np.pi / 2),
                              [0.5, 0, 0.22])
    else:
        best = np.argmin(costs)
        # best = np.argmin(rotation_costs)
        X_WG = X_Gs[best]

    return X_WG
