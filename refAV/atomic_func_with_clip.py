import numpy as np
from pathlib import Path
from typing import Literal
from copy import deepcopy
import inspect

from refAV.clip_helper import clip_sim
from refAV.utils import (
    cache_manager, composable, composable_relational, get_cuboid_from_uuid,
    get_ego_SE3, get_ego_uuid,
    get_map, get_nth_pos_deriv, get_nth_radial_deriv,
    get_nth_yaw_deriv, get_pedestrian_crossings,
    get_pos_within_lane, get_road_side, get_scenario_lanes,
    get_scenario_timestamps, get_timestamps, get_uuids_of_category,
    get_semantic_lane, cuboid_distance, to_scenario_dict,
    unwrap_func, dilate_convex_polygon, polygons_overlap, is_point_in_polygon,
    swap_keys_and_listed_values, has_free_will, at_stop_sign_, remove_empty_branches,
    scenario_at_timestamps, reconstruct_track_dict, create_mining_pkl,
    post_process_scenario, get_object
)


@composable_relational
@cache_manager.create_cache('has_objects_in_relative_direction')
def has_objects_in_relative_direction(
        track_candidates: dict,
        related_candidates: dict,
        log_dir: Path,
        direction: Literal["forward", "backward", "left", "right"],
        min_number: int = 1,
        max_number: int = np.inf,
        within_distance: float = 50,
        lateral_thresh: float = np.inf) -> dict:
    """
    Identifies tracked objects with at least the minimum number of related candidates in the specified direction.
    If the minimum number is met, will create relationships equal to the max_number of closest objects.
    """

    track_uuid = track_candidates

    # CLIP 早停：检查“object on its <direction> side”
    prompt = f"object on its {direction} side"
    if not clip_sim(track_uuid, log_dir, prompt):
        return [], {}

    candidate_uuids = related_candidates

    if track_uuid == get_ego_uuid(log_dir):
        # Ford Fusion dimensions offset from ego_coordinate frame
        track_width = 1
        track_front = 4.877 / 2 + 1.422
        track_back = 4.877 - (4.877 / 2 + 1.422)
    else:
        track_cuboid = get_cuboid_from_uuid(track_uuid, log_dir)
        track_width = track_cuboid.width_m / 2
        track_front = track_cuboid.length_m / 2
        track_back = -track_cuboid.length_m / 2

    timestamps_with_objects = []
    objects_in_relative_direction = {}
    in_direction_dict = {}

    for candidate_uuid in candidate_uuids:
        if candidate_uuid == track_uuid:
            continue

        pos, timestamps = get_nth_pos_deriv(candidate_uuid, 0, log_dir, coordinate_frame=track_uuid)

        for i in range(len(timestamps)):
            if direction == 'left' and pos[i, 1] > track_width and (
                    track_back - lateral_thresh < pos[i, 0] < track_front + lateral_thresh) \
               or direction == 'right' and pos[i, 1] < -track_width and (
                    track_back - lateral_thresh < pos[i, 0] < track_front + lateral_thresh) \
               or direction == 'forward' and pos[i, 0] > track_front and (
                    -track_width - lateral_thresh < pos[i, 1] < track_width + lateral_thresh) \
               or direction == 'backward' and pos[i, 0] < track_back and (
                    -track_width - lateral_thresh < pos[i, 1] < track_width + lateral_thresh):

                if not in_direction_dict.get(timestamps[i], None):
                    in_direction_dict[timestamps[i]] = []

                distance = cuboid_distance(track_uuid, candidate_uuid, log_dir, timestamp=timestamps[i])
                in_direction_dict[timestamps[i]].append((candidate_uuid, distance))

    for timestamp, objects in in_direction_dict.items():
        sorted_objects = sorted(objects, key=lambda row: row[1])

        count = 0
        true_uuids = []
        for candidate_uuid, distance in sorted_objects:
            if distance <= within_distance and count < max_number:
                count += 1
                true_uuids.append(candidate_uuid)

        if count >= min_number:
            for true_uuid in true_uuids:
                if true_uuid not in objects_in_relative_direction:
                    objects_in_relative_direction[true_uuid] = []
                objects_in_relative_direction[true_uuid].append(timestamp)
                timestamps_with_objects.append(timestamp)

    return timestamps_with_objects, objects_in_relative_direction


@cache_manager.create_cache('get_objects_in_relative_direction')
def get_objects_in_relative_direction(
        track_candidates: dict,
        related_candidates: dict,
        log_dir: Path,
        direction: Literal["forward", "backward", "left", "right"],
        min_number: int = 0,
        max_number: int = np.inf,
        within_distance: float = 50,
        lateral_thresh: float = np.inf) -> dict:
    """
    Returns a scenario dictionary of the related candidates that are in the relative direction of the track candidates.
    """
    tracked_objects = reverse_relationship(
        has_objects_in_relative_direction
    )(track_candidates, related_candidates, log_dir,
      direction,
      min_number=min_number, max_number=max_number,
      within_distance=within_distance,
      lateral_thresh=lateral_thresh)

    return tracked_objects


def get_objects_of_category(log_dir, category) -> dict:
    """
    Returns all objects from a given category from the log annotations,
    or supplemented by CLIP if annotation is missing.
    """
    all_uuids = get_uuids_of_category(log_dir, "ANY")  # 获取场景里所有 track
    result_uuids = []
    for uuid in all_uuids:
        # Annotation 判定
        ann_ok = uuid in get_uuids_of_category(log_dir, category)
        # CLIP 辅助判定
        prompt = category.lower().replace('_', ' ')
        clip_ok = clip_sim(uuid, log_dir, prompt)
        if ann_ok or clip_ok:
            result_uuids.append(uuid)

    return to_scenario_dict(result_uuids, log_dir)


@composable
def is_category(track_candidates: dict, log_dir: Path, category: str):
    """
    Returns all objects of a given category (annotation OR CLIP).
    """
    track_uuid = track_candidates

    ann_ok = track_uuid in get_uuids_of_category(log_dir, category)
    prompt = category.lower().replace('_', ' ')
    clip_ok = clip_sim(track_uuid, log_dir, prompt)

    if ann_ok or clip_ok:
        return unwrap_func(get_object)(track_uuid, log_dir)
    return []


@composable
@cache_manager.create_cache('turning')
def turning(
        track_candidates: dict,
        log_dir: Path,
        direction: Literal["left", "right", None] = None) -> dict:
    """
    Returns objects that are turning in the given direction.
    （无 CLIP 辅助）
    """
    track_uuid = track_candidates

    if direction and direction != 'left' and direction != 'right':
        direction = None
        print("Specified direction must be 'left', 'right', or None. Direction set to None automatically.")

    TURN_ANGLE_THRESH = 45  # degrees
    ANG_VEL_THRESH = 5  # deg/s

    ang_vel, timestamps = get_nth_yaw_deriv(track_uuid, 1, log_dir, coordinate_frame='self', in_degrees=True)

    turn_dict = {'left': [], 'right': []}

    start_index = 0
    end_index = start_index

    while start_index < len(timestamps) - 1:
        # Check if the object is continuing to turn in the same direction
        if ((ang_vel[start_index] > 0 and ang_vel[end_index] > 0
             or ang_vel[start_index] < 0 and ang_vel[end_index] < 0)
                and end_index < len(timestamps) - 1):
            end_index += 1
        else:
            # Check if the object's angle has changed enough to define a turn
            s_per_timestamp = float(timestamps[1] - timestamps[0]) / 1E9
            if np.sum(ang_vel[start_index:end_index + 1] * s_per_timestamp) > TURN_ANGLE_THRESH:
                turn_dict['left'].extend(timestamps[start_index:end_index + 1])
            elif np.sum(ang_vel[start_index:end_index + 1] * s_per_timestamp) < -TURN_ANGLE_THRESH:
                turn_dict['right'].extend(timestamps[start_index:end_index + 1])
            elif (unwrap_func(near_intersection)(track_uuid, log_dir)
                  and (start_index == 0 and unwrap_func(near_intersection)(track_uuid, log_dir)[0] == timestamps[0]
                       or end_index == len(timestamps) - 1 and unwrap_func(near_intersection)(track_uuid, log_dir)[
                           -1] == timestamps[-1])):

                if (((start_index == 0 and ang_vel[start_index] > ANG_VEL_THRESH)
                     or (end_index == len(timestamps) - 1 and ang_vel[end_index] > ANG_VEL_THRESH))
                        and np.mean(ang_vel[start_index:end_index + 1]) > ANG_VEL_THRESH
                        and np.sum(ang_vel[start_index:end_index + 1] * s_per_timestamp) > TURN_ANGLE_THRESH / 3):
                    turn_dict['left'].extend(timestamps[start_index:end_index + 1])
                elif (((start_index == 0 and ang_vel[start_index] < -ANG_VEL_THRESH)
                       or (end_index == len(timestamps) - 1 and ang_vel[end_index] < -ANG_VEL_THRESH))
                      and np.mean(ang_vel[start_index:end_index + 1]) < -ANG_VEL_THRESH
                      and np.sum(ang_vel[start_index:end_index + 1] * s_per_timestamp) < -TURN_ANGLE_THRESH / 3):
                    turn_dict['right'].extend(timestamps[start_index:end_index + 1])

            start_index = end_index
            end_index += 1

    if direction:
        return turn_dict[direction]
    else:
        return turn_dict['left'] + turn_dict['right']


@composable
@cache_manager.create_cache('changing_lanes')
def changing_lanes(
        track_candidates: dict,
        log_dir: Path,
        direction: Literal["left", "right", None] = None) -> dict:
    """
    Identifies lane change events for tracked objects in a scenario.
    （无 CLIP 辅助）
    """
    track_uuid = track_candidates

    if direction is not None and direction != 'right' and direction != 'left':
        print("Direction must be 'right', 'left', or None.")
        print("Setting direction to None.")
        direction = None

    COS_SIMILARITY_THRESH = .5  # vehicle must be headed in a direction at most 45 degrees from the lane boundary
    SIDEWAYS_VEL_THRESH = .1  # m/s

    lane_traj = get_scenario_lanes(track_uuid, log_dir)
    positions, timestamps = get_nth_pos_deriv(track_uuid, 0, log_dir)
    velocities, timestamps = get_nth_pos_deriv(track_uuid, 1, log_dir)

    lane_changes_exact = {'left': [], 'right': []}
    for i in range(1, len(timestamps)):
        prev_lane = lane_traj[timestamps[i - 1]]
        cur_lane = lane_traj[timestamps[i]]

        if prev_lane and cur_lane and abs(velocities[i, 1]) >= SIDEWAYS_VEL_THRESH:
            if prev_lane.right_neighbor_id == cur_lane.id:
                # calculate lane orientation
                closest_waypoint_idx = np.argmin(
                    np.linalg.norm(prev_lane.right_lane_boundary.xyz[:, :2] - positions[i, :2], axis=1))
                start_idx = max(0, closest_waypoint_idx - 1)
                end_idx = min(len(prev_lane.right_lane_boundary.xyz) - 1, closest_waypoint_idx + 1)
                lane_boundary_direction = prev_lane.right_lane_boundary.xyz[end_idx,
                                          :2] - prev_lane.right_lane_boundary.xyz[start_idx, :2]
                lane_boundary_direction /= np.linalg.norm(lane_boundary_direction + 1e-8)
                track_direction = velocities[i, :2] / np.linalg.norm(velocities[i, :2])
                lane_change_cos_similarity = abs(np.dot(lane_boundary_direction, track_direction))

                if lane_change_cos_similarity >= COS_SIMILARITY_THRESH:
                    lane_changes_exact['right'].append(i)
            elif prev_lane.left_neighbor_id == cur_lane.id:
                # calculate lane orientation
                closest_waypoint_idx = np.argmin(
                    np.linalg.norm(prev_lane.left_lane_boundary.xyz[:, :2] - positions[i, :2], axis=1))

                start_idx = min(0, closest_waypoint_idx - 1)
                end_idx = min(len(prev_lane.left_lane_boundary.xyz) - 1, closest_waypoint_idx + 1)
                lane_boundary_direction = prev_lane.left_lane_boundary.xyz[end_idx,
                                          :2] - prev_lane.left_lane_boundary.xyz[start_idx, :2]
                lane_boundary_direction /= np.linalg.norm(lane_boundary_direction + 1e-8)
                track_direction = velocities[i, :2] / np.linalg.norm(velocities[i, :2])
                lane_change_cos_similarity = abs(np.dot(lane_boundary_direction, track_direction))

                if lane_change_cos_similarity >= COS_SIMILARITY_THRESH:
                    lane_changes_exact['left'].append(i)

    lane_changes = {'left': [], 'right': []}

    for index in lane_changes_exact['left']:
        lane_change_start = index - 1
        lane_change_end = index

        while lane_change_start > 0:
            _, pos_along_width0 = get_pos_within_lane(positions[lane_change_start],
                                                      lane_traj[timestamps[lane_change_start]])
            _, pos_along_width1 = get_pos_within_lane(positions[lane_change_start + 1],
                                                      lane_traj[timestamps[lane_change_start + 1]])

            if (pos_along_width0 and pos_along_width1 and pos_along_width0 > pos_along_width1) \
               or lane_change_start == index - 1:
                lane_changes['left'].append(timestamps[lane_change_start])
                lane_change_start -= 1
            else:
                break

        while lane_change_end < len(timestamps):
            _, pos_along_width0 = get_pos_within_lane(positions[lane_change_end - 1],
                                                      lane_traj[timestamps[lane_change_end - 1]])
            _, pos_along_width1 = get_pos_within_lane(positions[lane_change_end],
                                                      lane_traj[timestamps[lane_change_end]])

            if (pos_along_width0 and pos_along_width1 and pos_along_width0 > pos_along_width1) \
               or lane_change_end == index:
                lane_changes['left'].append(timestamps[lane_change_end])
                lane_change_end += 1
            else:
                break

    for index in lane_changes_exact['right']:
        lane_change_start = index - 1
        lane_change_end = index

        while lane_change_start > 0:
            _, pos_along_width0 = get_pos_within_lane(positions[lane_change_start],
                                                      lane_traj[timestamps[lane_change_start]])
            _, pos_along_width1 = get_pos_within_lane(positions[lane_change_start + 1],
                                                      lane_traj[timestamps[lane_change_start + 1]])

            if pos_along_width0 and pos_along_width1 and pos_along_width0 < pos_along_width1 \
               or lane_change_start == index - 1:
                lane_changes['right'].append(timestamps[lane_change_start])
                lane_change_start -= 1
            else:
                break

        while lane_change_end < len(timestamps):
            _, pos_along_width0 = get_pos_within_lane(positions[lane_change_end - 1],
                                                      lane_traj[timestamps[lane_change_end - 1]])
            _, pos_along_width1 = get_pos_within_lane(positions[lane_change_end],
                                                      lane_traj[timestamps[lane_change_end]])

            if pos_along_width0 and pos_along_width1 and pos_along_width0 < pos_along_width1 \
               or lane_change_end == index:
                lane_changes['right'].append(timestamps[lane_change_end])
                lane_change_end += 1
            else:
                break

    if direction:
        lane_changing_timestamps = lane_changes[direction]
    else:
        lane_changing_timestamps = sorted(list(set(lane_changes['left'] + lane_changes['right'])))

    turning_timestamps = unwrap_func(turning)(track_uuid, log_dir)
    return sorted(list(set(lane_changing_timestamps).difference(set(turning_timestamps))))


@composable
@cache_manager.create_cache('has_lateral_acceleration')
def has_lateral_acceleration(
        track_candidates: dict,
        log_dir: Path,
        min_accel=-np.inf,
        max_accel=np.inf) -> dict:
    """
    Objects with a lateral acceleration between the minimum and maximum thresholds.
    Most objects with a high lateral acceleration are turning.
    （无 CLIP 辅助）
    """
    track_uuid = track_candidates

    hla_timestamps = []
    accelerations, timestamps = get_nth_pos_deriv(track_uuid, 2, log_dir, coordinate_frame='self')
    for i, accel in enumerate(accelerations):
        if min_accel <= accel[1] <= max_accel:  # m/s^2
            hla_timestamps.append(timestamps[i])

    if unwrap_func(stationary)(track_candidates, log_dir):
        return []

    return hla_timestamps


@composable_relational
@cache_manager.create_cache('facing_toward')
def facing_toward(
        track_candidates: dict,
        related_candidates: dict,
        log_dir: Path,
        within_angle: float = 22.5,
        max_distance: float = 50) -> dict:
    """
    Identifies objects in track_candidates that are facing toward objects in related candidates.
    （无 CLIP 辅助）
    """
    track_uuid = track_candidates
    facing_toward_timestamps = []
    facing_toward_objects = {}

    for candidate_uuid in related_candidates:
        if candidate_uuid == track_uuid:
            continue

        traj, timestamps = get_nth_pos_deriv(candidate_uuid, 0, log_dir, coordinate_frame=track_uuid)
        for i, timestamp in enumerate(timestamps):

            angle = np.rad2deg(np.arctan2(traj[i, 1], traj[i, 0]))
            distance = cuboid_distance(track_uuid, candidate_uuid, log_dir, timestamp=timestamp)

            if np.abs(angle) <= within_angle and distance <= max_distance:
                facing_toward_timestamps.append(timestamp)

                if candidate_uuid not in facing_toward_objects:
                    facing_toward_objects[candidate_uuid] = []
                facing_toward_objects[candidate_uuid].append(timestamp)

    return facing_toward_timestamps, facing_toward_objects


@composable_relational
@cache_manager.create_cache('heading_toward')
def heading_toward(
        track_candidates: dict,
        related_candidates: dict,
        log_dir: Path,
        angle_threshold: float = 22.5,
        minimum_speed: float = .5,
        max_distance: float = np.inf) -> dict:
    """
    Identifies objects in track_candidates that are heading toward objects in related candidates.
    （无 CLIP 辅助）
    """
    track_uuid = track_candidates
    heading_toward_timestamps = []
    heading_toward_objects = {}

    track_vel, track_timestamps = get_nth_pos_deriv(track_uuid, 1, log_dir, coordinate_frame=track_uuid)

    for candidate_uuid in related_candidates:
        if candidate_uuid == track_uuid:
            continue

        related_pos, _ = get_nth_pos_deriv(candidate_uuid, 0, log_dir, coordinate_frame=track_uuid)
        track_radial_vel, related_timestamps = get_nth_radial_deriv(
            track_uuid, 1, log_dir, coordinate_frame=candidate_uuid)

        for i, timestamp in enumerate(related_timestamps):
            if timestamp not in track_timestamps:
                continue
            timestamp_vel = track_vel[track_timestamps.index(timestamp)]

            vel_direction = timestamp_vel / (np.linalg.norm(timestamp_vel) + 1e-8)
            direction_of_related = related_pos[i] / np.linalg.norm(related_pos[i] + 1e-8)
            angle = np.rad2deg(np.arccos(np.dot(vel_direction, direction_of_related)))

            if -track_radial_vel[i] >= minimum_speed and angle <= angle_threshold \
                    and cuboid_distance(track_uuid, candidate_uuid, log_dir, timestamp) <= max_distance:

                heading_toward_timestamps.append(timestamp)
                if candidate_uuid not in heading_toward_objects:
                    heading_toward_objects[candidate_uuid] = []
                heading_toward_objects[candidate_uuid].append(timestamp)

    return heading_toward_timestamps, heading_toward_objects


@composable
@cache_manager.create_cache('accelerating')
def accelerating(
        track_candidates: dict,
        log_dir: Path,
        min_accel: float = .65,
        max_accel: float = np.inf) -> dict:
    """
    Identifies objects in track_candidates that have a forward acceleration above a threshold.
    （无 CLIP 辅助）
    """
    track_uuid = track_candidates

    acc_timestamps = []
    accelerations, timestamps = get_nth_pos_deriv(track_uuid, 2, log_dir, coordinate_frame='self')
    for i, accel in enumerate(accelerations):
        if min_accel <= accel[0] <= max_accel:  # m/s^2
            acc_timestamps.append(timestamps[i])

    if unwrap_func(stationary)(track_candidates, log_dir):
        return []

    return acc_timestamps


@composable
@cache_manager.create_cache('has_velocity')
def has_velocity(
        track_candidates: dict,
        log_dir: Path,
        min_velocity: float = .5,
        max_velocity: float = np.inf) -> dict:
    """
    Identifies objects with a velocity between the given maximum and minimum velocities in m/s.
    （无 CLIP 辅助）
    """
    track_uuid = track_candidates

    vel_timestamps = []
    vels, timestamps = get_nth_pos_deriv(track_uuid, 1, log_dir)
    for i, vel in enumerate(vels):
        if min_velocity <= np.linalg.norm(vel) <= max_velocity:  # m/s
            vel_timestamps.append(timestamps[i])

    if unwrap_func(stationary)(track_candidates, log_dir):
        return []

    return vel_timestamps


@composable
@cache_manager.create_cache('at_pedestrian_crossing')
def at_pedestrian_crossing(
        track_candidates: dict,
        log_dir: Path,
        within_distance: float = 1) -> dict:
    """
    Identifies objects that within a certain distance from a pedestrian crossing.
    （无 CLIP 辅助）
    """
    track_uuid = track_candidates

    avm = get_map(log_dir)
    ped_crossings = avm.get_scenario_ped_crossings()

    timestamps = get_timestamps(track_uuid, log_dir)
    ego_poses = get_ego_SE3(log_dir)

    timestamps_at_object = []
    for timestamp in timestamps:
        track_cuboid = get_cuboid_from_uuid(track_uuid, log_dir, timestamp=timestamp)
        city_vertices = ego_poses[timestamp].transform_from(track_cuboid.vertices_m)
        track_poly = np.array(
            [city_vertices[2], city_vertices[6], city_vertices[7], city_vertices[3], city_vertices[2]])[:, :2]

        for ped_crossing in ped_crossings:
            pc_poly = ped_crossing.polygon
            pc_poly = dilate_convex_polygon(pc_poly[:, :2], distance=within_distance)
            ped_crossings = get_pedestrian_crossings(avm, track_poly)

            if polygons_overlap(track_poly, pc_poly):
                timestamps_at_object.append(timestamp)

    return timestamps_at_object


@composable
@cache_manager.create_cache('on_lane_type')
def on_lane_type(
        track_uuid: dict,
        log_dir,
        lane_type: Literal["BUS", "VEHICLE", "BIKE"]) -> dict:
    """
    Identifies objects on a specific lane type (annotation OR CLIP).
    """
    # CLIP 早停：检查“on <lane_type> lane”
    prompt = f"on {lane_type.lower()} lane"
    if not clip_sim(track_uuid, log_dir, prompt):
        return []

    scenario_lanes = get_scenario_lanes(track_uuid, log_dir)
    timestamps = scenario_lanes.keys()

    return [timestamp for timestamp in timestamps
            if scenario_lanes[timestamp] and scenario_lanes[timestamp].lane_type == lane_type]


@composable
@cache_manager.create_cache('near_intersection')
def near_intersection(
        track_uuid: dict,
        log_dir: Path,
        threshold: float = 5) -> dict:
    """
    Identifies objects within a specified threshold of an intersection in meters.
    （无 CLIP 辅助）
    """
    traj, timestamps = get_nth_pos_deriv(track_uuid, 0, log_dir)

    avm = get_map(log_dir)
    lane_segments = avm.get_scenario_lane_segments()

    ls_polys = []
    for ls in lane_segments:
        if ls.is_intersection:
            ls_polys.append(ls.polygon_boundary)

    dilated_intersections = []
    for ls in ls_polys:
        dilated_intersections.append(dilate_convex_polygon(ls[:, :2], threshold))

    near_intersection_timestamps = []
    for i, pos in enumerate(traj):
        for dilated_intersection in dilated_intersections:
            if is_point_in_polygon(pos[:2], dilated_intersection):
                near_intersection_timestamps.append(timestamps[i])

    return near_intersection_timestamps


@composable
@cache_manager.create_cache('on_intersection')
def on_intersection(track_candidates: dict, log_dir: Path):
    """
    Identifies objects located on top of a road intersection.
    （无 CLIP 辅助）
    """
    track_uuid = track_candidates

    scenario_lanes = get_scenario_lanes(track_uuid, log_dir)
    timestamps = scenario_lanes.keys()

    timestamps_on_intersection = []
    for timestamp in timestamps:
        if scenario_lanes[timestamp] is not None and scenario_lanes[timestamp].is_intersection:
            timestamps_on_intersection.append(timestamp)

    return timestamps_on_intersection


@composable_relational
@cache_manager.create_cache('being_crossed_by')
def being_crossed_by(
        track_candidates: dict,
        related_candidates: dict,
        log_dir: Path,
        direction: Literal["forward", "backward", "left", "right"] = "forward",
        in_direction: Literal['clockwise', 'counterclockwise', 'either'] = 'either',
        forward_thresh: float = 10,
        lateral_thresh: float = 5) -> dict:
    """
    Identifies objects that are being crossed by one of the related candidate objects.
    """

    track_uuid = track_candidates

    # CLIP 早停：检查“crossing”
    if not clip_sim(track_uuid, log_dir, "crossing"):
        return [], {}

    VELOCITY_THRESH = .2  # m/s

    crossings = {}
    crossed_timestamps = []

    track = get_cuboid_from_uuid(track_uuid, log_dir)
    forward_thresh = track.length_m / 2 + forward_thresh
    left_bound = -track.width_m / 2
    right_bound = track.width_m / 2

    for candidate_uuid in related_candidates:
        if candidate_uuid == track_uuid:
            continue

        candidate_pos, timestamps = get_nth_pos_deriv(
            candidate_uuid, 0, log_dir, coordinate_frame=track_uuid, direction=direction)
        candidate_vel, timestamps = get_nth_pos_deriv(
            candidate_uuid, 1, log_dir, coordinate_frame=track_uuid, direction=direction)

        for i in range(1, len(candidate_pos)):
            y0 = candidate_pos[i - 1, 1]
            y1 = candidate_pos[i, 1]
            y_vel = candidate_vel[i, 1]
            if ((y0 < left_bound < y1 or y1 < right_bound < y0 or y0 < right_bound < y1 or y1 < left_bound < y0)
                and abs(y_vel) > VELOCITY_THRESH) \
               and (track.length_m / 2 <= candidate_pos[i, 0] <= forward_thresh) \
               and candidate_uuid != track_uuid:

                # 1 if moving right, -1 if moving left
                direction_sign = (y1 - y0) / abs(y1 - y0)
                start_index = i - 1
                end_index = i
                updated = True

                if (direction_sign == 1 and in_direction == 'clockwise'
                        or direction_sign == -1 and in_direction == 'counterclockwise'):
                    continue

                while updated:
                    updated = False
                    if start_index >= 0 and direction_sign * candidate_pos[start_index, 1] < lateral_thresh \
                       and direction_sign * candidate_vel[start_index, 1] > VELOCITY_THRESH:
                        if candidate_uuid not in crossings:
                            crossings[candidate_uuid] = []
                        crossings[candidate_uuid].append(timestamps[start_index])
                        crossed_timestamps.append(timestamps[start_index])
                        updated = True
                        start_index -= 1

                    if end_index < len(timestamps) and direction_sign * candidate_pos[end_index, 1] < lateral_thresh \
                       and direction_sign * candidate_vel[end_index, 1] > VELOCITY_THRESH:
                        if candidate_uuid not in crossings:
                            crossings[candidate_uuid] = []
                        crossings[candidate_uuid].append(timestamps[end_index])
                        crossed_timestamps.append(timestamps[end_index])
                        updated = True
                        end_index += 1

    return crossed_timestamps, crossings


@composable_relational
@cache_manager.create_cache('near_objects')
def near_objects(
        track_uuid: dict,
        candidate_uuids: dict,
        log_dir: Path,
        distance_thresh: float = 10,
        min_objects: int = 1,
        include_self: bool = False) -> dict:
    """
    Identifies timestamps when a tracked object is near a specified set of related objects.
    """

    # CLIP 早停：检查“near objects”
    if not clip_sim(track_uuid, log_dir, "near objects"):
        return [], {}

    if not min_objects:
        min_objects = len(candidate_uuids)

    near_objects_dict = {}
    for candidate in candidate_uuids:
        if candidate == track_uuid and not include_self:
            continue

        _, timestamps = get_nth_pos_deriv(candidate, 0, log_dir, coordinate_frame=track_uuid)

        for timestamp in timestamps:
            if cuboid_distance(track_uuid, candidate, log_dir, timestamp) <= distance_thresh:
                if timestamp not in near_objects_dict:
                    near_objects_dict[timestamp] = []
                near_objects_dict[timestamp].append(candidate)

    timestamps = []
    keys = list(near_objects_dict.keys())
    for timestamp in keys:
        if len(near_objects_dict[timestamp]) >= min_objects:
            timestamps.append(timestamp)
        else:
            near_objects_dict.pop(timestamp)

    near_objects_dict = swap_keys_and_listed_values(near_objects_dict)

    return timestamps, near_objects_dict


@composable_relational
@cache_manager.create_cache('following')
def following(
        track_uuid: dict,
        candidate_uuids: dict,
        log_dir: Path) -> dict:
    """
    Returns timestamps when the tracked object is following a lead object.
    """

    # CLIP 早停：检查“following vehicle”
    if not clip_sim(track_uuid, log_dir, "following vehicle"):
        return [], {}

    lead_timestamps = []
    leads = {}

    avm = get_map(log_dir)
    track_lanes = get_scenario_lanes(track_uuid, log_dir, avm=avm)
    track_vel, track_timestamps = get_nth_pos_deriv(track_uuid, 1, log_dir, coordinate_frame=track_uuid)

    track_cuboid = get_cuboid_from_uuid(track_uuid, log_dir)
    track_width = track_cuboid.width_m / 2
    track_length = track_cuboid.length_m / 2

    FOLLOWING_THRESH = 25 + track_length  # m
    LATERAL_TRHESH = 5  # m
    HEADING_SIMILARITY_THRESH = .5  # cosine similarity

    for j, candidate in enumerate(candidate_uuids):
        if candidate == track_uuid:
            continue

        candidate_pos, _ = get_nth_pos_deriv(candidate, 0, log_dir, coordinate_frame=track_uuid)
        candidate_vel, _ = get_nth_pos_deriv(candidate, 1, log_dir, coordinate_frame=track_uuid)
        candidate_yaw, timestamps = get_nth_yaw_deriv(candidate, 0, log_dir, coordinate_frame=track_uuid)
        candidate_lanes = get_scenario_lanes(candidate, log_dir, avm=avm)

        overlap_track_vel = track_vel[np.isin(track_timestamps, timestamps)]
        candidate_heading_similarity = np.zeros(len(timestamps))

        candidate_cuboid = get_cuboid_from_uuid(track_uuid, log_dir)
        candidate_width = candidate_cuboid.width_m / 2

        for i in range(len(timestamps)):
            if np.linalg.norm(candidate_vel[i]) > .5:
                candidate_heading = candidate_vel[i, :2] / np.linalg.norm(candidate_vel[i, :2] + 1e-8)
            else:
                candidate_heading = np.array([np.cos(candidate_yaw[i]), np.sin(candidate_yaw[i])])

            if np.linalg.norm(overlap_track_vel[i]) > .5:
                track_heading = overlap_track_vel[i, :2] / np.linalg.norm(overlap_track_vel[i, :2] + 1e-8)
            else:
                track_heading = np.array([1, 0])

            candidate_heading_similarity[i] = np.dot(track_heading, candidate_heading)

        for i in range(len(timestamps)):
            if track_lanes[timestamps[i]] and candidate_lanes[timestamps[i]] \
               and (((track_lanes[timestamps[i]].id == candidate_lanes[timestamps[i]].id \
                     or candidate_lanes[timestamps[i]].id in track_lanes[timestamps[i]].successors) \
                     and track_length < candidate_pos[i, 0] < FOLLOWING_THRESH \
                     and -LATERAL_TRHESH < candidate_pos[i, 1] < LATERAL_TRHESH \
                     and candidate_heading_similarity[i] > HEADING_SIMILARITY_THRESH) \
                    or (track_lanes[timestamps[i]].left_neighbor_id == candidate_lanes[timestamps[i]].id \
                        or track_lanes[timestamps[i]].right_neighbor_id == candidate_lanes[timestamps[i]].id) \
                    and track_length < candidate_pos[i, 0] < FOLLOWING_THRESH \
                    and (-track_width <= candidate_pos[i, 1] + candidate_width <= track_width \
                         or -track_width <= candidate_pos[i, 1] - candidate_width <= track_width) \
                    and candidate_heading_similarity[i] > HEADING_SIMILARITY_THRESH):

                if candidate not in leads:
                    leads[candidate] = []
                leads[candidate].append(timestamps[i])
                lead_timestamps.append(timestamps[i])

    return lead_timestamps, leads

import numpy as np
from pathlib import Path
from typing import Literal
from copy import deepcopy
import inspect

from refAV.clip_helper import clip_sim
from refAV.utils import (
    cache_manager, composable, composable_relational, get_cuboid_from_uuid,
    get_ego_SE3, get_ego_uuid,
    get_map, get_nth_pos_deriv, get_nth_radial_deriv,
    get_nth_yaw_deriv, get_pedestrian_crossings,
    get_pos_within_lane, get_road_side, get_scenario_lanes,
    get_scenario_timestamps, get_timestamps, get_uuids_of_category,
    get_semantic_lane, cuboid_distance, to_scenario_dict,
    unwrap_func, dilate_convex_polygon, polygons_overlap, is_point_in_polygon,
    swap_keys_and_listed_values, has_free_will, at_stop_sign_, remove_empty_branches,
    scenario_at_timestamps, reconstruct_track_dict, create_mining_pkl,
    post_process_scenario, get_object
)

@cache_manager.create_cache('get_objects_in_relative_direction')
def get_objects_in_relative_direction(
        track_candidates: dict,
        related_candidates: dict,
        log_dir: Path,
        direction: Literal["forward", "backward", "left", "right"],
        min_number: int = 0,
        max_number: int = np.inf,
        within_distance: float = 50,
        lateral_thresh: float = np.inf) -> dict:
    """
    Returns a scenario dictionary of the related candidates that are in the relative direction of the track candidates.
    """
    tracked_objects = reverse_relationship(
        has_objects_in_relative_direction
    )(track_candidates, related_candidates, log_dir,
      direction,
      min_number=min_number, max_number=max_number,
      within_distance=within_distance,
      lateral_thresh=lateral_thresh)

    return tracked_objects



@composable
@cache_manager.create_cache('turning')
def turning(
        track_candidates: dict,
        log_dir: Path,
        direction: Literal["left", "right", None] = None) -> dict:
    """
    Returns objects that are turning in the given direction.
    （无 CLIP 辅助）
    """
    track_uuid = track_candidates

    if direction and direction != 'left' and direction != 'right':
        direction = None
        print("Specified direction must be 'left', 'right', or None. Direction set to None automatically.")

    TURN_ANGLE_THRESH = 45  # degrees
    ANG_VEL_THRESH = 5  # deg/s

    ang_vel, timestamps = get_nth_yaw_deriv(track_uuid, 1, log_dir, coordinate_frame='self', in_degrees=True)

    turn_dict = {'left': [], 'right': []}

    start_index = 0
    end_index = start_index

    while start_index < len(timestamps) - 1:
        # Check if the object is continuing to turn in the same direction
        if ((ang_vel[start_index] > 0 and ang_vel[end_index] > 0
             or ang_vel[start_index] < 0 and ang_vel[end_index] < 0)
                and end_index < len(timestamps) - 1):
            end_index += 1
        else:
            # Check if the object's angle has changed enough to define a turn
            s_per_timestamp = float(timestamps[1] - timestamps[0]) / 1E9
            if np.sum(ang_vel[start_index:end_index + 1] * s_per_timestamp) > TURN_ANGLE_THRESH:
                turn_dict['left'].extend(timestamps[start_index:end_index + 1])
            elif np.sum(ang_vel[start_index:end_index + 1] * s_per_timestamp) < -TURN_ANGLE_THRESH:
                turn_dict['right'].extend(timestamps[start_index:end_index + 1])
            elif (unwrap_func(near_intersection)(track_uuid, log_dir)
                  and (start_index == 0 and unwrap_func(near_intersection)(track_uuid, log_dir)[0] == timestamps[0]
                       or end_index == len(timestamps) - 1 and unwrap_func(near_intersection)(track_uuid, log_dir)[
                           -1] == timestamps[-1])):

                if (((start_index == 0 and ang_vel[start_index] > ANG_VEL_THRESH)
                     or (end_index == len(timestamps) - 1 and ang_vel[end_index] > ANG_VEL_THRESH))
                        and np.mean(ang_vel[start_index:end_index + 1]) > ANG_VEL_THRESH
                        and np.sum(ang_vel[start_index:end_index + 1] * s_per_timestamp) > TURN_ANGLE_THRESH / 3):
                    turn_dict['left'].extend(timestamps[start_index:end_index + 1])
                elif (((start_index == 0 and ang_vel[start_index] < -ANG_VEL_THRESH)
                       or (end_index == len(timestamps) - 1 and ang_vel[end_index] < -ANG_VEL_THRESH))
                      and np.mean(ang_vel[start_index:end_index + 1]) < -ANG_VEL_THRESH
                      and np.sum(ang_vel[start_index:end_index + 1] * s_per_timestamp) < -TURN_ANGLE_THRESH / 3):
                    turn_dict['right'].extend(timestamps[start_index:end_index + 1])

            start_index = end_index
            end_index += 1

    if direction:
        return turn_dict[direction]
    else:
        return turn_dict['left'] + turn_dict['right']


@composable
@cache_manager.create_cache('changing_lanes')
def changing_lanes(
        track_candidates: dict,
        log_dir: Path,
        direction: Literal["left", "right", None] = None) -> dict:
    """
    Identifies lane change events for tracked objects in a scenario.
    （无 CLIP 辅助）
    """
    track_uuid = track_candidates

    if direction is not None and direction != 'right' and direction != 'left':
        print("Direction must be 'right', 'left', or None.")
        print("Setting direction to None.")
        direction = None

    COS_SIMILARITY_THRESH = .5  # vehicle must be headed in a direction at most 45 degrees from the lane boundary
    SIDEWAYS_VEL_THRESH = .1  # m/s

    lane_traj = get_scenario_lanes(track_uuid, log_dir)
    positions, timestamps = get_nth_pos_deriv(track_uuid, 0, log_dir)
    velocities, timestamps = get_nth_pos_deriv(track_uuid, 1, log_dir)

    lane_changes_exact = {'left': [], 'right': []}
    for i in range(1, len(timestamps)):
        prev_lane = lane_traj[timestamps[i - 1]]
        cur_lane = lane_traj[timestamps[i]]

        if prev_lane and cur_lane and abs(velocities[i, 1]) >= SIDEWAYS_VEL_THRESH:
            if prev_lane.right_neighbor_id == cur_lane.id:
                # calculate lane orientation
                closest_waypoint_idx = np.argmin(
                    np.linalg.norm(prev_lane.right_lane_boundary.xyz[:, :2] - positions[i, :2], axis=1))
                start_idx = max(0, closest_waypoint_idx - 1)
                end_idx = min(len(prev_lane.right_lane_boundary.xyz) - 1, closest_waypoint_idx + 1)
                lane_boundary_direction = prev_lane.right_lane_boundary.xyz[end_idx,
                                          :2] - prev_lane.right_lane_boundary.xyz[start_idx, :2]
                lane_boundary_direction /= np.linalg.norm(lane_boundary_direction + 1e-8)
                track_direction = velocities[i, :2] / np.linalg.norm(velocities[i, :2])
                lane_change_cos_similarity = abs(np.dot(lane_boundary_direction, track_direction))

                if lane_change_cos_similarity >= COS_SIMILARITY_THRESH:
                    lane_changes_exact['right'].append(i)
            elif prev_lane.left_neighbor_id == cur_lane.id:
                # calculate lane orientation
                closest_waypoint_idx = np.argmin(
                    np.linalg.norm(prev_lane.left_lane_boundary.xyz[:, :2] - positions[i, :2], axis=1))

                start_idx = min(0, closest_waypoint_idx - 1)
                end_idx = min(len(prev_lane.left_lane_boundary.xyz) - 1, closest_waypoint_idx + 1)
                lane_boundary_direction = prev_lane.left_lane_boundary.xyz[end_idx,
                                          :2] - prev_lane.left_lane_boundary.xyz[start_idx, :2]
                lane_boundary_direction /= np.linalg.norm(lane_boundary_direction + 1e-8)
                track_direction = velocities[i, :2] / np.linalg.norm(velocities[i, :2])
                lane_change_cos_similarity = abs(np.dot(lane_boundary_direction, track_direction))

                if lane_change_cos_similarity >= COS_SIMILARITY_THRESH:
                    lane_changes_exact['left'].append(i)

    lane_changes = {'left': [], 'right': []}

    for index in lane_changes_exact['left']:
        lane_change_start = index - 1
        lane_change_end = index

        while lane_change_start > 0:
            _, pos_along_width0 = get_pos_within_lane(positions[lane_change_start],
                                                      lane_traj[timestamps[lane_change_start]])
            _, pos_along_width1 = get_pos_within_lane(positions[lane_change_start + 1],
                                                      lane_traj[timestamps[lane_change_start + 1]])

            if (pos_along_width0 and pos_along_width1 and pos_along_width0 > pos_along_width1) \
               or lane_change_start == index - 1:
                lane_changes['left'].append(timestamps[lane_change_start])
                lane_change_start -= 1
            else:
                break

        while lane_change_end < len(timestamps):
            _, pos_along_width0 = get_pos_within_lane(positions[lane_change_end - 1],
                                                      lane_traj[timestamps[lane_change_end - 1]])
            _, pos_along_width1 = get_pos_within_lane(positions[lane_change_end],
                                                      lane_traj[timestamps[lane_change_end]])

            if (pos_along_width0 and pos_along_width1 and pos_along_width0 > pos_along_width1) \
               or lane_change_end == index:
                lane_changes['left'].append(timestamps[lane_change_end])
                lane_change_end += 1
            else:
                break

    for index in lane_changes_exact['right']:
        lane_change_start = index - 1
        lane_change_end = index

        while lane_change_start > 0:
            _, pos_along_width0 = get_pos_within_lane(positions[lane_change_start],
                                                      lane_traj[timestamps[lane_change_start]])
            _, pos_along_width1 = get_pos_within_lane(positions[lane_change_start + 1],
                                                      lane_traj[timestamps[lane_change_start + 1]])

            if pos_along_width0 and pos_along_width1 and pos_along_width0 < pos_along_width1 \
               or lane_change_start == index - 1:
                lane_changes['right'].append(timestamps[lane_change_start])
                lane_change_start -= 1
            else:
                break

        while lane_change_end < len(timestamps):
            _, pos_along_width0 = get_pos_within_lane(positions[lane_change_end - 1],
                                                      lane_traj[timestamps[lane_change_end - 1]])
            _, pos_along_width1 = get_pos_within_lane(positions[lane_change_end],
                                                      lane_traj[timestamps[lane_change_end]])

            if pos_along_width0 and pos_along_width1 and pos_along_width0 < pos_along_width1 \
               or lane_change_end == index:
                lane_changes['right'].append(timestamps[lane_change_end])
                lane_change_end += 1
            else:
                break

    if direction:
        lane_changing_timestamps = lane_changes[direction]
    else:
        lane_changing_timestamps = sorted(list(set(lane_changes['left'] + lane_changes['right'])))

    turning_timestamps = unwrap_func(turning)(track_uuid, log_dir)
    return sorted(list(set(lane_changing_timestamps).difference(set(turning_timestamps))))


@composable
@cache_manager.create_cache('has_lateral_acceleration')
def has_lateral_acceleration(
        track_candidates: dict,
        log_dir: Path,
        min_accel=-np.inf,
        max_accel=np.inf) -> dict:
    """
    Objects with a lateral acceleration between the minimum and maximum thresholds.
    Most objects with a high lateral acceleration are turning.
    （无 CLIP 辅助）
    """
    track_uuid = track_candidates

    hla_timestamps = []
    accelerations, timestamps = get_nth_pos_deriv(track_uuid, 2, log_dir, coordinate_frame='self')
    for i, accel in enumerate(accelerations):
        if min_accel <= accel[1] <= max_accel:  # m/s^2
            hla_timestamps.append(timestamps[i])

    if unwrap_func(stationary)(track_candidates, log_dir):
        return []

    return hla_timestamps


@composable_relational
@cache_manager.create_cache('facing_toward')
def facing_toward(
        track_candidates: dict,
        related_candidates: dict,
        log_dir: Path,
        within_angle: float = 22.5,
        max_distance: float = 50) -> dict:
    """
    Identifies objects in track_candidates that are facing toward objects in related candidates.
    （无 CLIP 辅助）
    """
    track_uuid = track_candidates
    facing_toward_timestamps = []
    facing_toward_objects = {}

    for candidate_uuid in related_candidates:
        if candidate_uuid == track_uuid:
            continue

        traj, timestamps = get_nth_pos_deriv(candidate_uuid, 0, log_dir, coordinate_frame=track_uuid)
        for i, timestamp in enumerate(timestamps):

            angle = np.rad2deg(np.arctan2(traj[i, 1], traj[i, 0]))
            distance = cuboid_distance(track_uuid, candidate_uuid, log_dir, timestamp=timestamp)

            if np.abs(angle) <= within_angle and distance <= max_distance:
                facing_toward_timestamps.append(timestamp)

                if candidate_uuid not in facing_toward_objects:
                    facing_toward_objects[candidate_uuid] = []
                facing_toward_objects[candidate_uuid].append(timestamp)

    return facing_toward_timestamps, facing_toward_objects


@composable_relational
@cache_manager.create_cache('heading_toward')
def heading_toward(
        track_candidates: dict,
        related_candidates: dict,
        log_dir: Path,
        angle_threshold: float = 22.5,
        minimum_speed: float = .5,
        max_distance: float = np.inf) -> dict:
    """
    Identifies objects in track_candidates that are heading toward objects in related candidates.
    （无 CLIP 辅助）
    """
    track_uuid = track_candidates
    heading_toward_timestamps = []
    heading_toward_objects = {}

    track_vel, track_timestamps = get_nth_pos_deriv(track_uuid, 1, log_dir, coordinate_frame=track_uuid)

    for candidate_uuid in related_candidates:
        if candidate_uuid == track_uuid:
            continue

        related_pos, _ = get_nth_pos_deriv(candidate_uuid, 0, log_dir, coordinate_frame=track_uuid)
        track_radial_vel, related_timestamps = get_nth_radial_deriv(
            track_uuid, 1, log_dir, coordinate_frame=candidate_uuid)

        for i, timestamp in enumerate(related_timestamps):
            if timestamp not in track_timestamps:
                continue
            timestamp_vel = track_vel[track_timestamps.index(timestamp)]

            vel_direction = timestamp_vel / (np.linalg.norm(timestamp_vel) + 1e-8)
            direction_of_related = related_pos[i] / np.linalg.norm(related_pos[i] + 1e-8)
            angle = np.rad2deg(np.arccos(np.dot(vel_direction, direction_of_related)))

            if -track_radial_vel[i] >= minimum_speed and angle <= angle_threshold \
                    and cuboid_distance(track_uuid, candidate_uuid, log_dir, timestamp) <= max_distance:

                heading_toward_timestamps.append(timestamp)
                if candidate_uuid not in heading_toward_objects:
                    heading_toward_objects[candidate_uuid] = []
                heading_toward_objects[candidate_uuid].append(timestamp)

    return heading_toward_timestamps, heading_toward_objects


@composable
@cache_manager.create_cache('accelerating')
def accelerating(
        track_candidates: dict,
        log_dir: Path,
        min_accel: float = .65,
        max_accel: float = np.inf) -> dict:
    """
    Identifies objects in track_candidates that have a forward acceleration above a threshold.
    （无 CLIP 辅助）
    """
    track_uuid = track_candidates

    acc_timestamps = []
    accelerations, timestamps = get_nth_pos_deriv(track_uuid, 2, log_dir, coordinate_frame='self')
    for i, accel in enumerate(accelerations):
        if min_accel <= accel[0] <= max_accel:  # m/s^2
            acc_timestamps.append(timestamps[i])

    if unwrap_func(stationary)(track_candidates, log_dir):
        return []

    return acc_timestamps


@composable
@cache_manager.create_cache('has_velocity')
def has_velocity(
        track_candidates: dict,
        log_dir: Path,
        min_velocity: float = .5,
        max_velocity: float = np.inf) -> dict:
    """
    Identifies objects with a velocity between the given maximum and minimum velocities in m/s.
    （无 CLIP 辅助）
    """
    track_uuid = track_candidates

    vel_timestamps = []
    vels, timestamps = get_nth_pos_deriv(track_uuid, 1, log_dir)
    for i, vel in enumerate(vels):
        if min_velocity <= np.linalg.norm(vel) <= max_velocity:  # m/s
            vel_timestamps.append(timestamps[i])

    if unwrap_func(stationary)(track_candidates, log_dir):
        return []

    return vel_timestamps


@composable
@cache_manager.create_cache('at_pedestrian_crossing')
def at_pedestrian_crossing(
        track_candidates: dict,
        log_dir: Path,
        within_distance: float = 1) -> dict:
    """
    Identifies objects that within a certain distance from a pedestrian crossing.
    （无 CLIP 辅助）
    """
    track_uuid = track_candidates

    avm = get_map(log_dir)
    ped_crossings = avm.get_scenario_ped_crossings()

    timestamps = get_timestamps(track_uuid, log_dir)
    ego_poses = get_ego_SE3(log_dir)

    timestamps_at_object = []
    for timestamp in timestamps:
        track_cuboid = get_cuboid_from_uuid(track_uuid, log_dir, timestamp=timestamp)
        city_vertices = ego_poses[timestamp].transform_from(track_cuboid.vertices_m)
        track_poly = np.array(
            [city_vertices[2], city_vertices[6], city_vertices[7], city_vertices[3], city_vertices[2]])[:, :2]

        for ped_crossing in ped_crossings:
            pc_poly = ped_crossing.polygon
            pc_poly = dilate_convex_polygon(pc_poly[:, :2], distance=within_distance)
            ped_crossings = get_pedestrian_crossings(avm, track_poly)

            if polygons_overlap(track_poly, pc_poly):
                timestamps_at_object.append(timestamp)

    return timestamps_at_object


@composable
@cache_manager.create_cache('near_intersection')
def near_intersection(
        track_uuid: dict,
        log_dir: Path,
        threshold: float = 5) -> dict:
    """
    Identifies objects within a specified threshold of an intersection in meters.
    （无 CLIP 辅助）
    """
    traj, timestamps = get_nth_pos_deriv(track_uuid, 0, log_dir)

    avm = get_map(log_dir)
    lane_segments = avm.get_scenario_lane_segments()

    ls_polys = []
    for ls in lane_segments:
        if ls.is_intersection:
            ls_polys.append(ls.polygon_boundary)

    dilated_intersections = []
    for ls in ls_polys:
        dilated_intersections.append(dilate_convex_polygon(ls[:, :2], threshold))

    near_intersection_timestamps = []
    for i, pos in enumerate(traj):
        for dilated_intersection in dilated_intersections:
            if is_point_in_polygon(pos[:2], dilated_intersection):
                near_intersection_timestamps.append(timestamps[i])

    return near_intersection_timestamps


@composable
@cache_manager.create_cache('on_intersection')
def on_intersection(track_candidates: dict, log_dir: Path):
    """
    Identifies objects located on top of a road intersection.
    （无 CLIP 辅助）
    """
    track_uuid = track_candidates

    scenario_lanes = get_scenario_lanes(track_uuid, log_dir)
    timestamps = scenario_lanes.keys()

    timestamps_on_intersection = []
    for timestamp in timestamps:
        if scenario_lanes[timestamp] is not None and scenario_lanes[timestamp].is_intersection:
            timestamps_on_intersection.append(timestamp)

    return timestamps_on_intersection
