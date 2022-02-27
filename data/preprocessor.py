#!/usr/bin/env python

"""
Preprocesses the data from csv to separate files to be used by the dataloader
"""

__author__ = "Aamir Hasan"
__version__ = "1.0"
__email__ = "hasanaamir215@gmail.com"

import pandas as pd
import argparse
import os
import pickle
import numpy as np

INPUT_CSV_NAME = "ppprocessed.csv"
OUTPUT_DIR = "pre_processed"


def preprocess(args):
    data_path = os.path.join(args.data_path, args.dataset, INPUT_CSV_NAME)
    output_path = os.path.join(args.data_path, args.dataset, OUTPUT_DIR)

    # if input dataset doesnt exist, cry
    if not os.path.exists(data_path):
        raise ValueError("Dataset does not exist")

    # if output folder doesnt exist, make one
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    print(f"Preprocessing {args.dataset}:")

    # load dataset
    dataset = pd.read_csv(data_path)

    # extracting all pedestrian and frame ids
    pedIDs = dataset.PedID.unique()
    frameIDs = dataset.FrameID.unique()

    # The two portions below could probably have been done in a cleaner way but I just wanted a simple solution.
    # TODO: Make the below implementation cleaner and more efficient

    # get map of what frames each pedestrian is in
    ped_frames = { }
    start_frame = { }
    end_frame = { }
    dataset.sort_values(["PedID", "FrameID"], inplace=True)

    x_max = dataset["x"].max()
    x_min = dataset["x"].min()
    y_min = dataset["y"].min()
    y_max = dataset["y"].max()
    
    print(x_max, x_min,  y_max, y_min)
    
    dataset["x"] = 2*(dataset["x"] - x_min)/(x_max - x_min) - 1
    dataset["y"] = 2*(dataset["y"] - y_min)/(y_max - y_min) - 1

    for pedID in pedIDs:
        ped_frames[pedID] = dataset[dataset.PedID == pedID].FrameID

        """
        The dataset was annotated at 2.5 frames per second. We want to observe trajectories for 3.2s (8 frames)
        and then predict trajectories for 4.8s (12 frames). Therefore, we need to remove any trajectories that are 
        shorter than this time period (20 frames).
        """
        if len(ped_frames[pedID]) < 21:
            del (ped_frames[pedID])
        else:
            start_frame[pedID] = ped_frames[pedID].min()
            end_frame[pedID] = ped_frames[pedID].max()

    # get map of what pedestrians are in each frame
    frame_peds = { }
    dataset.sort_values(["FrameID", "PedID"], inplace=True)
    for frameID in frameIDs:
        frame_peds[frameID] = dataset[dataset.FrameID == frameID].PedID.to_numpy()

    print(f"There are {len(pedIDs)} pedestrians in the dataset across {len(frameIDs)} frames.")

    """
     For simplicity we just divide the frames into groups of 21 frames (we will call this a scene) with skipping 1 + 
     skip frames between each scene. 
     The default skip is 0 so there is an overlap of 20 frames between two consecutive scenes.
     
     The data for each scene consists of:
        1. [{(x, y)_{v_1}}_{t=0}^{T_{pred}}, {(x, y)_{v_1}}_{t=0}^{T_{pred}}, ....] coordinates for all pedestrians in 
            that scene.
        2. V, the index of the last valid pedestrian. A valid pedestrian is one who's trajectory should be 
            predicted. Validity is decided based on how long the pedestrian has been in the scene. Only pedestrians who 
            are in the scene for the whole time are accounted for here, other pedestrians are included in the data 
            above but will not have their trajectories predicted.
        3. Number of pedestrians in scene - to make computation easier for later
    """
    print(f"Constructing scene for {args.dataset}")

    scene_id = 0
    for start_frame in range(frameIDs.min(), frameIDs.max() - 21 + 1, args.skip + 1):
        end_frame = start_frame + 21 * args.frame_stride

        peds_in_scene = set()
        # calculate number of valid pedestrians
        all_frames_valid = True
        for frameID in range(start_frame, end_frame, args.frame_stride):
            if frameID in frameIDs:
                peds_in_scene.update(frame_peds[frameID])
            else:
                all_frames_valid = False
                start_frame = frameID + 1
                # print(f"{frameID} not in frames.")

        if not all_frames_valid:
            continue

        valid_peds = set(frame_peds[start_frame]).intersection(set(frame_peds[end_frame - 1*args.frame_stride]))
        invalid_peds = peds_in_scene - valid_peds

        if len(valid_peds) > 0:
            num_peds_in_scene = len(peds_in_scene)

            # for all pedestrians in the scene create tensor with their coordinates
            # coordinates of (-2, -2) indicate that the pedestrian is not in the scene at that frame
            trajectories = np.ones((num_peds_in_scene, 21, 2)) * -2

            pedIdx = 0
            for pedID in valid_peds:
                # valid pedestrians are in all the frames of the scene
                trajectories[pedIdx, :, :] = dataset[(dataset.PedID == pedID) & (dataset.FrameID >= start_frame)
                                                     & (dataset.FrameID < end_frame)][['x', 'y']].to_numpy()[
                                             ::args.frame_stride]
                pedIdx += 1

            for pedID in invalid_peds:
                # invalid pedestrians will not be all the frames of the scene
                for frameID in range(0, 21*args.frame_stride, args.frame_stride):
                    if pedID in frame_peds[start_frame + frameID]:
                        trajectories[pedIdx, int(frameID/args.frame_stride), :] = dataset[(dataset.PedID == pedID) &
                                                                   (dataset.FrameID == (frameID + start_frame))
                                                                   ][["x", "y"]].to_numpy()
                pedIdx += 1

            # save the scene if there are pedestrians in it
            with open(os.path.join(output_path, f"{scene_id}.pkl"), 'wb') as f:
                pickle.dump([trajectories, len(valid_peds), num_peds_in_scene], f)
            scene_id += 1

    print(f"Constructed {scene_id} scenes and saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset', type=str, help='The dataset to be preprocessed')
    parser.add_argument('--data_path', type=str, default="./", help='path to the data directories')
    parser.add_argument('--skip', type=int, default=0, help='number of frames to skip between two samples')
    parser.add_argument('--frame_stride', type=int, default=1, help='number of frames before extracting next ped '
                                                                    'position')
    parser.add_argument('--sample_length', type=int, default=20, help='sample length for the trajectories')

    args = parser.parse_args()

    preprocess(args)


if __name__ == "__main__":
    main()

"""
Unused comments which might be used somewhere else:

     each sample for pedestrian _v_ will contain the data for sample_length time steps
     the data at each time step consists of: 
                    (x, y)          the interpolated pixel coordinates
                    e^{S}_{v.}    the spatial edge
                    e^{T}_{vv}    the temporal edge
                    e^{SS}_{v.}   the spatial-spatial meta-path
                    e^{ST}_{v.}   the spatial-temporal meta-path
                    e^{TS}_{v.}   the temporal-spatial meta-path
                    e^{TT}_{vv}   the temporal-temporal meta-path
                    
    Since some trajectories might be greater than 20 time steps long, there might be an overlap between 
    trajectories for samples. This overlap can be set by using the skip parameter.
    
    calculating the number of samples for the current pedestrian, 
                |------------------------------length------------------------------|
     sample1-   |-----sample-----|
     sample2-   <- 1+skip ->|-----sample-----|
     sample3-   <----- 2*(1+skip) ----->|-----sample-----|
     ...
                                                                                   |
     sampleN-   <----------------- N*(1+skip) ------------------->|-----sample-----|   
   
   
   sample_id = 0           # sample ID to save the sample by
    for pedID in pedIDs:
        # if not enough frames, move on
        if len(ped_frames[pedID]) < args.sample_length:
            continue

        # Assuming that frameIDs are continuous.
        

        num_samples = ((len(ped_frames[pedID]) - args.sample_length) / (1+args.skip)) + 1

        start_index = 0
        end_index = args.sample_length 
"""
