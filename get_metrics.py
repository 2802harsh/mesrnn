#!/usr/bin/env python
"""
Calculates the metrics for testing ETH/UCY outputs.

Input files with input and output trajectories
"""

__author__ = "Aamir Hasan"
__version__ = "1.0"
__email__ = "hasanaamir215@gmail.com; aamirh2@illinois.edu"

from argparse import ArgumentParser
from os.path import exists, join
from os import makedirs, listdir
import pickle
import numpy as np
import csv

def get_length(trajectory):
    end_pos = 20
    for j in range(20):
        if trajectory[j, 0] == -2 and trajectory[j, 1] == -2:
            end_pos = j
    return end_pos


def calculate_ADE(input_trajectory, output_trajectory):
    diff = output_trajectory - input_trajectory
    return np.sqrt((diff**2).sum(axis=-1)).mean()


def calculate_FDE(input_trajectory, output_trajectory):
    diff = output_trajectory[-1, :] - input_trajectory[-1, :]
    return np.sqrt((diff**2).sum(axis=-1))


def process_file(filename):
    # each file contains
    # trajectories, output_trajectories, valid_peds, N, ped_ids, min_val, max_val, scene_id, dataset_name
    # no need to look through dataset name since, all test files come in from a single dataset
    num_peds_per_tstep = np.zeros(12)
    other_ades = np.zeros(12)
    ade = 0
    fde = 0
    ade_obs = 0
    data = pickle.load(open(filename, 'rb'))

    # trajectories, output_trajectories, valid_peds, N, ped_ids, min_val, max_val, scene_id, dataset_name
    num_peds = data[2]
    input_trajectories = data[0].cpu().numpy()
    output_trajectories = data[1].cpu().numpy()

    max_vals = [1.0208, 1.0278]
    min_vals = [-0.99653, -1.0167]

    input_trajectories[:, :, 0] = (max_vals[0] - min_vals[0]) * (input_trajectories[:, :, 0] + 1) / 2 + min_vals[0]
    input_trajectories[:, :, 1] = (max_vals[1] - min_vals[1]) * (input_trajectories[:, :, 1] + 1) / 2 + min_vals[1]
    output_trajectories[:, :, 0] = (max_vals[0] - min_vals[0]) * (output_trajectories[:, :, 0] + 1) / 2 + min_vals[0]
    output_trajectories[:, :, 1] = (max_vals[1] - min_vals[1]) * (output_trajectories[:, :, 1] + 1) / 2 + min_vals[1]

    #   go through all trajectories and calculate
    #   ADE and FDE - functions
    for i in range(num_peds):
        end_pos = get_length(input_trajectories[i, :, :])
        ade += calculate_ADE(input_trajectories[i, :end_pos, :], output_trajectories[i, :end_pos, :])
        fde += calculate_FDE(input_trajectories[i, :end_pos, :], output_trajectories[i, :end_pos, :])
        ade_obs += calculate_ADE(input_trajectories[i, 8:end_pos, :], output_trajectories[i, 8:end_pos, :])
        for j in range(12):
            if 9 + j <= end_pos:
                other_ades[j] += calculate_ADE(input_trajectories[i, 8:9+j, :], output_trajectories[i, 8:9+j, :])
                num_peds_per_tstep[j] += 1
    #ade = ade
    #fde = fde / num_peds
    return ade, fde, ade_obs, num_peds, data[3], other_ades, num_peds_per_tstep


def parse_args():
    """
    Parses the arguments to the executable,

    :return: args
    :rtype: dictionary of all the arguments passed in
    """
    parser = ArgumentParser()

    parser.add_argument('--load_path', type=str, default='./save',
                        help='Path to the directory with the saved trajectories')

    parser.add_argument('--csv_save_path', type=str, default='./results',
                        help='Path to the save the results csv')

    parser.add_argument('--save_name', type=str, default='eth_ucy_0',
                        help='Name of the saved csv')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if not exists(args.load_path):
        raise ValueError("Saved files do not exist")

    if not exists(args.csv_save_path):
        makedirs(args.csv_save_path)

    csv_out_file = open(join(args.csv_save_path, args.save_name + ".csv"), 'w')
    csv_writer = csv.writer(csv_out_file, delimiter=',')
    csv_writer.writerow(['filename', 'ade', 'fde', 'ade_obs', 'num_peds', 'total_peds', 'ade_0', 'ade_1', 'ade_2', 'ade_3', 'ade_4', 'ade_5', 'ade_6', 'ade_7', 'ade_8', 'ade_9', 'ade_10', 'ade_11', 'num_0', 'num_1', 'num_2', 'num_3', 'num_4', 'num_5', 'num_6', 'num_7', 'num_8', 'num_9', 'num_10', 'num_11'])

    running_ade = 0
    running_fde = 0
    running_ade_obs = 0
    num_files = 0

    # go through all files in directory
    for filename in listdir(args.load_path):
        ade, fde, ade_obs, num_peds, N, all_ades, num_peds_t = process_file(join(args.load_path, filename))
        # save to csv
        csv_writer.writerow(np.concatenate([[filename, ade, fde, ade_obs, num_peds, N],  all_ades, num_peds_t]))

        running_ade += ade
        running_fde += fde
        running_ade_obs += ade_obs
        num_files += 1

    # Average out for all files in the directory
    running_fde /= num_files
    running_ade /= num_files
    running_ade_obs /= num_files

    csv_out_file.close()

    # Report
    print(f"FDE: {running_fde}")
    print(f"ADE: {running_ade}")
    print(f"ADE_obs: {running_ade_obs}")
    print("Done!")

if __name__ == "__main__":
    main()