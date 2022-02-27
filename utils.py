#!/usr/bin/env python

"""
Contains utility functions for the model
"""

__author__ = "Aamir Hasan"
__version__ = "1.0"
__email__ = "hasanaamir215@gmail.com; aamirh2@illinois.edu"

import matplotlib.pyplot as plt
import numpy as np
import colorsys
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

IGNORE_TENSOR = torch.ones(2).to(device) * -2
ZERO_TENSOR = torch.zeros(2).to(device)

SPATIAL_IDX = 0
TEMPORAL_IDX = 1
SS_IDX = 2
ST_IDX = 3
TS_IDX = 4
TT_IDX = 5


def getEdgesAndMetaPaths(trajectories, num_valid_peds, total_peds, t, compute_metapaths):
    """
    Calculates and returns all edges in the scene needed for time t

    :param compute_metapaths: If metapaths should be computed
    :type compute_metapaths: bool
    :param trajectories: Positions of the pedestrians in the scene
    :type trajectories: Tensor (N, T, 2)
    :param num_valid_peds: Number of valid pedestrians in the scene
    :type num_valid_peds: int
    :param total_peds: total number of pedestrians in the scene
    :type total_peds: int
    :param t: time index to calculate the paths at
    :type t: int 0 <= t < 20
    :return: metapaths array for each node
    :rtype: [[[spatial_edges], [temporal_edges], [ss_paths], [st_paths], [ts_paths], [tt_paths]] .. per valid ped]
    """
    """
    S - spatial; T - Temporal
    P_i^t = (x_i^t, y_i^t)
    S_{i,j}^t = P_i^t - P_j^t                               -> O(n^2)
    T_i^t = P_i^t - P_i^{t-1}                               -> O(n)

    Meta-paths are calculated as
    SS_{i, j}^t = S_{i, k}^t - S_{k, j}^t where k \in nodes -> O(n^3)
    ST_{i, j}^t = S_{i, j}^t - T_{j}^t                      -> O(n^2)
    TS_{i, j}^t = T_{i}^t - S_{i, j}^{t-1}                  -> O(n^2)
    TT_{i}^t = T_{i}^{t} - T_{i}^{t-1}                      -> O(n)
    """
    metapaths = []

    for i in range(num_valid_peds):
        # format lists to have placeholders
        metapaths.append([[], [], [], [], [], []])

    spatial_edge_t = calculateSpatialEdges(trajectories[:, t, :], total_peds)

    if compute_metapaths:
        # extract spatial edges and ss-metapaths and format them correctly
        for i in range(num_valid_peds):
            for j in range(total_peds):
                # dont format spatial edges if not needed
                if i != j:
                    metapaths[i][SPATIAL_IDX].append(spatial_edge_t[i, j, :].to(device))

                # dont calculate and format SS paths if not needed
                for k in range(total_peds):
                    if (k != i) and (k != j):
                        metapaths[i][SS_IDX].append((spatial_edge_t[i, j, :] * spatial_edge_t[j, k, :]).to(device))

    # at least one step has been taken in time
    if t > 0:
        temporal_edge_t = calculateTemporalEdges(trajectories[:, t - 1:t + 1, :], total_peds)

        if compute_metapaths:
            spatial_edge_t_1 = calculateSpatialEdges(trajectories[:, t - 1, :], total_peds)

            for i in range(num_valid_peds):
                # Temporal edges
                metapaths[i][TEMPORAL_IDX].append(temporal_edge_t[i, :].to(device))

                for j in range(total_peds):
                    if i != j:
                        # Spatial-Temporal Edges
                        metapaths[i][ST_IDX].append((spatial_edge_t[i, j, :] * temporal_edge_t[j, :]).to(device))

                        # Temporal-Spatial edges
                        metapaths[i][TS_IDX].append((temporal_edge_t[i, :] * spatial_edge_t_1[i, j, :]).to(device))

            # at least two steps have been taken in time, TT edges are now defined
            if t > 1:
                temporal_edge_t_1 = calculateTemporalEdges(trajectories[:, t - 2:t, :], total_peds)

                for i in range(num_valid_peds):
                    metapaths[i][TT_IDX].append((temporal_edge_t[i, :] * temporal_edge_t_1[i, :]).to(device))

    return metapaths


def calculateSpatialEdges(positions, N):
    """
    Calculates all spatial edge features given current positions of the nodes

    :param positions: positions of all nodes at current time step
    :type positions: tensor (N, 2)
    :param N: Total number of nodes
    :type N: int
    :return: All spatial edge features for current positions
    :rtype: Tensor (N, N, 2)
    """
    edges = torch.zeros((N, N, 2))
    for i in range(N):
        for j in range(i + 1, N):
            edges[i, j, :] = positions[i, :] - positions[j, :]
            edges[j, i, :] = -1*edges[i, j, :]

    edges = torch.clamp(edges, -1.0, 1.0)
    return edges.to(device)


def calculateTemporalEdges(positions, N):
    """
    Calculate all temporal edge features given current and previous positions of the nodes

    :param positions: positions of all nodes at current time step and previous time step
    :type positions: Tensor (N, 2, 2)
    :param N: total number of nodes
    :type N: int
    :return: All temporal edge features for current positions
    :rtype: Tensor (N, 2)
    """
    edges = torch.zeros((N, 2))

    for i in range(N):
        edges[i, :] = positions[i, 1, :] - positions[i, 0, :]

    edges = torch.clamp(edges, -1.0, 1.0)
    return edges.to(device)


def plotTrajectories(input_trajectories, output_trajectories, last_valid_idx, num_peds, obs_length,
                    save_dir, draw_output=True, show_figure=False, test=False):
    """
    Plots and saves the trajectories of the scene

    :param input_trajectories: ground trurth trajectories
    :type input_trajectories: Tensor (N, 20, 2)
    :param output_trajectories: predicted trajectories
    :type output_trajectories: Tensor (N, 20, 2)
    :param last_valid_idx: Number of valid pedestrians
    :type last_valid_idx: int
    :param num_peds: Number of total pedestrians
    :type num_peds: int
    :param obs_length: length of the observation period
    :type obs_length: int
    :param save_dir: Directory to save the plots in
    :type save_dir: str
    :param draw_output: If output trajectory should be drawn on the plot
    :type draw_output: Bool
    :param show_figure: If the figures should be shown when theyre plotted
    :type show_figure: Bool
    :return: None
    :rtype: None
    """
    # new figure for scene
    plt.figure(figsize=(10, 5))
    input_traj = input_trajectories.cpu().detach().numpy()
    output_traj = output_trajectories.cpu().detach().numpy()
    if draw_output:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.set_title("Input Trajectories")
        ax2.set_title("Output Trajectories")
        ax3.set_title("Overlaid Trajectories")
    else:
        plt.title("Input Trajectories")

    # plot all trajectories
    for n in range(num_peds):
        color = np.random.rand(3,)

        # draw observed trajectory in a lighter color
        lighter_color = list(colorsys.rgb_to_hls(color[0], color[1], color[2]))
        lighter_color[1] = 1 - 0.5 * (1 - lighter_color[1])
        lighter_color = colorsys.hls_to_rgb(lighter_color[0], lighter_color[1], lighter_color[2])
        lighter_color = list(lighter_color)

        if n < last_valid_idx:
            traj_x = []
            traj_y = []
            out_traj_x = []
            out_traj_y = []

            for t in range(21):
                if not torch.equal(input_trajectories[n, t, :], ZERO_TENSOR) and not torch.equal(input_trajectories[n, t, :], IGNORE_TENSOR):
                    traj_x.append(input_traj[n, t, 0])
                    traj_y.append(input_traj[n, t, 1])
            for t in range(21):
                if test or (not torch.equal(input_trajectories[n, t, :], ZERO_TENSOR) and not torch.equal(input_trajectories[n, t, :], IGNORE_TENSOR)):
                    out_traj_x.append(output_traj[n, t, 0])
                    out_traj_y.append(output_traj[n, t, 1])
            if draw_output:
                # if output should be shown, draws input and output trajectories side by side and overlaid as well
                # otherwise just draws the input trajectories
                ax1.plot(traj_x, traj_y, c=color, marker='.')

                ax2.plot(out_traj_x[:obs_length + 1], out_traj_y[:obs_length + 1], c=lighter_color,
                       marker='+')
                ax2.plot(out_traj_x[obs_length:], out_traj_y[obs_length:], c=color, marker='+')

                ax3.plot(traj_x, traj_y, c=color, linestyle="dashed")
                ax3.plot(out_traj_x[:obs_length + 1], out_traj_y[:obs_length + 1], c=lighter_color,
                       linestyle='None', marker='+')
                ax3.plot(out_traj_x[obs_length:], out_traj_y[obs_length:], c=color, linestyle='None',
                       marker='+')
            else:
                plt.plot(traj_x, traj_y, c=color, marker='.')
        else:
            traj_x = []
            traj_y = []

            for t in range(21):
                if not torch.equal(input_trajectories[n, t, :], ZERO_TENSOR) and not torch.equal(input_trajectories[
                                                                                                 n, t, :], IGNORE_TENSOR):
                    traj_x.append(input_traj[n, t, 0])
                    traj_y.append(input_traj[n, t, 1])
            if draw_output:
                # see comment above
                ax1.plot(traj_x, traj_y, c=color, marker='1')
                ax2.plot(traj_x, traj_y, c=color, marker='1')
                ax3.plot(traj_x, traj_y, c=color, marker='1')
            else:
                plt.plot(traj_x, traj_y, c=color, marker='1')

    # show the plot if needed
    if show_figure:
        plt.show()
    plt.tight_layout()
    # saving the plot
    plt.savefig(save_dir + ".png")
    plt.close()

def plotLossCurve(losses, save_name, title=""):
    """
    Plot the loss curves

    :param losses: all the losses for the process
    :type losses: list
    :param save_name: the name to save the figure with
    :type save_name: str
    :param title: the title for the figure
    :type title: str
    :return: None
    :rtype: None
    """
    # new figure for loss
    plt.figure()

    plt.plot(losses, np.arange(1, len(losses) + 1))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.savefig(save_name)
