#!/usr/bin/env python

"""
Tests the model
"""

__author__ = "Aamir Hasan"
__version__ = "1.0"
__email__ = "hasanaamir215@gmail.com; aamirh2@illinois.edu"

from sys import stdout
from argparse import ArgumentParser
from models.mesrnn import MESRNN
from models.vanilla_lstm import VLSTM
from data.dataloader import TrajectoryDataset
import utils
from torch.autograd import Variable
from os.path import exists, join
from os import makedirs
import torch
from torch.utils.data.dataset import random_split
from pickle import dump
import logging
from time import time


def test_loop(model, dataset, args, save_prefix="", save=False):
    """
    """
    with torch.no_grad():
        epoch_start = time()
        for i in range(len(dataset)):

            logging.info(f"Test sample {i}/{len(dataset)}")
            start_sample = time()

            # trajectories is a tensor of shape [N, obs_length + pred_length, input_length] which has already been
            # covered to the device on hand
            trajectories, valid_peds, N, _, ped_ids, min_val, max_val, scene_id, dataset_name = dataset[i]
            trajectories = trajectories.float()
            output_trajectories = Variable(torch.zeros((N, args.obs_length + args.pred_length, args.input_length))).to(
                    args.device)

            # initializing state variables for model
            edgeRNN_hidden_states = []
            edgeRNN_cell_states = []
            nodeRNN_hidden_states = []
            nodeRNN_cell_states = []
            for n in range(valid_peds):
                edgeRNN_hidden_states.append([])
                edgeRNN_cell_states.append([])
                nodeRNN_hidden_states.append(Variable(torch.zeros(1, args.nodeRNN_hidden_length).to(args.device)))
                nodeRNN_cell_states.append(Variable(torch.zeros(1, args.nodeRNN_cell_length).to(args.device)))

                for j in range(args.num_edges):
                    edgeRNN_hidden_states[-1].append(
                            Variable(torch.zeros(1, args.edgeRNN_hidden_length)).to(args.device))
                    edgeRNN_cell_states[-1].append(Variable(torch.zeros(1, args.edgeRNN_hidden_length)).to(args.device))

            # observed trajectories are taken as is
            output_trajectories[:, :args.obs_length, :] = trajectories[:, :args.obs_length, :]

            logging.info("Starting observed period")
            # observed period
            for t in range(1, args.obs_length):
                # get edges for all nodes
                all_node_edges = utils.getEdgesAndMetaPaths(trajectories, valid_peds, N, t - 1, args.use_ss or
                                                            args.use_st or args.use_ts or args.use_tt)

                # learn for all valid peds
                for n in range(valid_peds):
                    # get all relevant edge features
                    edges = []
                    if args.use_spatial:
                        edges.append(all_node_edges[n][utils.SPATIAL_IDX])
                    if args.use_temporal:
                        edges.append(all_node_edges[n][utils.TEMPORAL_IDX])
                    if args.use_ss:
                        edges.append(all_node_edges[n][utils.SS_IDX])
                    if args.use_st:
                        edges.append(all_node_edges[n][utils.ST_IDX])
                    if args.use_ts:
                        edges.append(all_node_edges[n][utils.TS_IDX])
                    if args.use_tt:
                        edges.append(all_node_edges[n][utils.TT_IDX])

                    # run through
                    _, edgeRNN_hidden_states[n], edgeRNN_cell_states[n], \
                    nodeRNN_hidden_states[n], nodeRNN_cell_states[n] = \
                        model(node_pos=trajectories[n, t - 1, :], edges=edges,
                              edgeRNN_hidden_states=edgeRNN_hidden_states[n],
                              edgeRNN_cell_states=edgeRNN_cell_states[n],
                              nodeRNN_hidden_state=nodeRNN_hidden_states[n], 
                              nodeRNN_cell_state=nodeRNN_cell_states[n])

            logging.info("Starting predicted period")
            # prediction period
            for t in range(args.obs_length, args.obs_length + args.pred_length):
                # get edges for all nodes
                all_node_edges = utils.getEdgesAndMetaPaths(output_trajectories, valid_peds, N, t - 1, args.use_ss or
                                                            args.use_st or args.use_ts or args.use_tt)

                for n in range(valid_peds):
                    # get all relevant edge features
                    edges = []
                    if args.use_spatial:
                        edges.append(all_node_edges[n][utils.SPATIAL_IDX])
                    if args.use_temporal:
                        edges.append(all_node_edges[n][utils.TEMPORAL_IDX])
                    if args.use_ss:
                        edges.append(all_node_edges[n][utils.SS_IDX])
                    if args.use_st:
                        edges.append(all_node_edges[n][utils.ST_IDX])
                    if args.use_ts:
                        edges.append(all_node_edges[n][utils.TS_IDX])
                    if args.use_tt:
                        edges.append(all_node_edges[n][utils.TT_IDX])

                    # run through
                    output_traj, edgeRNN_hidden_states[n], edgeRNN_cell_states[n], \
                    nodeRNN_hidden_states[n], nodeRNN_cell_states[n] = \
                        model(node_pos=trajectories[n, t - 1, :], edges=edges,
                              edgeRNN_hidden_states=edgeRNN_hidden_states[n],
                              edgeRNN_cell_states=edgeRNN_cell_states[n],
                              nodeRNN_hidden_state=nodeRNN_hidden_states[n], 
                              nodeRNN_cell_state=nodeRNN_cell_states[n])
                    output_trajectories[n, t, :] = output_traj

            # Save trajectories if needed
            if save:
                # save input and output trajectories
                traj_save_path = join(args.save_dir, save_prefix, "trajectories")
                if not exists(traj_save_path):
                    makedirs(traj_save_path)
                    logging.info(f"Created directory {traj_save_path}.")
                traj_save_path = join(traj_save_path, f"{i}.pkl")

                logging.info(f"Saving trajectories saved at {traj_save_path}.")
                dump([trajectories, output_trajectories, valid_peds, N, ped_ids, min_val, max_val, scene_id, dataset_name], open(traj_save_path, 'wb'))
                logging.info(f"Trajectories saved.")

            # save the plots
            plot_save_path = join(args.plot_dir, save_prefix)
            if not exists(plot_save_path):
                makedirs(plot_save_path)
                logging.info(f"Created directory {plot_save_path}.")

            logging.info("Plotting trajectories")
            plot_save_path = join(plot_save_path, str(i))
            # utils.plotTrajectories(trajectories, output_trajectories, valid_peds, N, args.obs_length, plot_save_path, test=True)

            logging.info(f"Time for sample: {time() - start_sample}")
        logging.info(f"Finished running epoch loop on {len(dataset)} scenes. Took {time() - epoch_start} seconds.")


def main():
    args = parse_args()

    # getting device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Adding one more argument to
    args.device = device

    #log_path = join(args.log_dir, str(args.test_dataset))
    log_path = join(args.log_dir)
    if not exists(log_path):
        makedirs(log_path)

    log_path = join(log_path, f"test_{args.model}_{args.test_dataset}.log")
    
    logging.basicConfig(filename=log_path, datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.NOTSET)
    if args.show_log_on_out:
        logging.getLogger().addHandler(logging.StreamHandler(stdout))
    print(f"Logs written to {log_path}.")

    # loading the dataset
    print(args.data_dir)

    test_set = TrajectoryDataset(args.test_dataset, args.data_dir, test=True)
    
    # defining the model
    model = MESRNN(args.input_length, args.output_length, args.num_edges,
                   args.edgeRNN_embed_length, args.edgeRNN_hidden_length, args.edgeRNN_cell_length,
                   args.nodeRNN_embed_length, args.nodeRNN_hidden_length, args.nodeRNN_cell_length,
                   args.dropout_p).to(device)
    if args.model != "mesrnn":
        # Can add more model types here,
        # ensure that all models accept the forward function with the following prototype so that
        # we can use the same training, validation and testing loop code and not change that.
        #
        # If your model does not use meta-paths, set the use_ss, use_st, use_ts and use_tt
        # arguments to False - that way some (most of the) computation can be avoided.
        if args.model == "srnn":
            model = MESRNN(args.input_length, args.output_length, args.num_edges,
                   args.edgeRNN_embed_length, args.edgeRNN_hidden_length, args.edgeRNN_cell_length,
                   args.nodeRNN_embed_length, args.nodeRNN_hidden_length, args.nodeRNN_cell_length,
                   args.dropout_p).to(device)
        elif args.model == "vlstm":
            model = VLSTM(args.input_length, args.output_length,
                          args.nodeRNN_embed_length, args.nodeRNN_hidden_length, args.nodeRNN_cell_length,
                          args.dropout_p).to(device)
        else:
            raise ValueError("No such model type")
    
    model_path = join(args.save_dir, args.model_name)
    if not exists(model_path):
        print(f"Model {model_path} does not exist")
        exit(-1)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    logging.info(f"Loaded model from file {args.model_name}.")

    logging.info(f"Running Testing.")
    save_prefix = f"test_{args.save_name}"

    test_loop(model, test_set, args, save_prefix=save_prefix, save=True)
    
    logging.info("Finished.")
    print("Finished.")


def parse_args():
    """
    Parses the arguments to the executable

    :return: args
    :rtype: dictionary of all the arguments passed in
    """

    parser = ArgumentParser()
    # Input and output length
    parser.add_argument('--input_length', type=int, default=2,
                        help='Dimension of the input trajectory')
    parser.add_argument('--output_length', type=int, default=2,
                        help='Dimension of the output trajectory')

     # number of edges to be used and which types should be used
    parser.add_argument('--num_edges', type=int, default=6,
                        help='Number of edge types to be used for the model')
    parser.add_argument('--use_spatial', action='store_true', default=False,
                        help='If the model should take spatial edges as inputs')
    parser.add_argument('--use_temporal', action='store_true', default=False,
                        help='If the model should take temporal edges as inputs')
    parser.add_argument('--use_ss', action='store_true', default=False,
                        help='If the model should take spatial-spatial meta-paths as inputs')
    parser.add_argument('--use_st', action='store_true', default=False,
                        help='If the model should take spatial-temporal meta-paths as inputs')
    parser.add_argument('--use_ts', action='store_true', default=False,
                        help='If the model should take temporal-spatial meta-paths as inputs')
    parser.add_argument('--use_tt', action='store_true', default=False,
                        help='If the model should take temporal-temporal meta-paths as inputs')

    # Edge RNN Arguments
    parser.add_argument('--edgeRNN_embed_length', type=int, default=64,
                        help='Dimension of the embedder in the edgeRNNs')
    parser.add_argument('--edgeRNN_hidden_length', type=int, default=128,
                        help='Dimension of the hidden state for edgeRNNs')
    parser.add_argument('--edgeRNN_cell_length', type=int, default=128,
                        help='Dimension of the cell state for edgeRNNs')

    # Node RNN Arguments
    parser.add_argument('--nodeRNN_embed_length', type=int, default=128,
                        help='Dimension of the embedder in the edgeRNNs')
    parser.add_argument('--nodeRNN_hidden_length', type=int, default=256,
                        help='Dimension of the hidden state for edgeRNNs')
    parser.add_argument('--nodeRNN_cell_length', type=int, default=256,
                        help='Dimension of the cell state for edgeRNNs')

    # other model params
    parser.add_argument('--dropout_p', type=float, default=0,
                        help='Dropout probability for model')

    # dataset arguments
    parser.add_argument('--obs_length', type=int, default=9,
                        help='Length of the observed sequence')
    parser.add_argument('--pred_length', type=int, default=12,
                        help='Length of the predicted sequence')

    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to the directory with the data')
    parser.add_argument('--save_dir', type=str, default='./save',
                        help='Path to the directory to save the models and trajectories')
    parser.add_argument('--plot_dir', type=str, default='./plot',
                        help='Path to the directory to save the plots to')
    parser.add_argument('--log_dir', type=str, default='./log',
                        help='path to the directory to store the logs')

    parser.add_argument('--show_log_on_out', action='store_true',
                        help='flag if the log output should be shown on the screen')

    parser.add_argument('--model', type=str, default="mesrnn",
                        help='Name of the model to use, ex: mesrnn, srnn, vlstm')

    parser.add_argument('--model_name', type=str, default='MESRNN_99',
                        help='name of the model in the save directory')

    parser.add_argument('--save_name', type=str, default='', help="The name of the folder in the save directory. path will be 'save_dir/test_<this_name>'")

    parser.add_argument('--test_dataset', type=int, default=0,
                        help='If using ETH/UCY, the number of the test dataset')

    args = parser.parse_args()
    if args.obs_length + args.pred_length != 21:
        raise ValueError("Observed and predicted length do not add up to 20")
    return args


if __name__ == "__main__":
    main()
