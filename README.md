# MESRNN
This repository contains the code for our paper titled "Meta-path Analysis on Spatio-Temporal Graphs for Pedestrian Trajectory Prediction" in ICRA 2022. 
For more details, please refer to the [project website](https://sites.google.com/illinois.edu/mesrnn/home)<!-- and [arXiv preprint](https://arxiv.org/abs/2011.04820)-->.

## Abstract
Spatio-temporal graphs (ST-graphs) have been used to model time series tasks such as traffic forecasting, human motion modeling, and action recognition. The high-level structure and corresponding features from ST-graphs have led to improved performance over traditional architectures. However, current methods tend to be limited by simple features, despite the rich information provided by the full graph structure, which leads to inefficiencies and suboptimal performance in downstream tasks. We propose the use of features derived from meta-paths, walks across different types of edges, in ST-graphs to improve the performance of Structural Recurrent Neural Network. In this paper, we present the Meta-path Enhanced Structural Recurrent Neural Network (MESRNN), a generic framework that can be applied to any spatio-temporal task in a simple and scalable manner. We employ MESRNN for pedestrian trajectory prediction, utilizing these meta-path based features to capture the relationships between the trajectories of pedestrians at different points in time and space. We compare our MESRNN against state-of-the-art ST-graph methods on standard datasets to show the performance boost provided by meta-path information. The proposed model consistently outperforms the baselines in trajectory prediction over long time horizons by over 32%, and produces more socially compliant trajectories in dense crowds.

## Setup
1. Install Python3.8 (The code may work with other versions of Python, but 3.8 is highly recommended).
2. Install the required python package using pip or conda. For pip, use the following command:  
```
pip install -r requirements.txt
```
For conda, please install each package in `requirements.txt` into your conda environment manually and 
follow the instructions on the anaconda website.  

## Getting started
This repository is organized as follows: 
- The `models/` folder contains the definition of the MESRNN and Vanilla LSTM models. 
- The `data/` folder contains the dataloader and the preprocessing scripts.
- `train.py` is the main training script. Use the command `python train.py --help` to see how to use the script.
- `test.py` is the main testing script. Use the command `python test.py --help` to see how to use the script.

Please run the `create_dirs.sh` script to create the `log`, `save` and `plot` directories.
```
./create_dirs.sh
```
If you encounter errors, please make sure that the script is marked as executable, if they are not, please run
```
chmod +x create_dirs.sh
```

### Dataset
The `data` folder contains the preprocessed trajectories from all 5 scenes in the ETH-UCY dataset from [this source](https://github.com/erichhhhho/DataExtraction).
In order to preprocess the data to be used by the dataloader, please run `preprocess.sh` script.
```
./preprocess.sh
```
If you encounter errors, please make sure that the script is marked as executable, if they are not, please run
```
chmod +x preprocess.sh
```
Please remember to save the values output by the script as these are the values that the trajectories were normalized using, i.e, the min and max values for each dimension. They will be needed to calculate metrics later.

This should create a directory called `pre_processed` in each scene directory. 
The `data` directory should now have the following structure:
```
- data
----- eth_hotel
--------- ppprocessed.csv
--------- pre_processed
------------- 0.pkl
------------- 1.pkl
            ...
----- eth_univ
--------- ppprocessed.csv
--------- pre_processed
------------- 0.pkl
------------- 1.pkl
            ...
----- ucy_zara01
        ...
----- ucy_zara02
        ...
----- ucy_univ
        ...
----- dataloader.py
----- preprocessor.py 
``` 
You should now be able to run the training and testing scripts.

### Training the models
To train a model you can run the `train.py` script. 
Run the following command to see how to use the script.
```
python train.py --help
```

It is recommended for users to read and thoroughly understand the output of the help prompt before training the model.

To train the `MESRNN` model for 100 epochs that will be tested on the `ETH_Univ` dataset, please run
```
python train.py --epochs 100 \
                --test_dataset 0 \
                --num_edges 6 \
                --use_temporal --use_spatial --use_ss --use_st --use_ts -use_tt \
                --model "mesrnn" 
```

To train the `SRNN` model for 100 epochs that will be tested on the `ETH_Univ` dataset, please run
```
python train.py --epochs 100 \
                --test_dataset 0 \
                --num_edges 2 \
                --use_temporal --use_spatial \
                --model "srnn"
```

To train the `Vanilla LSTM` model for 100 epochs that will be tested on the `ETH_Univ` dataset, please run
```
python train.py --epochs 100 \
                --test_dataset 0 \
                --num_edges 0 \
                --model "vlstm"
```

Models can be trained on other datasets by changing the `test_dataset` argument.
Please look at the dataloader in `./data` for more information.

### Testing the models
Once the models are trained, you can test the models by using the `test.py` script.

To test the `MESRNN` model on the `ETH_Univ` dataset, please run
```
python test.py --model_name "MESRNN_99" \
                --test_dataset 0 \
                --num_edges 6 \
                --use_temporal --use_spatial --use_ss --use_st --use_ts -use_tt \
                --save_name mesrnn_0 --save_dir ./save/mesrnn_0
```

To test the `SRNN` model on the `ETH_Univ` dataset, please run
```
python test.py --model_name "MESRNN_99" \
                --test_dataset 0 \
                --num_edges 2 \
                --use_temporal --use_spatial \
                --data_dir ./data \
                --save_name srnn_0 --save_dir ./save/srnn_0
```

To test the `VLSTM` model on the `ETH_Univ` dataset, please run
```
python test.py --model_name "VLSTM_99" \
                --test_dataset 0 \
                --num_edges 0 \
                --data_dir ./data \
                --save_name vlstm_0 --save_dir ./save/vlstm_0
```

### Getting Metrics
Once the models have been tested, the predicted trajectories are stored in pkl files in the folders .

Modify lines 53 and 54 of `get_metrics.py` to be the the values printed out by the preprocessing script.

Run the following to get the ADE and FDE for the different models.
```
python get_metrics.py \
    --load_path ./save/mesrnn_0/test_mesrnn_99/trajectories \
    --csv_save_path ./results --save_name mesrnn_99
```

Running this script will output a csv file with the ADE and FDE for each trajectory and also display the ADE and FDE for that dataset.

## Citation
If you find the code or the paper useful for your research, please cite our paper:
```
@inproceedings{hasan2022metapath,
  title={Meta-path Analysis on Spatio-Temporal Graphs for Pedestrian Trajectory Prediction},
  author={Hasan, Aamir and Sriram, Pranav and Driggs-Campbell, Katherine},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2022}
}
```

## Credits
Other contributors:  
[Pranav Sriram]()  

## Contact
If you have any questions or find any bugs, please feel free to open an issue or pull request.