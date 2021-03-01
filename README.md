# Cloud Removal from Satellite Imagery using Multispectral Edge-filtered Conditional Generative Adversarial Networks

This repository provides the code, models, datasets, and results of the manuscript titled "Cloud Removal from Satellite Imagery using Multispectral Edge-filtered Conditional Generative Adversarial Networks".

# Datasets

The datasets described in the manuscript, i.e. the edge-intensive, large, and small datasets, are made available <a href="">here</a>.

# Code & Installation

The MEcGANs folder contains the implementation of our method. It extends the original McGANs method introduced in <a href="https://arxiv.org/abs/1710.04835">K. Enomoto et. al. Filmy Cloud Removal on Satellite Imagery with Multispectral Conditional Generative Adversarial Nets. In Proc. IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 1533-1541, 2017.</a>

To setup the Python environment for MEcGANs, please consult the 'Requirements' section of McGANs' README.md available [here](https://github.com/enomotokenji/mcgan-cvprw2017-chainer).

To run MEcGANs on one of the datasets described in the manuscript, please take the following steps.

1. Download the MEcGANs folder with its content.
2. Download the 'Data' folder available [here]() and place it directly into the MEcGANs folder downloaded in Step 1.
3. Replace the MEcGANs/config/config_nirrgb2rgbcloud.yml with a config_nirrgb2rgbcloud.yml file provided for specific a dataset in one of the subfolders of 'Data/datasets'.
4. In the replaced config_nirrgb2rgbcloud.yml file, set the intended NIR cloud penetrability parameter by providing the value of (1 - NIR cloud penetrability) under dataset -> args -> args_train -> nir_cloud_penetrability and dataset -> args -> args_test -> nir_cloud_penetrability. For example, 1% and 0.5% NIR cloud penetrabilities are set with 0.99 and 0.995, respectively.
5. The code of MEcGANs is run in the same way in as the original code of McGANs, i.e. by executing the following command on a GPU-equipped machine:
```
> CUDA_VISIBLE_DEVICES=0 python train_pix2pix.py --config_path configs/config_nirrgb2rgbcloud.yml --results_dir <folder_where_the_results_are_saved>
```

The McGANs_modified folder provides the implementation of the original McGANs method where the generation of clouded images has been modified to comply with the method employed in MEcGANs. The original code of McGANs can be found <a href="https://github.com/enomotokenji/mcgan-cvprw2017-chainer">here</a>. The modified implementation of McGANs can be run in the same way as MEcGANs.

# Results

The complete sets of results presented in the manuscript are made available <a href="">here</a>.

# Trained Model Checkpoints

Model checkpoints of the individual experiments described in the manuscript can be downloaded from [here](). 

A checkpoint can be loaded with the following command, which restarts the training from the chechpoint and runs it till the requested number of iterations in the configs/config_nirrgb2rgbcloud.yml file is reached.

```
> CUDA_VISIBLE_DEVICES=0 python train_pix2pix.py --config_path configs/config_nirrgb2rgbcloud.yml --results_dir <folder_where_the_results_are_saved> --snapshot <snapshot_iter_file>
```
