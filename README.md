# Cloud Removal from Satellite Imagery using Multispectral Edge-filtered Conditional Generative Adversarial Networks

This repository provides the code, models, datasets, and the full set of results for the manuscript titled "Cloud Removal from Satellite Imagery using Multispectral Edge-filtered Conditional Generative Adversarial Networks" by Cengis Hasan, Ross Horne, Sjouke Mauw, and Andrzej Mizera.

# Datasets

The datasets described in the manuscript, i.e., the edge-intensive, large, and small datasets, are made available via Mendeley Data and can be accessed [here](https://data.mendeley.com/datasets/jk3wr7crj7/2).

# Code & Installation

The MEcGANs folder contains the implementation of the MEcGANs method for cloud removal from satellite imagery in [Chainer](https://chainer.org/). It extends the original McGANs method introduced in <a href="https://arxiv.org/abs/1710.04835">K. Enomoto et. al. Filmy Cloud Removal on Satellite Imagery with Multispectral Conditional Generative Adversarial Nets. In Proc. IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 1533-1541, 2017.</a>

To setup the Python environment for MEcGANs, please consult the 'Requirements' section of McGANs' README.md available [here](https://github.com/enomotokenji/mcgan-cvprw2017-chainer). The environment should be installed and MEcGANs should be run on a GPU-equipped machine.

To run MEcGANs on one of the datasets described in the manuscript, please take the following steps.

1. Download the 'MEcGANs' folder and the 'Config_files' folder to locations of your choice.
2. Download the 'Data.zip' file available [here](https://data.mendeley.com/datasets/jk3wr7crj7/2) and unzip it directly in the 'MEcGANs' folder downloaded in Step 1, i.e. the data should be available in 'MEcGANs/Data'.
3. Edit the settings in the 'MEcGANs/config/config_nirrgb2rgbcloud.yml' file or replace it with one of the 'config_nirrgb2rgbcloud.yml' files provided for the three datasets in the respective subfolders of 'Config_files/MEcGANs' where 'Config_files' is the folder downloaded in Step 1.
4. In the edited or replaced 'MEcGANs/config/config_nirrgb2rgbcloud.yml' file, set the intended NIR cloud penetrability parameter by providing the value of (1 - NIR cloud penetrability) under dataset -> args -> args_train -> nir_cloud_penetrability and dataset -> args -> args_test -> nir_cloud_penetrability. For example, 1% and 0.5% NIR cloud penetrabilities are set with 0.99 and 0.995, respectively.
5. The code of MEcGANs is run in the same way in as the original code of McGANs, i.e. by executing the following command on a GPU-equipped machine from the MEcGANs folder:
```
> CUDA_VISIBLE_DEVICES=0 python train_pix2pix.py --config_path configs/config_nirrgb2rgbcloud.yml --results_dir <folder_where_the_results_are_saved>
```

The McGANs_modified folder provides the implementation of the original McGANs method where the generation of clouded images has been modified to comply with the method employed in MEcGANs, i.e. clouded RGB and NIR images are synthesised as described in the manuscript and the discriminator of McGANs is presented with clouded NIR images instead of cloud-free ones as in the original version. The original code of McGANs can be found <a href="https://github.com/enomotokenji/mcgan-cvprw2017-chainer">here</a>. The modified implementation of McGANs can be run in the same way as MEcGANs; for this, follow the five steps above with 'McGANs_modified' replacing 'MEcGANs'.

# Results

The complete sets of results presented in the manuscript are made available [here](https://data.mendeley.com/datasets/jk3wr7crj7/2).

# Running of trained generators

Saved generator models used in the experiments described in the manuscript are placed in the 'Checkpoints' folder available [here](https://data.mendeley.com/datasets/jk3wr7crj7/2). The saved generator model files are named Generator_<number of training iteration>.npz.

To load and run a pretrained generator on an intended dataset of clouded RGB and co-registered NIR images, please use the following command from the MEcGANs folder:

```
> CUDA_VISIBLE_DEVICES=0 python test.py --dir_nir <path_to_clouded_nir_images_dir> --dir_rgb <path_to_clouded_rgb_images_dir> --imlist_nir <path_to_clouded_nir_files_list> --imlist_rgb <path_to_clouded_rgb_files_list> --results_dir <path_to_results_dir> --config_path <path_to_config_nirrgb2rgbcloud.yml> --gen_model <path_to_Generator_<iterations>.npz>
```

# Trained Model Checkpoints

Model checkpoints of the individual experiments described in the manuscript can be downloaded from [here](https://data.mendeley.com/datasets/jk3wr7crj7/2). 

A checkpoint can be loaded with the following command, which resumes the training from the chechpoint and runs it till the requested number of iterations specified in the configs/config_nirrgb2rgbcloud.yml file is reached.

```
> CUDA_VISIBLE_DEVICES=0 python train_pix2pix.py --config_path configs/config_nirrgb2rgbcloud.yml --results_dir <folder_where_the_results_are_saved> --snapshot <snapshot_iter_file>
```

For example, to load the model of MEcGANs trained for 500K iterations with the edge-intensive dataset and NIR cloud penetrability set to 1%, please use the following command from the MEcGANs folder:

```
> CUDA_VISIBLE_DEVICES=0 python train_pix2pix.py --config_path ../Config_files/MEcGANs/edge-intensive_dataset/config_nirrgb2rgbcloud.yml --results_dir <folder_where_the_results_are_saved> --snapshot ../Checkpoints/edge-intensive_dataset/NIR_penetrability_0.01/MEcGANs/snapshot_iter_500000
```
