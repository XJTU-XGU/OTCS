# Optimal Transport-guided Conditional Score-based diffusion model (OTCS)
Official code for "Xiang Gu, Liwei Yang, Jian Sun, and Zongben Xu. Optimal Transport-Guided Conditional Score-Based Diffusion Model. NeurIPS, 2023."[\[OpenReview Version\]](https://openreview.net/forum?id=9Muli2zoFn&referrer=%5Bthe%20profile%20of%20Xiang%20Gu%5D(%2Fprofile%3Fid%3D~Xiang_Gu1)) [\[ArXive Version\]](https://arxiv.org/abs/2311.01226)

To our knowledge, this is the first conditional score-based model for unpaired or partially paired data settings. 

![](https://github.com/XJTU-XGU/OTCS/blob/main/figures/OT_guided_SBDMs.png)
## News
[2023.11] The code for OTCS has been uploaded! Welcome to try out and free to contact xianggu@stu.xjtu.edu.cn.

## Demos
We provide a [1-D demo](https://github.com/XJTU-XGU/OTCS/blob/main/notebooks/demo_1D_unpaired.ipynb) and a [2-D demo](https://github.com/XJTU-XGU/OTCS/blob/main/notebooks/demo_2D_semi_paired.ipynb) of OTCS in the folder of notebooks, to help understand OTCS. Please refer to them.

## Instruction

We provide code for unpaired super-resolution on CelebA dataset and semi-paired image-to-image translation on animal dataset. For your own task, please prepare your datasets as in "the use the code for your project".

### Unpaired super-resolution on CelebA
For celeba, first modify the data_dir in celeba.yml to the dataset, download the pretrained model of [DDIM](https://drive.google.com/file/d/1R_H-fJYXSH79wfSKs9D-fuKQVan5L-GR/view) and put it in path "./pretrained_model". Then train the potential networks by 

```
python train_OT.py --ot_type unsupervised --epsilon 1e-7 --lr 1e-6 --dataset celeba 
```

and train the score_based model by 
```
python main_ddpm.py --config celeba.yml --gpu_id 0,1,2,3
```
Our trained model is available [here](https://drive.google.com/file/d/1-7L0vR8R0qBKcKKC2hM-QeZzr-27xJrA/view?usp=sharing). For inference, run 
 
```
python main_ddpm.py --config celeba.yml --sample True --gpu_id 0,1,2,3
```

![](https://github.com/XJTU-XGU/OTCS/blob/main/figures/qualitative_results_celeba.png)

### Semi-paired image-to-image translation on animal data
For animal images, first download the [animal images](https://drive.google.com/file/d/1-1WhUJV8NkJLZEmQi1TpA5DUq4Zn61pJ/view?usp=sharing), and modify the data_dir in animal.yml to the dataset. Note that the animal images are originally from [AFHQ](https://github.com/clovaai/stargan-v2). Please also cite their paper if you use the animal images. Download the pretrained model of [ILVR on Dog Images](https://onedrive.live.com/?authkey=%21AOIJGI8FUQXvFf8&id=72419B431C262344%21103807&cid=72419B431C262344) and put it in the path "./pretrained_model". Then train the potential networks by 

```
python train_OT.py --ot_type semi-supervised --epsilon 1e-5 --lr 1e-6 --dataset animal --batch_size 256
```

and train the score_based model by 
```
python main_ddpm.py --config animal.yml  --gpu_id 0,1,2,3
```
Our trained model is available [here](https://drive.google.com/file/d/1-LJd-PMVdPhDf0g6yqaTfqtBTBc1C5ej/view?usp=drive_link). For inference, run 
```
python main_ddpm.py --config animal.yml --sample True --gpu_id 0,1,2,3
```

![](https://github.com/XJTU-XGU/OTCS/blob/main/figures/qualitative_results_animal.png)
## How the use the code for your project
For better understanding the following implementation, please refer to Algorithms 1 and 2 in the Appendix.
### For unsupervised OT
#### Stage I: train OT potentials
*The following code should be put in Line 101 of "train_OT.py".*

If you choose unsupervised OT, first prepare the source and target datasets. You may use the ImageFolder for convenience.
Note that each item in the dataset should be like (image,label) where the label is not used.
```python
source_dataset = ImageFolder("root/to/source_data")
target_dataset = ImageFolder("root/to/target_data")
```
Defining the OT solver:
```python
from runners import OT_solver
ot_solver = OT_solver.LargeScaleOTSolver(ot_type="unsupervised")
```
If the datasets are not too large for the computer memory (not GPU), you may preload the images
to the computer memory, which will faster the training process for potentials. This is optional.
```python
# optional
save_dir = "dir for save data or results" # default is "exp/OT/models"
source_dataset = ot_solver.preloading_images_for_dataset(source_dataset,f"{save_dir}/uot_source_images.pkl")
target_dataset = ot_solver.preloading_images_for_dataset(target_dataset,f"{save_dir}/uot_target_images.pkl")
```
Additionally, if you would like performing the unsupervised OT in feature space, you may extract the features and construct feature datasets using the follow function. This is optional.
```python
# optional
feature_extractor = ??? #a pretrained feature extractor
source_dataset = ot_solver.extracting_features_for_dataset(source_dataset,feature_extractor,f"{save_dir}/uot_source_features.pkl")
target_dataset = ot_solver.extracting_features_for_dataset(target_dataset,feature_extractor,f"{save_dir}/uot_target_features.pkl")
```
Defining the configs for the OT solver.
```python
cost = "l2"
# Define the cost. You may choose "l1", "l2", "cosine", which are respectively the mean l1-distance, mean l2-distance, 
# and cosine distance. You can also define your own cost function which takes two batches of Tensors X (N,..),Y (N,..) 
# and returns a N*N tensor with each entry being the distance of a pair from X and Y. 
epsilon = 1e-7 #you may set other values
# The config for the potential networks.
network_dict = {"input_size":64*64*3,"num_hidden_layers":5, "dim_hidden_layers":512
                "act_function":"SiLU"} 
# Feeding in configs                 
ot_solver.feed_unsupervised_OT_params(cost=cost,epsilon=epsilon,**network_dict)
```
Training potential networks.
```python
batch_size = 64 #you may set other values
lr = 1e-6 #you may set other values
iterations = 300000 #you may set other values
ot_solver.train(source_dataset,target_dataset,batch_size=batch_size,lr=lr,
                        num_train_steps=iterations,save_dir=save_dir)
```
Computing and storing potential values for the dataset.
```python
ot_solver.save_potentials(source_dataset,target_dataset,save_dir)
```
Storing the dict of non-zero H.
```python
ot_solver.save_non_zero_dict(source_dataset,target_dataset,save_dir)
```

*Finally, run the following command to train the potential networks.*
```bash
python train_OT.py --ot_type "unsupervised"
```

#### Stage II: train conditional score-based model
You should define the configs as in the "configs/celeba.yml". You may modify the architecture of score-based model in the class "Model" in "functions/models.py", in which you can modify the block for conditioning on the source data. For more details for training the score-based model, you may refer to the Official code of DDIM.

*Next, you can prepare you dataset in Lines 184-187 in the file "main_ddpm.py" as follows:*
```python
source_dataset = ImageFolder("root/to/source_data")
target_dataset = ImageFolder("root/to/target_data")
save_dir = "dir for saving OT results"

```
If you would like the conditioning on source features rather than images, you may use the dataset of features as follows. 
```python
# optional
feature_extractor = ??? #a pretrained feature extractor
from runners import OT_solver
ot_solver = OT_solver.LargeScaleOTSolver(ot_type="unsupervised")
source_dataset = ot_solver.extracting_features_for_dataset(source_dataset,feature_extractor,f"{save_dir}/uot_source_features.pkl")
```
Constructing the unpaired dataset.
```python
from datasets import datasets_factory
unpaired_dataset = datasets_factory.UnPairedDataset(source_dataset,target_datset,f"{save_dir}/non_zero_dict.pkl")
```

*Finally, run the following command to train the score-based model.*
```bash
python main_ddpm.py --config {your config name}.yml --gpu_id 0,1,2,3
```


### For semi-supervised OT
#### Stage I: train OT potentials
*The following code should be put in Line 101 of "train_OT.py".*

If you choose semi-supervised OT, first prepare the source and target unpaired/paired datasets. You may use the ImageFolder for convenience.
Note that each item in the dataset should be like (image,label) where the label is not used.
```python
source_dataset = ImageFolder("root/to/unpaired source data")
target_dataset = ImageFolder("root/to/unpaired targetdata")
source_dataset_paired = ImageFolder("root/to/paired source data")
target_dataset_paired = ImageFolder("root/to/paired target data")
```
Defining the OT solver:
```python
from runners import OT_solver
ot_solver = OT_solver.LargeScaleOTSolver(ot_type="semi-supervised")
```
If the datasets are not too large for the computer memory (not GPU), you may preload the images to the computer memory, which will faster the training process for potentials. This is optional.
```python
# optional
save_dir = "dir for save data or results" # default is "exp/OT/models"
source_dataset = ot_solver.preloading_images_for_dataset(source_dataset,f"{save_dir}/ssot_source_images.pkl")
target_dataset = ot_solver.preloading_images_for_dataset(target_dataset,f"{save_dir}/ssot_target_images.pkl")
source_dataset_paired = ot_solver.preloading_images_for_dataset(source_dataset,f"{save_dir}/ssot_source_images_paired.pkl")
target_dataset_paired = ot_solver.preloading_images_for_dataset(target_dataset,f"{save_dir}/ssot_target_images_paired.pkl")
```
Additionally, if you would like performing the unsupervised OT in feature space, you may extract the features and construct feature datasets using the follow function. This is optional.
```python
# optional
feature_extractor = ??? #a pretrained feature extractor
source_dataset = ot_solver.extracting_features_for_dataset(source_dataset,feature_extractor,f"{save_dir}/ssot_source_features.pkl")
target_dataset = ot_solver.extracting_features_for_dataset(target_dataset,feature_extractor,f"{save_dir}/ssot_target_features.pkl")
source_dataset_paired = ot_solver.extracting_features_for_dataset(source_dataset_paired,feature_extractor,f"{save_dir}/ssot_source_features_paired.pkl")
target_dataset_paired = ot_solver.extracting_features_for_dataset(target_dataset_paired,feature_extractor,f"{save_dir}/ssot_target_features_paired.pkl")
```
Constructing paired dataset.
```python
from from datasets import datasets_factory
paired_dataset = datasets_factory.PairedDataset(source_dataset_paired, target_dataset_paired)
```
Defining the configs for the OT solver.
```python
cost = "l2"
# Define the cost. You may choose "l1", "l2", "cosine", which are respectively the mean l1-distance, mean l2-distance, 
# and cosine distance. You can also define your own cost function which takes two batches of Tensors X (N,..),Y (N,..) 
# and returns a N*N tensor with each entry being the distance of a pair from X and Y. 
epsilon = 1e-5 #you may set other values
alpha = 1.0 #The combination factor of transport cost and guiding loss. you may set other values. Please refer to the
            #KPG-RL-KP model in [1].
tau = 0.1 #you may set other values
# The config for the potential networks.
network_dict = {"input_size":512,"num_hidden_layers":5, "dim_hidden_layers":512
                "act_function":"SiLU"} 
# Feeding in configs                 
ot_solver.feed_semi_supervised_OT_params(cost=cost, epsilon=epsilon, alpha=alpha, tau=tau,**network_dict)
```
Training potential networks.
```python
batch_size = 64 #you may set other values
lr = 1e-6 #you may set other values
iterations = 300000 #you may set other values
ot_solver.train(source_dataset, target_dataset, paired_dataset, batch_size=batch_size, lr=lr, num_train_steps=iterations, save_dir=save_dir)
```
Concat unpaired and paired datasets.
```python
source_dataset_concated = datasets_factory.ConcatDatasets(source_dataset, source_dataset_paired)
target_dataset_concated = datasets_factory.ConcatDatasets(target_dataset, target_dataset_paired)
```
Computing and storing potential values for the dataset.
```python
ot_solver.save_potentials(source_dataset_concated, target_dataset_concated, save_dir)
```
Storing the dict of non-zero H.
```python
ot_solver.save_non_zero_dict(source_dataset_concated, target_dataset_concated, paired_dataset, save_dir)
```

*Finally, run the following command to train the potential networks.*
```bash
python train_OT.py --ot_type "semi-supervised"
```

[1] [Xiang Gu, Yucheng Yang, Wei Zeng, Jian Sun, Zongben Xu. Keypoint-Guided Optimal Transport](https://arxiv.org/abs/2303.13102).


#### Stage II: train conditional score-based model
You should define the configs as in the "configs/celeba.yml". You may modify the architecture of score-based model in the class "Model" in "functions/models.py", in which you can modify the block for conditioning on the source data. For more details for training the score-based model, you may refer to the Official code of DDIM.

*Next, you can prepare you dataset in Lines 184-187 in the file "main_ddpm.py" as follows:*
```python
source_dataset = ImageFolder("root/to/source_data")
target_dataset = ImageFolder("root/to/target_data")
source_dataset_paired = ImageFolder("root/to/paired source data")
target_dataset_paired = ImageFolder("root/to/paired target data")
save_dir = "dir for saving OT results"
```
If you would like the conditioning on source features rather than images, you may use the dataset of features as follows. 
```python
# optional
feature_extractor = ??? #a pretrained feature extractor
from runners import OT_solver
ot_solver = OT_solver.LargeScaleOTSolver(ot_type="unsupervised")
source_dataset = ot_solver.extracting_features_for_dataset(source_dataset,feature_extractor,f"{save_dir}/ssot_source_features.pkl")
source_dataset_paired = ot_solver.extracting_features_for_dataset(source_dataset_paired,feature_extractor,f"{save_dir}/ssot_source_features_paired.pkl")
```

Constructing the unpaired dataset.
```python
from datasets import datasets_factory
unpaired_dataset = datasets_factory.UnPairedDataset(datasets_factory.ConcatDatasets(source_dataset,source_dataset_paired),
                                                    datasets_factory.ConcatDatasets(target_dataset,target_dataset_paired),
                                                    f"{save_dir}/non_zero_dict.pkl"
                                                   )
```

*Finally, run the following command to train the score-based model.*
```bash
python main_ddpm.py --config {your config name}.yml --gpu_id 0,1,2,3
```

## Citation
```
@inproceedings{
Gu2023optimal,
title={Optimal Transport-Guided Conditional Score-Based Diffusion Model},
author={Gu, Xiang and Yang, Liwei and Sun, Jian and Xu, Zongben},
booktitle={NeurIPS},
year={2023}
}
```
## Contact
For any problem, please do not hesitate to contact xianggu@stu.xjtu.edu.cn.
