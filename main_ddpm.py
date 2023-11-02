import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
from runners.ddpm import Diffusion
from datasets import data_list,datasets_factory
from runners import OT_solver

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument( "--config", type=str, default="celeba.yml", help="Path to the config file")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--exp", type=str, default="exp/score", help="Path for saving running related data.")
    parser.add_argument("--doc",type=str,default="log",help="A string for documentation purpose,"
                                                            "Will be the name of the log folder.")
    parser.add_argument("--comment", type=str, default="", help="A string for experiment comment" )
    parser.add_argument("--verbose",type=str,default="info",help="Verbose level: info | debug | warning | critical",)
    parser.add_argument("--sample", type= bool,default=False,help="Whether to produce samples from the model")
    parser.add_argument( "--resume_training", default=False,action="store_true", help="Whether to resume training")
    parser.add_argument("-i","--image_folder",type=str,default="images3",help="The folder name of samples",)
    parser.add_argument("--use_pretrained_model", default=True)
    parser.add_argument("--gpu_id",type=str, default="0,1")


    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    if not args.sample:
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = False

                response = input("Folder already exists. Overwrite? (Y/N)")
                if response.upper() == "Y":
                    overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    shutil.rmtree(tb_path)
                    os.makedirs(args.log_path)
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
                else:
                    print("Folder exists. Program go on.")
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(new_config, f, default_flow_style=False)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        if args.sample:
            os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
            args.image_folder = os.path.join(
                args.exp, "image_samples", args.image_folder
            )
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                overwrite = True

                if overwrite:
                    shutil.rmtree(args.image_folder)
                    os.makedirs(args.image_folder)
                else:
                    print("Output image folder exists. Program halted.")
                    sys.exit(0)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    '''
    Preparing the datasets. 
    '''
    if config.dataset == "celeba":
        '''Using unsupervised OT in image space.'''
        source_dataset,target_datset,source_dataset_test = data_list.get_celeba_dataset(config.data.data_dir)
        unpaired_dataset = datasets_factory.UnPairedDataset(source_dataset,target_datset,"exp/OT/models/non_zero_dict.pkl")
        sampling_dataset = source_dataset_test

    elif config.dataset == "animal":
        '''Using semi-supervised OT in feature space.'''
        source_dataset, target_dataset = data_list.get_animal_dataset(config.data.data_dir)
        source_dataset_paired, target_dataset_paired = data_list.get_animal_dataset_keypoints(config.data.data_dir)
        ot_solver = OT_solver.LargeScaleOTSolver(ot_type="semi-supervised")

        '''Obtaining source feature datasets.'''
        source_feature_dataset = ot_solver.extracting_features_for_dataset(source_dataset, None,
                                                                   f"exp/OT/models/animal_source_features.pkl")
        source_feature_dataset_paired = ot_solver.extracting_features_for_dataset(source_dataset_paired,
                                                                          None,
                                                                          f"exp/OT/models/animal_source_paired_features.pkl")

        '''The unpaired training dataset contains source features vs target images.'''
        unpaired_dataset = datasets_factory.UnPairedDataset(datasets_factory.ConcatDatasets(source_feature_dataset,
                                                                                            source_feature_dataset_paired),
                                                            datasets_factory.ConcatDatasets(target_dataset,
                                                                                            target_dataset_paired),
                                                            "exp/OT/models/non_zero_dict.pkl"
                                                            )
        '''The sampling dataset contains source images vs source features.'''
        sampling_dataset = datasets_factory.PairedDataset(source_dataset,source_feature_dataset)

    else:
        '''For you own task/datasets, please prepare datasets by refering to the celeba and animal. Also, you can
         modify the config file (like celeba.yml) to define your model and configs.'''
        NotImplementedError

    trained_model_path = os.path.join(args.log_path, "ckpt.pth")
    try:
        runner = Diffusion(args, config)
        if args.sample:
            runner.sample(sampling_dataset,trained_model_path,bs=2)
        else:
            runner.train(unpaired_dataset)
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
