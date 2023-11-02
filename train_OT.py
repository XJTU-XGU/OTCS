import os
import argparse
from runners import OT_solver
from datasets import data_list,datasets_factory
import clip

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--ot_type",type=str, default="unsupervised", choices=["unsupervised","semi-supervised"],
                        help="type of OT")
    parser.add_argument("--lr", type = float,default=1e-6, help="learning rate")
    parser.add_argument("--epsilon", type = float, default=1e-7, help="regularization coeffecient")
    parser.add_argument("--gpu_id", type=str,default="0", help="gpu ids")
    parser.add_argument("--batch_size", type = int,default=64, help="gpu ids")
    parser.add_argument("--alpha", type = float, default=1.0, help="learning rate")
    parser.add_argument("--iterations", type = int, default=300000, help="gpu ids")
    parser.add_argument("--dataset", type=str, default="celeba", help="experiments")
    parser.add_argument("--save_dir", type=str, default="exp/OT/models", help="dir to save data")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    ''' Constructing the datasets and defining the network config.'''
    args.datasets_dict = {}
    args.network_dict = {}
    if args.dataset == "celeba":
        ''' This is for unsupervised OT. Each item of the dataset is like (image,label). '''
        source_dataset,target_dataset,_ = data_list.get_celeba_dataset("/home/sun_guxiang/dataset")
        args.datasets_dict["source"] = source_dataset
        args.datasets_dict["target"] = target_dataset

        args.network_dict["input_size"] = 64*64*3
        args.network_dict["num_hidden_layers"] = 5
        args.network_dict["dim_hidden_layers"] = 512
        args.network_dict["act_function"] = "SiLU"

        ot_solver = OT_solver.LargeScaleOTSolver(ot_type=args.ot_type)
        ''' Preloading images or extracting features. '''
        source_dataset = ot_solver.preloading_images_for_dataset(args.datasets_dict["source"],
                                                                 f"{args.save_dir}/celeba_source_images.pkl")
        target_dataset = ot_solver.preloading_images_for_dataset(args.datasets_dict["target"],
                                                                 f"{args.save_dir}/celeba_target_images.pkl")

        '''Feeding configs.'''
        ot_solver.feed_unsupervised_OT_params(cost="l1", epsilon=args.epsilon, **args.network_dict)
        '''Training potential networks.'''
        ot_solver.train(source_dataset, target_dataset, batch_size=args.batch_size, lr=args.lr,
                        num_train_steps=args.iterations, save_dir=args.save_dir)
        '''Computing and storing potential values.'''
        ot_solver.save_potentials(source_dataset, target_dataset, args.save_dir)
        '''Storing the dict of non-zero H.'''
        ot_solver.save_non_zero_dict(source_dataset, target_dataset, args.save_dir)

    elif args.dataset == "animal":
        ''' This is for semi-supervised OT. Each item of the dataset is like (image,label). '''
        encoder, preprocessing = clip.load("ViT-B/32")
        args.feature_extractor = encoder.cuda().encode_image
        source_dataset,target_dataset = data_list.get_animal_dataset("/data/guxiang/OT/data/animal_images/train",transform=preprocessing)
        source_dataset_paired,target_dataset_paired = data_list.get_animal_dataset_keypoints("/data/guxiang/OT/data/animal_images/train",transform=preprocessing)
        args.datasets_dict["source"] = source_dataset
        args.datasets_dict["target"] = target_dataset
        args.datasets_dict["source_paired"] = source_dataset_paired
        args.datasets_dict["target_paired"] = target_dataset_paired

        args.tau = 0.1
        args.network_dict["input_size"] = 512
        args.network_dict["num_hidden_layers"] = 2
        args.network_dict["dim_hidden_layers"] = 512
        args.network_dict["act_function"] = "SiLU"

        ot_solver = OT_solver.LargeScaleOTSolver(ot_type=args.ot_type)
        ''' Preloading images or extracting features. '''
        source_dataset = ot_solver.extracting_features_for_dataset(args.datasets_dict["source"], args.feature_extractor,
                                                                   f"{args.save_dir}/animal_source_features.pkl")
        target_dataset = ot_solver.extracting_features_for_dataset(args.datasets_dict["target"], args.feature_extractor,
                                                                   f"{args.save_dir}/animal_target_features.pkl")
        source_dataset_paired = ot_solver.extracting_features_for_dataset(args.datasets_dict["source_paired"],
                                                                          args.feature_extractor,
                                                                          f"{args.save_dir}/animal_source_paired_features.pkl")
        target_dataset_paired = ot_solver.extracting_features_for_dataset(args.datasets_dict["target_paired"],
                                                                          args.feature_extractor,
                                                                          f"{args.save_dir}/animal_target_paired_features.pkl")
        ''' Preparing paired dataset.'''
        paired_dataset = datasets_factory.PairedDataset(source_dataset_paired, target_dataset_paired)

        '''Feeding configs.'''
        ot_solver.feed_semi_supervised_OT_params(cost="cosine", epsilon=args.epsilon, alpha=args.alpha, tau=args.tau,
                                                 **args.network_dict)
        '''Training potential networks.'''
        ot_solver.train(source_dataset, target_dataset, paired_dataset, batch_size=args.batch_size, lr=args.lr,
                        num_train_steps=args.iterations, save_dir=args.save_dir)
        '''Concat unpaired and paired datasets.'''
        source_dataset_concated = datasets_factory.ConcatDatasets(source_dataset, source_dataset_paired)
        target_dataset_concated = datasets_factory.ConcatDatasets(target_dataset, target_dataset_paired)
        '''Computing and storing potential values.'''
        ot_solver.save_potentials(source_dataset_concated, target_dataset_concated, save_dir=args.save_dir)
        '''Storing the dict of non-zero H.'''
        ot_solver.save_non_zero_dict(source_dataset_concated, target_dataset_concated, paired_dataset,
                                     save_dir=args.save_dir)

    else:
        raise Exception("Please construct you own datasets and define the code for your task by referring to the celeba and animal datasets.")


