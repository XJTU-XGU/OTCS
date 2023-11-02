import torch
from functions import models
import itertools
import os
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
import tqdm

class LargeScaleOTSolver(object):
    def __init__(self,ot_type="unsupervised"):
        '''Three type of OTs are feasible: unsupervised, semi-supervised.
        For unsupervised, the objective is $min_{\pi}E_{\pi}[c(x,y)]$.
        For semi-supervised, the objective is $min_{\pi}E_{\pi}[\alpha*c(x,y)+(1-\alpha)*g(x,y)]$. Note that in the paper,
        the semi-supervised OT is the case $\alpha = 0$.
        :param ot_type: choices could be unsupervised, semi-supervised.
        '''
        super(LargeScaleOTSolver,self).__init__()
        self.param_fed = False
        self.ot_type = ot_type
        if ot_type not in ["unsupervised", "semi-supervised"]:
            raise Exception("ot_type must be 'unsupervised' or 'semi-supervised'.")

    def feed_unsupervised_OT_params(self,cost="l2",epsilon=1e-6,input_size=64*64,num_hidden_layers=3,
                                    dim_hidden_layers=512,act_function = "silu"):
        '''
        :param cost: "l2" is the squred l2-distance. "cosine" is the cosine distance. You can also define your own cost
        function, which takes two batches of Tensors X (N,..),Y (N,..) and returns a N*N tensor with each entry being
        the distance of a pair from X and Y.
        :param epsilon: The regularization factor.
        :param input_size: The dimension of the input layer of potentials.
        :param num_hidden_layers: The number of hidden layers of potentials.
        :param dim_hidden_layers: The dimension of the hidden layers of potentials.
        :param act_function: activation function of potentials. Choices could be "relu", "silu", "leaky_relu", "tanh".
        '''
        self.cost = cost
        self.epsilon = epsilon
        self.potential_dict = {"input_size":input_size,"num_hidden_layers":num_hidden_layers,
                               "dim_hidden_layers":dim_hidden_layers,"act_fun":act_function}

        self.param_fed = True

        ###### building networks
        self.u_net = models.Potential(self.potential_dict).cuda()
        self.v_net = models.Potential(self.potential_dict).cuda()

    def feed_semi_supervised_OT_params(self, cost="l2", epsilon=1e-6, alpha = 0, tau = 0.1, tau_= None,
                                       input_size=64 * 64, num_hidden_layers=3,
                                       dim_hidden_layers=512, act_function="silu"):
        '''
        :param cost: "l2"/"l1"/"cosine" are respectively the mean squred l2-distance / mean l1-distance/ cosine distance.
         You can also define your own cost function, which takes two batches of Tensors X (N,..),Y (N,..) and returns a
         N*N tensor with each entry being the distance of a pair from X and Y.
        :param epsilon: The regularization factor.
        :param alpha: The combination coefficient.
        :param tau: The temperature of source domain.
        :param tau_: The temperature of target domain. If None, tau_ = tau.
        :param input_size: The dimension of the input layer of potentials.
        :param num_hidden_layers: The number of hidden layers of potentials.
        :param dim_hidden_layers: The dimension of the hidden layers of potentials.
        :param act_function: activation function of potentials. Choices could be "ReLU", "SiLU", "Leaky_ReLU", "Tanh".
        '''
        if tau_ is None:
            tau_ = tau
        self.cost = cost
        self.epsilon = epsilon
        self.tau_ = tau_
        self.tau = tau
        self.alpha = alpha
        self.potential_dict = {"input_size": input_size, "num_hidden_layers": num_hidden_layers,
                               "dim_hidden_layers": dim_hidden_layers, "act_fun": act_function}

        self.param_fed = True

        ###### building networks
        self.u_net = models.Potential(self.potential_dict).cuda()
        self.v_net = models.Potential(self.potential_dict).cuda()

    def computeGuidingMatrix(self,Xs_batch,Xt_batch,xs_paired,xt_paired):
        '''
        Computing the guiding function for each pair of tensors in Xs_batch,Xt_batch
        :param Xs_batch: Tensor (N,d)
        :param Xt_batch: Tensor (N',d)
        :param xs_paired: source paired data / keypoints, Tensor (K,d)
        :param xt_paired: target paired data / keypoints, Tensor (K,d)
        :return: Tensor (N,N')
        '''
        with torch.no_grad():
            Cs = self.cost_matrix(Xs_batch, xs_paired)
            Ct = self.cost_matrix(Xt_batch, xt_paired)
            Rs = torch.nn.functional.softmax(-Cs/self.tau,dim=1)
            Rt = torch.nn.functional.softmax(-Ct/self.tau_,dim=1)
            G = self.JS_matrix(Rs,Rt)
        return G.float()

    def KL_matrix(self, p, q, eps=1e-10):
        '''
        Computing KL divergence.
        '''
        return torch.sum(p * torch.log(p + eps) - p * torch.log(q + eps), dim=-1)

    def JS_matrix(self,P, Q, eps=1e-10):
        '''
        Computing JS-divergence matrix
        :param P: Tensor (N,K)
        :param Q: Tensor (N',K)
        :return: Tensor (N,N')
        '''
        P = P.unsqueeze(1)
        Q = Q.unsqueeze(0)
        kl1 = self.KL_matrix(P, (P + Q) / 2, eps)
        kl2 = self.KL_matrix(Q, (P + Q) / 2, eps)
        return 0.5 * (kl1 + kl2)

    def cost_matrix(self,x, y):
        '''
        Computing the cost matrix.
        :param x: Tensor (N,d)
        :param y: Tensor (N',d)
        :return: Tensor (N,N')
        '''
        if self.cost == "l2":
            cost = torch.mean(torch.abs(x.unsqueeze(1) - y.unsqueeze(0)) ** 2, dim=-1)
        elif self.cost == "l1":
            cost = torch.mean(torch.abs(x.unsqueeze(1) - y.unsqueeze(0)) ** 1, dim=-1)
        elif self.cost == "cosine":
            x = torch.nn.functional.normalize(x, dim=1)
            y = torch.nn.functional.normalize(y, dim=1)
            cost = 1.0 - torch.matmul(x, y.t())
        else:
            if callable(self.cost):
                cost = self.cost(x,y)
            else:
                raise Exception("The cost is illegal.")
        return cost

    def generate_batch_mask(self,idx_sk,idx_tk,batchsize):
        '''
        Generating the mask matrix for a batch.
        :param idx_sk: indexes of the source paired data in the batch
        :param idx_tk: indexes of the target paired data in the batch
        :param batchsize: The batch size
        :return: Mask matrix (batchsize, batchsize)
        '''
        M = torch.ones(batchsize,batchsize).cuda()
        M[idx_sk,:] = 0.0
        M[:,idx_tk] = 0.0
        M[idx_sk,idx_tk] = 1.0
        return M

    def dual_semi_supervised_OT_loss(self, u_batch, v_batch, G, M):
        '''
        The loss function of the dual OT.
        :param u_batch: Output of u_net.
        :param v_batch: Output of v_net.
        :param G: Guiding matrix.
        :param M: Mask matrix.
        '''
        V = torch.reshape(u_batch, (-1, 1))+ torch.reshape(v_batch, (1, -1)) - G
        tmp = torch.max(torch.zeros(1).to(G.device), V)
        loss_batch = torch.mean(u_batch) + torch.mean(v_batch) \
                         - (1. / (4. * self.epsilon)) * torch.mean(M * tmp**2)
        return -loss_batch

    def dual_unsupervised_OT_loss(self, u_batch, v_batch, C):
        '''
        The loss function of dual OT.
        :param u_batch: Output of u_net.
        :param v_batch: Output of v_net.
        :param C: Cost matrix.
        '''
        V = torch.reshape(u_batch, (-1, 1))+ torch.reshape(v_batch, (1, -1)) - C
        tmp = torch.max(torch.zeros(1).to(C.device), V)
        loss_batch = torch.mean(u_batch) + torch.mean(v_batch) \
                         - (1. / (4. * self.epsilon)) * torch.mean( tmp**2)
        return -loss_batch

    def ckeck_params_loaded(self):
        if not self.param_fed:
            if self.ot_type == "unsupervised":
                print("Please feed the unsupervised OT parameters by function 'feed_unsupervised_OT_params'.")
            elif self.ot_type == "semi-supervised":
                print("Please feed the unsupervised OT parameters by function 'feed_semi_supervised_OT_params'.")


    def train(self,source_dataset,target_dataset,paired_dataset = None, batch_size=64,num_workers=0,
              lr=1e-6,num_train_steps=100000,save_interval=10000,save_dir = "exp/OT/models"):
        '''
        Training the potentials u_net, v_net.
        '''
        self.ckeck_params_loaded()
        ###### building dataloader
        source_loader = DataLoader(source_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,
                                   drop_last=True)
        target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                   drop_last=True)
        if paired_dataset is None:
            if self.ot_type == "semi-supervised":
                raise Exception("Semi-supervised OT requires paired data. Otherwise, please choose unsupervised OT.")
        else:
            paired_loader = DataLoader(paired_dataset, batch_size=batch_size, num_workers=num_workers,
                                       drop_last=False)

        trainable_params = itertools.chain(self.u_net.parameters(), self.v_net.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        tqdm_range = tqdm.trange(num_train_steps)
        loss_cum = 0
        for step in tqdm_range:
            try:
                xs,_ = source_iter.next()
            except:
                source_iter = iter(source_loader)
                xs,_ = source_iter.next()
            try:
                xt,_ = target_iter.next()
            except:
                target_iter = iter(target_loader)
                xt,_ = target_iter.next()
            if paired_dataset is not None:
                try:
                    xs_k,xt_k = paired_iter.next()
                except:
                    paired_iter = iter(paired_loader)
                    xs_k,xt_k = paired_iter.next()
                    xs_k,xt_k = xs_k.cuda(),xt_k.cuda()
                    idx_keypoints = np.arange(len(xs_k)).tolist()

            xs,xt,= xs.cuda(),xt.cuda()
            xs,xt = xs.view(batch_size,-1),xt.view(batch_size,-1)

            if self.ot_type == "semi-supervised":
                xs = torch.cat((xs_k,xs),dim=0)
                xt = torch.cat((xt_k,xt),dim=0)
                G = self.computeGuidingMatrix(xs,xt,xs_k,xt_k)
                C = self.cost_matrix(xs,xt)
                G_ = (1-self.alpha)*C + self.alpha * G
                M = self.generate_batch_mask(idx_keypoints,idx_keypoints,len(xs))
                u = self.u_net(xs)
                v = self.v_net(xt)
                loss = self.dual_semi_supervised_OT_loss(u,v,G_,M)
            else:
                C = self.cost_matrix(xs, xt)
                u = self.u_net(xs)
                v = self.v_net(xt)
                loss = self.dual_unsupervised_OT_loss(u,v,C)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_cum += loss.item()
            tqdm_range.set_description("step:{:d}/{:d}\tloss:{:.12f}".format(step,num_train_steps,loss_cum/(step+1)))

            if step%save_interval == 0:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save({"u_net":self.u_net.state_dict(),
                            "v_net":self.v_net.state_dict()},
                           f"{save_dir}/potential_net_dict.pkl")

    def save_potentials(self,source_dataset,target_dataset,save_dir="exp/OT/models"):
        '''
        Computing and saving the outputs of potential networks on source and target data.
        :param source_dataset: source dataset.
        :param target_dataset: target dataset.
        :param save_dir: directory to save potential values.
        '''
        if not os.path.exists(f"{save_dir}/potential_net_dict.pkl"):
            raise Exception("Please train the potentials first!!!")
        stat_dict = torch.load(f"{save_dir}/potential_net_dict.pkl")
        self.u_net.load_state_dict(stat_dict["u_net"])
        self.v_net.load_state_dict(stat_dict["v_net"])

        source_potentials = self.compute_potentials(source_dataset, self.u_net)
        target_potentials = self.compute_potentials(target_dataset, self.v_net)
        torch.save({"source_potential": source_potentials,
                    "target_potential": target_potentials,
                    "epsilon": self.epsilon}, f"{save_dir}/potential_values.pkl")


    def compute_potentials(self,dataset,net):
        '''
        Computing the potential values.
        :param dataset:
        :param net: u_net or v_net
        :return: computed potential values.
        '''
        loader = torch.utils.data.DataLoader(dataset,batch_size=256)
        P = []
        with torch.no_grad():
            for x,_ in tqdm.tqdm(loader):
                x = x.cuda().view(len(x),-1)
                P.append(net(x).cpu())
            P = torch.cat(P,dim=0)
        return P

    def save_non_zero_dict(self,source_dataset,target_dataset,paired_dataset = None,save_dir="exp/OT/models"):
        '''
        Saving the dict of non-zero H.
        :return: Dict, like {i: (indexes, H values)}
        '''
        if not os.path.exists(f"{save_dir}/potential_values.pkl"):
            raise Exception("Please compute and store the potential values first!!!")
        if self.ot_type == "semi-supervised":
            print("loading paired data...")
            source_keypoints = []
            target_keypoints = []
            for i in tqdm.trange(len(paired_dataset)):
                source_keypoints.append(paired_dataset[i][0].unsqueeze(0))
                target_keypoints.append(paired_dataset[i][1].unsqueeze(0))
            source_keypoints = torch.cat(source_keypoints,dim=0).cuda()
            target_keypoints = torch.cat(target_keypoints,dim=0).cuda()

        stat_dict = torch.load(f"{save_dir}/potential_values.pkl")
        u_batch = stat_dict["source_potential"].cuda()
        v_batch = stat_dict["target_potential"].cuda()
        print("loading data...")
        xt = [x.unsqueeze(0) for x,_ in target_dataset]
        xt = torch.cat(xt,dim=0).cuda()
        xs = [x.unsqueeze(0) for x, _ in source_dataset]
        xs = torch.cat(xs, dim=0).cuda()
        xs,xt = xs.view(len(xs),-1),xt.view(len(xt),-1)
        print("saving dict....")
        dicts = {}
        for i in tqdm.trange(len(source_dataset)):
            x = xs[[i]]
            u = u_batch[i]
            if self.ot_type == "semi-supervised":
                G = self.computeGuidingMatrix(x, xt,source_keypoints,target_keypoints)
                C = self.cost_matrix(x,xt)
                G = (1 - self.alpha) * C + self.alpha * G
            else:
                G = self.cost_matrix(x,xt)
            V = torch.reshape(u, (-1, 1)) + torch.reshape(v_batch, (1, -1)) - G
            h = (1. / (2. * self.epsilon)) * torch.max(torch.zeros(1).to(V.device), V)
            h = h.view(-1,).cpu()
            index = (h>0.1).nonzero(as_tuple=False).reshape(-1,)
            h_ = h[index]
            dicts[i] = (index.numpy(),h_.numpy())

        torch.save(dicts,f"{save_dir}/non_zero_dict.pkl")

    def preloading_images_for_dataset(self,dataset, save_file=None):
        '''
        To save time in training potential, we can load data first.
        :param dataset: original dataset
        :param save_file: save loaded images
        :return: TosorDataset
        '''

        save_dir,file_name = os.path.split(save_file)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(save_file):
            print("loading images...")
            imags = []
            labels = []
            for i in tqdm.trange(len(dataset)):
                imags.append(dataset[i][0].unsqueeze(0))
                labels.append(dataset[i][1].unsqueeze(0))
            imags = torch.cat(imags, dim=0)
            labels = torch.cat(labels, dim=0)
            if save_file is not None:
                torch.save({"images": imags, "labels": labels}, save_file)
        else:
            print("reloading images...")
            data = torch.load(save_file)
            imags,labels = data["images"],data["labels"]

        dataset = TensorDataset(imags, labels)
        return dataset

    def extracting_features_for_dataset(self,dataset,feature_extractor,save_file=None, batch_size=256):
        '''
        Extracting features of the images and construct the feature dataset
        :param feature_extractor: pretrained feature extractor
        :param save_file: path to save features
        :return: dataset of features
        '''
        save_dir, file_name = os.path.split(save_file)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(save_file):
            data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
            print("extracting features...")
            Features = []
            Labels = []
            with torch.no_grad():
                for x, y in tqdm.tqdm(data_loader):
                    Features.append(feature_extractor(x.cuda()).float().cpu())
                    Labels.append(y)
            Features = torch.cat(Features, dim=0).float()
            Labels = torch.cat(Labels, dim=0)
            torch.save({"features": Features, "labels": Labels}, save_file)
        else:
            print("loading stored features...")
            data = torch.load(save_file)
            Features, Labels = data["features"], data["labels"]
        dataset = TensorDataset(Features, Labels)
        return dataset

