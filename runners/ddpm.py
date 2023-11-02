import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from functions.models import EMAHelper
from functions.script_util import create_model
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.models import Model
import torchvision.utils as tvu


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self,unpaired_dataset):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset = unpaired_dataset
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        if config.dataset == "celeba":
            model = Model(config)
            states = torch.load(self.config.model.pretrained_dict, map_location=self.device)
            states[-1].pop("conv_in.weight")
            model.load_state_dict(states[-1],strict=False)
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)

        elif config.dataset == "animal":
            model = create_model(image_size=config.data.image_size,
                                 num_class=config.model.num_class,
                                 num_channels=config.model.num_channels,
                                 num_res_blocks=config.model.num_res_blocks,
                                 learn_sigma=config.model.learn_sigma,
                                 class_cond=config.model.class_cond,
                                 attention_resolutions=config.model.attention_resolutions,
                                 num_heads=config.model.num_heads,
                                 num_head_channels=config.model.num_head_channels,
                                 num_heads_upsample=config.model.num_heads_upsample,
                                 use_scale_shift_norm=config.model.use_scale_shift_norm,
                                 dropout=config.model.dropout,
                                 resblock_updown=config.model.resblock_updown,
                                 use_fp16=config.model.use_fp16,
                                 use_new_attention_order=config.model.use_new_attention_order)
            states = torch.load(self.config.model.pretrained_dict, map_location=self.device)
            model.load_state_dict(states, strict=False)
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)

        else:
            '''
            You can define you own score-based model (Unet) by modifying the file of "functions/models". 
            '''
            model = Model(config)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (cond,x) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                cond = cond.to(self.device)
                x = data_transform(self.config, x)
                cond = data_transform(self.config,cond)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, cond, e, b, dataset = config.dataset)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

    def sample(self,source_dataset,trained_model_path,bs=4):
        config = self.config
        if config.dataset == "celeba":
            model = Model(config)
        elif config.dataset == "animal":
            model = create_model(image_size=config.data.image_size,
                                 num_class=config.model.num_class,
                                 num_channels=config.model.num_channels,
                                 num_res_blocks=config.model.num_res_blocks,
                                 learn_sigma=config.model.learn_sigma,
                                 class_cond=config.model.class_cond,
                                 attention_resolutions=config.model.attention_resolutions,
                                 num_heads=config.model.num_heads,
                                 num_head_channels=config.model.num_head_channels,
                                 num_heads_upsample=config.model.num_heads_upsample,
                                 use_scale_shift_norm=config.model.use_scale_shift_norm,
                                 dropout=config.model.dropout,
                                 resblock_updown=config.model.resblock_updown,
                                 use_fp16=config.model.use_fp16,
                                 use_new_attention_order=config.model.use_new_attention_order)
        else:
            '''
            You can define you own score-based model (Unet) by modifying the file of "functions/models". 
            '''
            model = Model(config)
        states = torch.load(trained_model_path)
        model.load_state_dict(states[4])
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        model.eval()
        self.sample_transported_data(model,source_dataset,bs)


    def sample_transported_data(self,model,source_dataset,bs):
        config = self.config
        n = bs
        loader = torch.utils.data.DataLoader(source_dataset,batch_size=n,shuffle=True)
        data_iter = iter(loader)
        if self.config.dataset == "celeba":
            ''' Example for conditioned on source image.
            '''
            img_source,_ = data_iter.__next__()
            x = torch.randn(
                n,
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )
            img_source = data_transform(self.config,img_source).cuda()
            with torch.no_grad():
                x = self.sample_image(x, img_source, model)

            x = inverse_data_transform(config, x)
            img_source = inverse_data_transform(config, img_source.cpu())
            x = torch.cat((img_source, x), dim=-1)

            for i in range(len(x)):
                tvu.save_image(x[i], os.path.join(self.args.image_folder, f"s2t_{i}.png"))

        elif self.config.dataset == "animal":
            ''' Example for conditioned on source features.'''
            img_source, feat_source = data_iter.__next__()
            x = torch.randn(
                n,
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            feat_source = data_transform(self.config, feat_source)
            with torch.no_grad():
                x = self.sample_image(x, feat_source, model)

            x = inverse_data_transform(config, x)
            x = torch.cat((img_source.cpu(),x), dim=-1)

            for i in range(len(x)):
                tvu.save_image(x[i], os.path.join(self.args.image_folder, f"s2t_{i}.png"))

    def sample_image(self, x, cond, model, last=True):
        seq = range(0, self.num_timesteps, 1)

        from functions.denoising import ddpm_steps_conditioned_on_features,ddpm_steps_conditioned_on_images
        if self.config.dataset == "celeba":
            x = ddpm_steps_conditioned_on_images(x,seq,model,cond, self.betas,source_init=True,init_time=0.2)
        elif self.config.dataset == "animal":
            x = ddpm_steps_conditioned_on_features(x, seq, model, cond, self.betas)

        # if last:
        #     x = x[0][-1]
        return x

    def test(self):
        pass


