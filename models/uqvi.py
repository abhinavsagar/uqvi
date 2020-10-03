import torch
import torch.nn as nn
import torch.distributions as dist
import pytorch_lightning as pl

import os
import nn.loss as losses
import nn.metrics as metr
import nn.bayes_conv as bayes_conv


def sample_gaussian(mu, logsigma):
    sigma = torch.exp(0.5 * logsigma)
    eps = sigma.data.new(sigma.size()).normal_()
    return eps.mul(sigma) + mu


class BaseModel(pl.LightningModule):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.hparams = args

        self.train_dset, self.test_dset, args = utils.load_dataset(self.hparams)
        self.net, self.hparams = models.init_net(self.hparams)

        self.batch_size = self.hparams.batch_size
        self.lr = self.hparams.lr

        if self.hparams.task == 'clf':
            self.loss_fun = losses.cr_ent_ll
            self.metrics = {'accuracy': metr.accuracy}
        elif self.hparams.task == 'seg':
            self.loss_fun = losses.unet_binary_ll
            self.metrics = {'IOU':metr.iou, 'DICE':metr.dice}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dset, self.batch_size,
                                           num_workers=4, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dset, self.batch_size,
                                           num_workers=4, shuffle=False)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.hparams.lr_step_freq,
                                                    gamma=0.5)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        X, y = batch
        out = self(X)
        loss = self.loss_fun(out, y)['ll']
        if self.hparams.l2 > 0:
            l2_norm = torch.sum(
                torch.stack([torch.sum(p ** 2) for p in self.net.net.parameters()]))

        loss = loss + self.hparams.l2*l2_norm

        logs = {'train_loss':loss, 'l2':l2_norm}

        _, y_pred = torch.max(out.detach(), 1)
        for m in self.metrics:
            logs[m] = self.metrics[m](y.cpu().numpy(),
                                      y_pred.cpu().numpy())
        return {'loss':loss, 'log':logs}

    def validation_step(self, batch, batch_idx):
        X, y = batch
        out = self(X)
        loss = self.loss_fun(out, y)['ll']

        logs = {'val_loss': loss}
        _, y_pred = torch.max(out.detach(), 1)
        for m in self.metrics:
            logs['val_' + m] = self.metrics[m](y.cpu().numpy(), y_pred.cpu().numpy())
        return logs

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        avg_logs = {'val_loss': avg_loss}
        for m in self.metrics:
            avg_logs['val_' + m] = torch.stack([torch.tensor(x['val_' + m]) for x in outputs]).mean()
        avg_logs['log'] = avg_logs.copy()
        return avg_logs


class BayesNet(pl.LightningModule):
    def __init__(self, args):
        super(BayesNet, self).__init__()
        self.hparams = args

        self.train_dset, self.test_dset, args = utils.load_dataset(self.hparams)
        self.net, self.hparams = models.init_net(self.hparams)
        self.batch_size = self.hparams.batch_size
        self.lr = self.hparams.lr

        if self.hparams.task == 'clf':
            ll = losses.cr_ent_ll
            self.metrics = {'accuracy': metr.accuracy}
        elif self.hparams.task == 'seg':
            ll = losses.unet_binary_ll
            self.metrics = {'IOU': metr.iou, 'DICE': metr.dice}

        if self.hparams.dwp:
            self.loss_fun = losses.DWPLoss(log_lik=ll,
                                           anneal=None, N=len(self.train_dset))
            self.prior = 'dwp'
            self.prior_dataset = args.prior
            if self.hparams.kernel_dimention == 2:
                self.vae_class = models.kernel_vae.KernelVAE
            elif self.hparams.kernel_dimention == 3:
                self.vae_class = models.kernel_vae.KernelVAE3D
            self.vaes, self.layer_names = self.load_vae()
            print(self.layer_names)
        else:
            self.loss_fun = losses.VarDropoutLoss(log_lik=ll, anneal=None,
                                                  N=len(self.train_dset))
            self.prior = 'normal'


    def train_dataloader(self):
        loader =  torch.utils.data.DataLoader(self.train_dset, self.batch_size,
                                           num_workers=8, shuffle=True)
        return loader

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dset, self.batch_size,
                                           num_workers=4, shuffle=False)

    def forward(self, x):
        return self.net(x)

    def load_vae(self):
        root = os.path.join('runs/', self.prior_dataset, 'prior')
        mods = os.listdir(root)
        v = []
        names = []
        for m in mods:
            w = os.path.join(root, m, 'lightning_logs/version_0/checkpoints/last.ckpt')
            par = os.path.join(root, m, 'lightning_logs/version_0/hparams.yaml')
            curr_vae = self.vae_class.load_from_checkpoint(checkpoint_path=w,
                                                           hparams_file=par)
            curr_vae.freeze()
            v.append(curr_vae)
            names.append(curr_vae.layer_name)
        return nn.ModuleList(v), names

    def compute_kl(self):
        kl_div = []
        for name, m in self.net.named_modules():
            if isinstance(m, bayes_conv._BayesConvNd):
                kl = self.kl(m, name)
                kl_div.append(kl)

        kl = sum(kl_div)
        return kl

    def kl(self, m, name):
        name = 'net.' + name + '.weight'

        ker_dim = torch.Size([-1, 1]) + m.mu_weight.shape[2:]
        w_mu = m.mu_weight.reshape(ker_dim)
        w_logsigma = m.logsigma_weight.reshape(ker_dim)

        if self.prior == 'normal':
            kl = self.loss_fun.approx_KL(w_mu, w_logsigma)
        elif self.prior == 'dwp':
            for i in range(len(self.vaes)):
                if name in self.layer_names[i]:
                    curr_vae = self.vaes[i]
            w = sample_gaussian(w_mu, w_logsigma)
            with torch.no_grad():
                rec_w_mu, rec_w_logsigma, z_mu, z_logsigma, z = curr_vae(w)
            kl = self.loss_fun.approx_KL(w, w_mu, w_logsigma, z, z_mu, z_logsigma,
                                         rec_w_mu, rec_w_logsigma)
        return kl

    def configure_optimizers(self):
        if self.hparams.dwp:
            for p in self.vaes.named_parameters():
                p[1].requires_grad = False
            print([p[0] for p in self.named_parameters() if p[1].requires_grad])
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                     lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.hparams.lr_step_freq,
                                                    gamma=0.5)
        return [optimizer], [scheduler]


    def training_step(self, batch, batch_idx):
        kl = self.compute_kl()
        X, y = batch
        out = self(X)
        loss, logs = self.loss_fun(out, y, kl)

        logs['train_loss'] = loss.item()
        _, y_pred = torch.max(out.detach(), 1)
        for m in self.metrics:
            logs[m] = self.metrics[m](y.cpu().numpy(),
                                      y_pred.cpu().numpy())
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        kl = self.compute_kl()
        X, y = batch
        out = self(X)
        loss, logs = self.loss_fun(out, y, kl)

        logs['val_loss'] = loss.data.cpu()
        _, y_pred = torch.max(out.detach(), 1)
        for m in self.metrics:
            logs['val_' + m] = self.metrics[m](y.cpu().numpy(), y_pred.cpu().numpy())
        return logs

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        avg_logs = {'val_loss': avg_loss}
        keys = outputs[0].keys()
        for k in keys:
            if k != 'val_loss':
                k_val = k
                if 'val_' not in k:
                    k_val = 'val_' + k
                avg_logs[k_val] = torch.stack([torch.tensor(x[k]).float() for x in outputs]).mean()
        avg_logs['log'] = avg_logs.copy()
        return avg_logs
