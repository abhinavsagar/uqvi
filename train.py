import argparse
import torch
import utils
import models
import os
import pytorch_lightning as pl

parser = argparse.ArgumentParser()

parser.add_argument('--root', type=str, default='./data', help='root folder')
parser.add_argument('--dataset_name', type=str, default='MNIST', help='name of the dataset')
parser.add_argument('--resume', action='store_true', default=False, help='Continue training')

parser.add_argument('--test_size', type=int, default=-1,
                    help='Number of images in test, -1 stands for the full dataset')
parser.add_argument('--train_size', type=int, default=-1,
                    help='Number of images in train, -1 stands for the full dataset')
parser.add_argument('--kernel_dimention', default=2, type=int)

parser.add_argument('--batch_size', type=int, default=500, help='input batch size for training (default: 5)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
parser.add_argument('--patience', type=float, default=15, help='Patients for lr scheduler')
parser.add_argument('--lr_step_freq', type=float, default=50, help='How often to reduce lr')
parser.add_argument('--l2', type=float, default=1e-3, help='Weight of the L2 norm')

parser.add_argument('--pretrain', action='store_true', default=False, help='Use shorter version of the model')
parser.add_argument('--freeze', action='store_true', default=False, help='Freeze Layers in the middle')

parser.add_argument('--dwp', action='store_true', default=False, help='Train with dwp')
parser.add_argument('--prior', type=str, default=None, help='prior: BRATS or MS')
parser.add_argument('--bayes', action='store_true', default=False, help='Train bayes net')

parser.add_argument('--device', type=str, default='cuda:0', help='enables CUDA training')

parser.add_argument('--short', action='store_true', default=False, help='Use shorter version of the model')
parser.add_argument('--f', type=int, default=32, help='Floating point precision')
parser.add_argument('--data_type', type=str, default=None,
                    help='MRI type, if applicable. If dataset contains only one modality, it is ignored')


def main(args):
    if args.dwp or args.bayes:
        args.bayes = True
        mod = models.dwp.BayesNet(args)
    else:
        mod = models.dwp.BaseModel(args)

    args = utils.create_model_name(args)
    print('Model name:', args.model_name)

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.00,
        patience=args.patience,
        verbose=True,
        mode='max',
        strict=False
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_last=True
    )

    trainer = pl.Trainer(gpus=[0], show_progress_bar=True,
                         default_root_dir=os.path.join('runs', args.model_name),
                         early_stop_callback=early_stop_callback,
                         precision=args.f, terminate_on_nan=True,
                         checkpoint_callback=checkpoint_callback, max_epochs=10000,
                         check_val_every_n_epoch=args.patience//10)
    trainer.fit(mod)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)