import math
import json
import torch
from log_utils import Logging, show_image, log_prediction
from transformer import Transformer
from attrdict import AttrDict
from torchvision.utils import make_grid
from torchvision import datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from feature_extractor import ResNet18

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb

from cgan import weights_init, DCGANGenerator, DCGANDiscriminator

# TODO Implement target shifting + masking

# TODO Input mask should zero attention where there is pad
# Target mask should do this and also the triu mask

def get_img_grids(img_seq):
    '''
    img_ seq = [seq_len, h, w] = [4, 16, 16]
    '''
    img_seq = img_seq.unsqueeze(1) # Restore Channel dim
    seq_len, c, h, w = img_seq.shape
    grid = torch.zeros(seq_len, c, h, w, device=img_seq.device)
    img_grids = []
    for i, img in enumerate(img_seq):
        # Place the image in the right quadrant
        grid[i] = img
        # Construct the grid from the batch of images
        img_grid = make_grid(grid.cpu(), nrow=2, padding=0)
        # Take the first channel (Grayscale images, the channels are all the same)
        img_grids.append(img_grid[0].unsqueeze(0).unsqueeze(0)) # Restore channel and batch dimensions
    return torch.cat(img_grids)

def get_targets(feature_extractor, images, n_channels=1):
    
    '''
    images shape: [seq_len, seq_batch, h, w]
    '''
    seq_len = images.shape[0]
    seq_bs = images.shape[1]

    # Iterate over sequences
    img_grids = [get_img_grids(images[:,i,:,:]) for i in range(images.shape[1])]

    tgt_images = torch.cat(img_grids, dim=1).to(images.device)
    tgt_images = tgt_images.view(seq_len * seq_bs, n_channels, tgt_images.shape[2], tgt_images.shape[3])
    with torch.no_grad():
        tgt_vectors = feature_extractor.get_vectors(tgt_images)
    tgt_vectors = tgt_vectors.view(seq_len, seq_bs, -1)
    return tgt_vectors, tgt_images

class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir="data"):
        super(MNISTDataModule, self).__init__()
        self.data_dir = data_dir

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        mnist = datasets.MNIST(download=False, train=True, root="data").data.float()
        self.transforms = T.Compose([ 
            T.Resize((args.data_loader.img_h, args.data_loader.img_w)), 
            T.ToTensor(), 
            T.Normalize((mnist.mean()/255,), (mnist.std()/255,))
        ])

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist = datasets.MNIST(self.data_dir, train=True, transform=self.transforms)
            self.train_dataset, self.val_dataset = random_split(mnist, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.MNIST(self.data_dir, train=False, transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=args.data_loader.batch_size, num_workers=0, shuffle=True, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=args.data_loader.batch_size, num_workers=0, shuffle=False, drop_last=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=args.data_loader.batch_size, num_workers=0, shuffle=False, drop_last=True, pin_memory=True)

class Sceneformer(pl.LightningModule):
    
    def __init__(self, feature_extractor, args, device):
        super(Sceneformer, self).__init__()
        self.dev = device
        self.feature_extractor = feature_extractor
        # Variables for logging
        self.train_step = 1
        self.val_step = 1
        self.args = args
        
        # Transformer
        image_size = (args.data_loader.img_w * int(math.sqrt(args.model.seq_len)), args.data_loader.img_h * int(math.sqrt(args.model.seq_len)))
        self.transformer = Transformer(
            feature_extractor,
            args.model.emb_size,
            args.model.seq_bs,
            args.model.seq_len,
            args.model.n_heads,
            args.model.ff_dim,
            args.model.n_layers,
            args.model.dropout,
            args.model.num_classes,
            image_size,
            self.dev)
        
        # self.disc = DCGANDiscriminator(image_channels=1).cuda()
        # self.disc.apply(weights_init)

        # self.init_weights()

    def forward(self, labels, targets, src_mask=None, tgt_mask=None):
        
        return self.transformer(labels, targets)

    def training_step(self, batch, batch_idx):#, optimizer_idx):

        # Data loading
        images, labels = batch
        images = images.view(args.model.seq_len, args.model.seq_bs, args.data_loader.img_h, args.data_loader.img_w)
        # Model forward
        step_loss, predictions, target_imgs = self.transformer_step(labels, images)
        # Logging
        self.logger.experiment.log({'Train t_loss with custom step': step_loss, 'train_step': self.train_step})
        self.train_step += 1
        if self.global_step % args.trainer.log_every_n_steps == 0:
            log_prediction(target_imgs, predictions, self.logger, title="Train Transformer Target and Ouptut")

        return step_loss
        #return {'loss': step_loss, 'preds': predictions}

    def training_epoch_end(self, training_step_outputs):
        pass
        # for pred in training_step_outputs:
        #     pass

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.view(args.model.seq_len, args.model.seq_bs, args.data_loader.img_h, args.data_loader.img_w)
        step_loss, predictions, target_imgs = self.transformer_step(labels, images)
        
        # Logging
        self.logger.experiment.log({'Val t_loss': step_loss, 'val_step': self.val_step})
        self.val_step += 1
        if self.global_step % args.trainer.log_every_n_steps == 0:
            log_prediction(target_imgs, predictions, self.logger, title="Validation Transformer Target and Ouptut")
        return step_loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.model.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        return [optimizer], [scheduler]

    def transformer_loss(self, outputs, targets):
        # loss = torch.nn.KLDivLoss()
        loss = torch.nn.MSELoss()
        return loss(outputs, targets)

    def transformer_step(self, labels, images):

        '''
        images shape:  [B,C,H,W]
        '''
        # Create target images
        targets, tgt_imgs = get_targets(self.feature_extractor, images)
        # Forward transformer
        predictions = self(labels.unsqueeze(1), targets)
        # Compute loss
        t_loss = self.transformer_loss(predictions, tgt_imgs)
        return t_loss, predictions, tgt_imgs

def main(args):

    assert args.model.emb_size % args.model.n_heads == 0, "Embedding size not divisible by number of heads"
    assert args.data_loader.batch_size == args.model.seq_len * args.model.seq_bs, "Batch size is not seq_len * transformer_batch"

    data_module = MNISTDataModule()
    wandb_logger = WandbLogger(project="MNIST Transformer")
    trainer = pl.Trainer(
        gpus=1,
        # fast_dev_run=True,
        # overfit_batches=0.01, # 1% of training set used as batch to make it overfit
        limit_train_batches=0.1, # 10% of training data
        # limit_val_batches=0.1, # 10% of validation data
        num_sanity_val_steps=2,
        flush_logs_every_n_steps=20,
        progress_bar_refresh_rate=20,
        #max_epochs=epochs,
        #profiler=True,
        logger=wandb_logger,
        callbacks=[
            Logging(),
        ]
    )

    wandb.login()

    hparams = dict(
        nhid = args.model.ff_dim, # the dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = args.model.n_layers, # the number of nn.TransformerEncoderLayer in nn.TransformerEncowandder
        nheads = args.model.n_heads, # the number of heads in the multiheadattention models
        dropout = args.model.dropout, # the dropout value
        lr = args.model.lr,
        emb_size = args.model.emb_size,
        img_h = args.data_loader.img_h,
        img_w = args.data_loader.img_w,
    )

    wandb.init(
        config=hparams,
        #mode="disabled"
    )

    wandb.run.name = "MSE Loss"

    config = wandb.config

    device = 'cuda:0' if args['n_gpu'] == 1 else 'cpu'
    
    feature_extractor = ResNet18(args.model.fe_weights_path).cuda()
    model = Sceneformer(feature_extractor, args, device)

    trainer.fit(model, data_module)
    
if __name__ == '__main__':
    with open("config.json") as c:
        args = AttrDict(json.load(c))

    main(args)