import math
import torch
from log_utils import Logging, show_image, log_prediction
from transformer import Transformer
import json

from torchvision import datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger# TensorBoardLogger
#from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler

import wandb

from cgan import weights_init, DCGANGenerator, DCGANDiscriminator

w_path = 'ckpt/resnet18_mnist.pt'
emb_size = 512
epochs = 50
num_classes = 10
img_h = 16
img_w = 16
log_every = 20



# TODO Change hp values to argparse

hp = dict(
        nhid = 2048, # the dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 12, # the number of nn.TransformerEncoderLayer in nn.TransformerEncowandder
        nheads = 8, # the number of heads in the multiheadattention models
        dropout = 0.5, # the dropout value
        lr = 1e-3,
        seq_len = 4,
        seq_bs = 16,
        img_bs = 64,
        emb_size = 512,
        img_h = 16,
        img_w = 16,
        log_every = 20,
    )

assert emb_size % hp['nheads'] == 0, "Embedding size not divisible by number of heads"
assert hp['img_bs'] == hp['seq_len'] * hp['seq_bs'], "Batch size is not seq_len * transformer_batch"

class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir="data"):
        super(MNISTDataModule, self).__init__()
        self.data_dir = data_dir

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        mnist = datasets.MNIST(download=False, train=True, root="data").data.float()
        self.transforms = T.Compose([ T.Resize((img_h, img_w)), T.ToTensor(), T.Normalize((mnist.mean()/255,), (mnist.std()/255,))])

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist = datasets.MNIST(self.data_dir, train=True, transform=self.transforms)
            self.train_dataset, self.val_dataset = random_split(mnist, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.MNIST(self.data_dir, train=False, transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=hp['img_bs'], num_workers=0, shuffle=True, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=hp['img_bs'], num_workers=0, shuffle=False, drop_last=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=hp['img_bs'], num_workers=0, shuffle=False, drop_last=True, pin_memory=True)

class Sceneformer(pl.LightningModule):
    
    def __init__(self, device, emb_size: int = 512):
        super(Sceneformer, self).__init__()
        self.dev = device
        # Variables for logging
        self.train_step = 1
        self.val_step = 1
        
        # Transformer
        image_size = (img_w * int(math.sqrt(hp['seq_len'])), img_h * int(math.sqrt(hp['seq_len'])))
        self.transformer = Transformer(emb_size, hp['seq_bs'], hp['seq_len'], self.dev, image_size, hp['nheads'], hp['nhid'], hp['nlayers'], hp['dropout'], num_classes)
        
        # self.disc = DCGANDiscriminator(image_channels=1).cuda()
        # self.disc.apply(weights_init)

        # self.init_weights()

    def forward(self, labels, images, targets, src_mask=None, tgt_mask=None):
        
        return self.transformer(labels, images, targets)

    def training_step(self, batch, batch_idx):#, optimizer_idx):

        # Data loading
        images, labels = batch
        # Model forward
        step_loss, predictions, target_imgs = self.transformer_step(labels, images)
        # Logging
        self.logger.experiment.log({'Train t_loss with custom step': step_loss, 'train_step': self.train_step})
        self.train_step += 1
        if self.global_step % log_every == 0:
            log_prediction(target_imgs, predictions, self.logger, title="Train Transformer Target and Ouptut")

        return step_loss
        #return {'loss': step_loss, 'preds': predictions}

    def training_epoch_end(self, training_step_outputs):
        pass
        # for pred in training_step_outputs:
        #     pass

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        step_loss, predictions, targets = self.transformer_step(labels, images)
        
        # Logging
        self.logger.experiment.log({'Val t_loss': step_loss, 'val_step': self.val_step})
        self.val_step += 1
        #self.log_prediction(targets, predictions, self.logger, "Validation Transformer Target and Ouptut")
        return step_loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=hp['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        return [optimizer], [scheduler]

    def transformer_loss(self, outputs, targets):
        #loss = nn.KLDivLoss()
        loss = torch.nn.MSELoss()
        return loss(outputs, targets)

    def transformer_step(self, labels, images):

        targets, target_imgs = self.transformer.get_targets(images)
        predictions = self(labels.unsqueeze(1), images, targets)
        t_loss = self.transformer_loss(predictions, target_imgs)
        return t_loss, predictions, target_imgs

def main(args):
    print(args)
    
    # input should be three-dimensional (Seq_len, N_batchs, Embedding)

    data_module = MNISTDataModule()
    # tb_logger = TensorBoardLogger('tb_logs', name='mnist_transformer')
    wandb_logger = WandbLogger(project="MNIST Transformer")
    trainer = pl.Trainer(
        gpus=1,
        # fast_dev_run=True,
        # overfit_batches=0.01, # 1% of training set used as batch to make it overfit
         limit_train_batches=0.05, # 10% of training data
        # limit_val_batches=0.1, # 10% of validation data
        num_sanity_val_steps=2,
        flush_logs_every_n_steps=20,
        progress_bar_refresh_rate=20,
        #max_epochs=epochs,
        #profiler=True,
        logger=wandb_logger,
        callbacks=[
            Logging(),
            #TensorboardGenerativeModelImageSampler()
        ]
    )

    wandb.login()

    hparams = dict(
        nhid = hp['nhid'], # the dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = hp['nlayers'], # the number of nn.TransformerEncoderLayer in nn.TransformerEncowandder
        nheads = hp['nheads'], # the number of heads in the multiheadattention models
        dropout = hp['dropout'], # the dropout value
        lr = hp['lr'],
        emb_size = hp['emb_size'],
        img_h = hp['img_h'],
        img_w = hp['img_w'],
    )

    wandb.init(
        config=hparams,
        #mode="disabled"
    )

    wandb.run.name = "Log Test"

    config = wandb.config

    device = 'cuda:0' if args['n_gpu'] == 1 else 'cpu'

    model = Sceneformer(device, emb_size)

    trainer.fit(model, data_module)
    
if __name__ == '__main__':
    with open("config.json") as c:

        args = json.load(c)
    main(args)

