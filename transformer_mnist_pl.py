import math, time, random, cv2
import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torchvision import datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import Callback
from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler

import wandb

from cgan import weights_init, DCGANGenerator, DCGANDiscriminator


# TODO Current objective is to translate embedding (flattened image) to sequences maintaining an ordering between successive elements

hp = dict(
        nhid = 200, # the dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 8, # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nheads = 8, # the number of heads in the multiheadattention models
        dropout = 0.2, # the dropout value
        noise_size = 100,
        lr = 0.1,
        seq_len = 4,
        t_bs = 16,
        batch_size = 64,
        emb_size = 1024,
        epochs = 30,
        num_classes = 10,
        img_h = 16,
        img_w = 16,
    )

assert hp['emb_size'] % hp['nheads'] == 0, "Embedding size not divisible by number of heads"
assert hp['batch_size'] == hp['seq_len'] * hp['t_bs'], "Batch size is not seq_len * transformer_batch"

class Logging(Callback):

    def __init__(self):
        self.start_time = None

    def on_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_fit_start(self, trainer, pl_module):
        if not trainer.fast_dev_run:
            trainer.logger.watch(trainer.model)

    def on_epoch_end(self, trainer, pl_module):
        #trainer.logger.experiment.log({"Reconstructed Image": [wandb.Image(np_image, caption="First of each batch")]})
        # self.log('pl_logger_train_loss', pl_module.step_loss)
        # trainer.logger.experiment.add_scalar('training_loss', train_loss_mean, global_step=pl_module.current_epoch)
        pl_module.log('Elapsed Time', time.time() - self.start_time)

def show_image(img):
    plt.imshow(np.transpose(img.numpy(), (1,2,0)), interpolation='nearest')
    plt.show()

class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir="data"):
        super(MNISTDataModule, self).__init__()
        self.data_dir = data_dir

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        mnist = datasets.MNIST(download=False, train=True, root="data").data.float()
        self.transforms = T.Compose([ T.Resize((hp['img_h'], hp['img_w'])), T.ToTensor(), T.Normalize((mnist.mean()/255,), (mnist.std()/255,))])

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist = datasets.MNIST(self.data_dir, train=True, transform=self.transforms)
            self.train_dataset, self.val_dataset = random_split(mnist, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.MNIST(self.data_dir, train=False, transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=64, num_workers=0, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=64, num_workers=0, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=64, num_workers=0, shuffle=False, drop_last=True)


class Transformer(nn.Module):
    def __init__(self, emb_size, nhead=3, nhid=256, nlayers=3, dropout=0.5, num_classes=10):
        super(Transformer, self).__init__()
        
        self.emb_size = emb_size
        self.embedding = nn.Embedding(num_classes, emb_size)
        #self.feature_extractor = ResNet18(w_path)
        # Encoder
        encoder_layers = TransformerEncoderLayer(emb_size, nhead, nhid, dropout)
        self.encoder = TransformerEncoder(encoder_layers, nlayers)
        # Decoder
        decoder_layers = TransformerDecoderLayer(emb_size, nhead, nhid, dropout)
        self.decoder = TransformerDecoder(decoder_layers, nlayers)

        # self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def get_seq_targets(self, img_seq):
        '''
        img_ seq = [seq_len, h, w] = [4, 16, 16]
        '''
        img_seq = img_seq.unsqueeze(1) # Restore Channel info
        seq_len, c, h, w = img_seq.shape
        grid = torch.zeros(seq_len, c, h, w)
        img_embeddings = []
        for i, img in enumerate(img_seq):
            # Place the image in the right quadrant
            grid[i] = img
            # Construct the grid from the batch of images
            img_grid = make_grid(grid.cpu(), nrow=2, padding=0)
            #show_image(img_grid)
            # Take the first channel (Grayscale images, the channels are all the same)
            img_embedding = img_grid[0].flatten()
            img_embeddings.append(img_embedding.unsqueeze(0)) # Unsqueeze to cat along dim 0 later
        return torch.cat(img_embeddings)

    def get_targets(self, images):
        img_h, img_w = images.shape[2:]
        images = images.view(hp['seq_len'], hp['t_bs'], img_h, img_w)
        images = images.permute(1,0,2,3) # transpose axes to allow iterating through sequences
        targets = [self.get_seq_targets(img_seq) for img_seq in images]
        targets = list(map(lambda x: x.unsqueeze(0), targets))
        targets = torch.cat(targets)
        return targets

    def forward(self, labels, images, src_mask=None, tgt_mask=None):
        seq_len = hp['seq_len']
        bs = hp['t_bs']
        #targets = self.feature_extractor.get_vectors(images).view(hp['seq_len'], hp['t_bs'], -1)
        src_mask = self.generate_square_subsequent_mask(seq_len)
        obj_emb = self.embedding(labels.view(seq_len, bs)) #* math.sqrt(self.emb_size)
        # Run encoder forward
        enc_out = self.encoder(obj_emb)
        # extract image embeddings and reshape for decoder input
        #img_embeddings = self.feature_extractor.get_vectors(images).view(hp['seq_len'], hp['t_bs'], -1)
        targets = self.get_targets(images)
        # Run decoder forward
        output = self.decoder(targets.cuda(), enc_out)

        return output

def log_prediction(gt, pred, logger, title : str = "Logged Image"):
    h, w = hp['img_h'], hp['img_w']

    gt = gt.view(hp['seq_len'] * hp['t_bs'], 1, h * int(math.sqrt(hp['seq_len'])), w * int(math.sqrt(hp['seq_len']))) # Reshape to [B,C,H,W]
    pred = pred.view(hp['seq_len'] * hp['t_bs'], 1, h * int(math.sqrt(hp['seq_len'])), w * int(math.sqrt(hp['seq_len']))) # Reshape to [B,C,H,W]

    gt_grid = make_grid(gt.cpu(), nrow=hp['seq_len'], padding=0)
    p_grid = make_grid(pred.cpu(), nrow=hp['seq_len'], padding=0)

    logger.experiment.log({title :[
        wandb.Image(p_grid.cpu(), caption="Prediction"),
        wandb.Image(gt_grid.cpu(), caption="Ground Truth"), 
    ]})

class Sceneformer(pl.LightningModule):
    
    def __init__(self, emb_size, noise_size=100):
        super(Sceneformer, self).__init__()
        
        # Transformer
        #self.feature_extractor = ResNet18(w_path)
        self.transformer = Transformer(emb_size, hp['nheads'], hp['nhid'], hp['nlayers'], hp['dropout'], hp['num_classes'])

        # GAN
        # self.gen = DCGANGenerator(emb_size, self.noise_size, image_channels=1).cuda()
        # self.gen.apply(weights_init)
        # self.disc = DCGANDiscriminator(image_channels=1).cuda()
        # self.disc.apply(weights_init)

        self.generated_image = None

        # self.init_weights()

    def configure_optimizers(self):
        optimizerT = torch.optim.SGD(self.parameters(), lr=hp['lr'])
        # optimizerD = optim.Adam(self.gen.parameters(), lr=hp['lr'])
        # optimizerG = optim.Adam(self.disc.parameters(), lr=hp['lr'])
        return [optimizerT]#, optimizerG, optimizerD]

    def transformer_loss(self, outputs, targets):
        #loss = nn.KLDivLoss()
        loss = torch.nn.MSELoss()
        outputs = outputs.view(-1, hp['emb_size'])
        targets = targets.view(-1, hp['emb_size'])
        return loss(outputs, targets)

    def forward(self, labels, images, src_mask=None, tgt_mask=None):
        
        return self.transformer(labels, images)

    def transformer_step(self, labels, images):

        # targets = get_targets(labels, images)
        targets = self.transformer.get_targets(images).cuda()
        #log_images(targets, self.logger, "Target Batch sample", "Batch of ground truth data to pass to the decoder (each row is a sequence)")
        trf_output = self(labels.unsqueeze(1), images)
        log_prediction(targets, trf_output, self.logger, "Transformer Target and Ouptut")
        # targets = self.feature_extractor.get_vectors(images).view(hp['seq_len'], hp['t_bs'], -1)
        t_loss = self.transformer_loss(trf_output, targets)

        self.log('Default t_loss', t_loss)
        #self.logger.log_metrics({'Logger t_loss': t_loss}, step=self.current_epoch)
        return t_loss

    def training_step(self, batch, batch_idx):#, optimizer_idx):

        # Data loading
        images, labels = batch
        
        # Model forward
        step_loss = self.transformer_step(labels, images)
        #if optimizer_idx == 0:
        #    step_loss = self.transformer_step(labels, images)
#
        #if optimizer_idx == 1:
        #    step_loss = self.generator_step(labels, images)
#
        #if optimizer_idx == 2:
        #    step_loss = self.discriminator_step(labels, images)

        # Image generation
        #reconst_images = ((gen_output + 1) * 256) // 2
        #np_images = np.asarray(reconst_images.detach().cpu()).transpose(0,2,3,1) # Batch + PNG Image shape
        ## Log random sample from batch
        #sample_idx = math.floor(random.random() * batch_size)
        #self.logger.experiment.log({"Reconstructed Image": [wandb.Image(np_images[sample_idx], caption=f'Batch: {batch_idx}, Sample: {sample_idx}')]})

        return step_loss

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

def main():
    
    model = Sceneformer(hp['emb_size'])

    wandb.login()

    wandb.init(config=hp)

    config = wandb.config

    optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # input should be three-dimensional (Seq_len, N_batchs, Embedding)

    data_module = MNISTDataModule()
    # tb_logger = TensorBoardLogger('tb_logs', name='mnist_transformer')
    wandb_logger = WandbLogger(project="MNIST Transformer")
    trainer = pl.Trainer(
        gpus=1, 
        # fast_dev_run=True,
        # overfit_batches=0.01, # 1% of training set used as batch to make it overfit
        # limit_train_batches=0.1, # 10% of training data
        # limit_val_batches=0.1, # 10% of validation data
        flush_logs_every_n_steps=100,
        progress_bar_refresh_rate=20, 
        profiler=True,
        logger=wandb_logger, 
        callbacks=[
            Logging(), 
            #TensorboardGenerativeModelImageSampler()
        ]
    )
    trainer.fit(model, data_module)
    
main()

