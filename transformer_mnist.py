import math
from tqdm import tqdm
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import datasets
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback

import torch.nn.functional as F
import torch.optim as optim
import random
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class Logging(Callback):
    
    def on_init_start(self, trainer):
        print('Starting to init trainer!')

    def on_init_end(self, trainer):
        print('trainer is init now')

    def on_epoch_end(self, trainer, pl_module):
        #trainer.logger.experiment.add_scalar("Loss/Train", trainer.cur_loss, trainer.epoch)
        train_loss_mean = np.mean(pl_module.training_losses)
        self.log('pl_logger_train_loss', pl_module.current_loss)
        trainer.logger.experiment.add_scalar('training_loss', train_loss_mean, global_step=pl_module.current_epoch)
        self.training_losses = []  # reset for next epoch
        print('do something when training ends')

class ResNet50(nn.Module):
    def __init__(self, w_path, num_classes=10):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.load_state_dict(torch.load(w_path))

    def forward(self, imgs):
        output = self.model(imgs)
        return output

# Select device
# dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
w_path = 'ckpt/best.pt'
criterion = nn.KLDivLoss()
lr = 0.01 # learning rate
seq_len = 16
bs = 4
emb_size = 10
feature_extractor = ResNet50(w_path).cuda()
epochs = 30 # The number of epochs

class MNISTDataModule(pl.LightningDataModule):

    def normalize_(self, x):
        x[x > 0.5] = 1
        x[x <= 0.5] = -1
        return x

    def setup(self, stage):
        to_tensor = T.ToTensor()
        normalize = self.normalize_
        extend_chans = T.Lambda(lambda x: x.repeat(3, 1, 1))
        self.transforms = T.Compose([to_tensor, normalize])

    def prepare_data(self):
        pass

    def train_dataloader(self):
        train_dataset = datasets.MNIST(root="data", download=False, train=True, transform=self.transforms)

        # Compute dataset sizes
        num_train = len(train_dataset)
        # List of indexes on the training set
        train_idx = list(range(num_train))
        # Compute dataset sizes
        num_train = len(train_dataset)
        # Shuffle training set
        random.shuffle(train_idx)

        # Validation fraction
        val_frac = 0.1
        # Compute number of samples
        num_val = int(num_train*val_frac)
        num_train = num_train - num_val
        # Split training set
        val_idx = train_idx[num_train:]
        train_idx = train_idx[:num_train]

        # Split train_dataset into training and validation
        self.val_dataset = Subset(train_dataset, val_idx)
        self.train_dataset = Subset(train_dataset, train_idx)
        return DataLoader(self.train_dataset, batch_size=64, num_workers=0, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=64, num_workers=0, shuffle=True, drop_last=True)


    def test_dataloader(self):
        self.test_dataset = datasets.MNIST(root="data", download=False, train=False, transform=self.transforms)
        return DataLoader(self.test_dataset, batch_size=64, num_workers=0, shuffle=False, drop_last=True)

class TransformerModel(pl.LightningModule):
    
    def __init__(self, emb_size, nhead=3, nhid=256, nlayers=3, dropout=0.5):
        super(TransformerModel, self).__init__()
        
        self.model_type = 'Transformer'
        encoder_layers = TransformerEncoderLayer(emb_size, nhead, nhid, dropout)
        self.encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = one_hot_encoder
        self.feature_extractor = ResNet50(w_path)
        self.emb_size = emb_size
        decoder_layers = TransformerDecoderLayer(emb_size, nhead, nhid, dropout)
        self.decoder = TransformerDecoder(decoder_layers, nlayers)
        self.training_losses = []

        # self.init_weights()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        return optimizer

    def KLDivergence(self, outputs, targets):
        loss = nn.KLDivLoss()
        return loss(outputs, targets)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, in_objs, imgs, src_mask=None, tgt_mask=None):
        src = self.embedding(in_objs) #* math.sqrt(self.emb_size)
        # reshape input for transformer  
        src = src.view(seq_len, bs, -1).type(torch.cuda.FloatTensor)
        # Run encoder forward      
        enc_out = self.encoder(src) 
        # extract image embeddings and reshape for decoder input
        img_embeddings = self.feature_extractor(imgs).view(seq_len, bs, -1) 
        # Run decoder forward
        output = self.decoder(img_embeddings, enc_out)
        return output

    def training_step(self, batch, batch_idx):
        images, labels = batch
        targets = feature_extractor(images)
        src_mask = self.generate_square_subsequent_mask(seq_len)
        output = self.forward(labels.unsqueeze(1), images, src_mask)
        step_loss = self.KLDivergence(output.view(-1, emb_size), targets)
        self.step_loss = step_loss
        self.training_losses.append(self.step_loss.item())

        return step_loss

def one_hot_encoder(labels, bs=64):
    encoded_vector = torch.cat([torch.tensor([0,1,2,3,4,5,6,7,8,9]).unsqueeze(0)] * bs).cuda()
    return torch.where(encoded_vector == labels, 1, 0)


def main():
    emsize = 10 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    model = TransformerModel(emsize, nhead, nhid, nlayers, dropout)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # input should be three-dimensional (Seq_len, N_batchs, Embedding)

    data_module = MNISTDataModule()
    logger = TensorBoardLogger('tb_logs', name='mnist_transformer')
    trainer = pl.Trainer(gpus=1, logger=logger, callbacks=[Logging()])
    trainer.fit(model, data_module)

def train(model, loaders, epoch, optimizer, scheduler):
    model.train() # Turn on the train mode
    
    total_loss = 0.
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(seq_len).to(dev)
    for batch, (images, labels) in tqdm(enumerate(loaders['train'])):
        images = images.to(dev) 
        labels = labels.to(dev)
        targets = feature_extractor(images)
        optimizer.zero_grad()
        output = model(labels.unsqueeze(1), images, src_mask)
        loss = criterion(output.view(-1, emb_size), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(loaders['train']), scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    src_mask = model.generate_square_subsequent_mask(seq_len).to(dev)
    with torch.no_grad():
        for batch, (images, labels) in tqdm(enumerate(loaders['val'])):
            images = images.to(dev) 
            labels = labels.to(dev)
            targets = feature_extractor(images)
            output = eval_model(labels.unsqueeze(1), images, src_mask)
            output_flat = output.view(-1, emb_size)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)

if __name__ == '__main__':
    main()