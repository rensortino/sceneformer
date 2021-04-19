import math
from tqdm import tqdm
import time
import cv2
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
from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler

import torch.nn.functional as F
import torch.optim as optim
import random
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer

w_path = 'ckpt/resnet18_mnist.pt'
criterion = nn.KLDivLoss()
lr = 0.01 
seq_len = 4
t_bs = 16
batch_size = seq_len * t_bs
emb_size = 512
epochs = 30

"""## Helper Classes"""

class Logging(Callback):

    def on_epoch_end(self, trainer, pl_module):
        train_loss_mean = np.mean(pl_module.training_losses)
        # self.log('pl_logger_train_loss', pl_module.step_loss)
        trainer.logger.experiment.add_scalar('training_loss', train_loss_mean, global_step=pl_module.current_epoch)
        self.training_losses = []  # reset for next epoch

class ResNet18(nn.Module):
    def __init__(self, w_path, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.load_state_dict(torch.load(w_path)['model_state_dict'])
        self.out_layer = self.model._modules.get('avgpool')
        self.embedding_size = self.model._modules.get('fc').in_features

    def get_vectors(self, images):
        # Create a vector of zeros that will hold our feature vector
        # The 'avgpool' layer has an output size of 512
        visual_embedding = torch.zeros(self.embedding_size * batch_size).cuda()

        # Define a function that will copy the output of a layer
        def copy_data(module, input, output):
            visual_embedding.copy_(output.flatten())

        # Attach that function to our selected layer
        hook = self.out_layer.register_forward_hook(copy_data)
        # Run the model on our transformed image
        with torch.no_grad():
            self.model(images)
        # Detach our copy function from the layer
        hook.remove()
        # Return the feature vector
        return visual_embedding

    def forward(self, imgs):
        output = self.model(imgs)
        return output

def one_hot_encoder(labels, bs=64):
    encoded_vector = torch.cat([torch.tensor([0,1,2,3,4,5,6,7,8,9]).unsqueeze(0)] * bs).cuda()
    return torch.where(encoded_vector == labels, 1, 0)

class Generator(nn.Module):
    
    def __init__(self, cond_size, noise_size, base_filters=128, channels=3):
        super().__init__()
        # Set parameters
        self.cond_size = cond_size
        self.noise_size = noise_size
        # Alias for base filters
        F = base_filters
        self.F = F
        # Input size: Z+C -> 4F x 4 x 4
        self.convt_1 = nn.ConvTranspose2d(noise_size + cond_size, 4*F, kernel_size=4, stride=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(4*F)
        # Input size: 4F x 4 x 4 -> 2F x 8 x 8
        self.convt_2 = nn.ConvTranspose2d(4*F, 2*F, kernel_size=4, padding=1, stride=2)
        self.bn_2 = nn.BatchNorm2d(2*F)
        # Input size: 2F x 8 x 8 -> 2F x 16 x 16
        self.convt_3 = nn.ConvTranspose2d(2*F, 2*F, kernel_size=4, padding=1, stride=2)
        self.bn_3 = nn.BatchNorm2d(2*F)
        # Input size: 2F x 16 x 16 -> F x 32 x 32
        self.convt_4 = nn.ConvTranspose2d(2*F, F, kernel_size=4, padding=1, stride=2)
        self.bn_4 = nn.BatchNorm2d(F)
        # Input size: F x 32 x 32 -> 3 x 64 x 64
        self.convt_5 = nn.ConvTranspose2d(F, channels, kernel_size=4, padding=1, stride=2)

    # Input: BxZ
    def forward(self, n, c):
        # Concat noise and condition
        x = torch.cat((n,c), 1)
        x = x.view(x.size(0), x.size(1), 1, 1)
        # Input size: Z+C -> 4F x 4 x 4
        x = F.leaky_relu(self.bn_1(self.convt_1(x)), 0.2)
        x = x.view(-1, 4*self.F, 4, 4)
        # Input size: 4F x 4 x 4 -> 2F x 8 x 8
        x = F.leaky_relu(self.bn_2(self.convt_2(x)), 0.2)
        # Input size: 2F x 8 x 8 -> 2F x 16 x 16
        x = F.leaky_relu(self.bn_3(self.convt_3(x)), 0.2)
        # Input size: 2F x 16 x 16 -> F x 32 x 32
        x = F.leaky_relu(self.bn_4(self.convt_4(x)), 0.2)
        # Input size: F x 32 x 32 -> 3 x 64 x 64
        x = torch.tanh(self.convt_5(x))
        return x

class MNISTDataModule(pl.LightningDataModule):

    def prepare_data(self):
        #self.transforms = T.Compose([to_tensor, normalize])
        mnist = datasets.MNIST(download=False, train=True, root="data").train_data.float()
        self.transforms = T.Compose([ T.Resize((224, 224)), T.ToTensor(), T.Normalize((mnist.mean()/255,), (mnist.std()/255,))])
        train_dataset = datasets.MNIST(root="data", download=True, train=True, transform=self.transforms)
        self.test_dataset = datasets.MNIST(root="data", download=True, train=False, transform=self.transforms)

        # Compute dataset sizes
        num_train = len(train_dataset)
        # List of indexes on the training set
        train_idx = list(range(num_train))
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
        self.train_dataset = Subset(train_dataset, train_idx)
        self.val_dataset = Subset(train_dataset, val_idx)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=64, num_workers=0, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=64, num_workers=0, shuffle=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=64, num_workers=0, shuffle=False, drop_last=True)

class TransformerModel(pl.LightningModule):
    
    def __init__(self, emb_size, nhead=3, nhid=256, nlayers=3, dropout=0.5, num_classes=10,):
        super(TransformerModel, self).__init__()
        
        self.model_type = 'Transformer'
        encoder_layers = TransformerEncoderLayer(emb_size, nhead, nhid, dropout)
        self.encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(num_classes, emb_size)
        self.feature_extractor = ResNet18(w_path)
        self.emb_size = emb_size
        decoder_layers = TransformerDecoderLayer(emb_size, nhead, nhid, dropout)
        self.decoder = TransformerDecoder(decoder_layers, nlayers)
        self.noise_size = 100
        self.gen = Generator(emb_size, self.noise_size)
        self.training_losses = []

        # self.init_weights()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        return optimizer

    def KLDivergence(self, outputs, targets):
        loss = nn.KLDivLoss()
        outputs = outputs.view(-1)
        targets = targets.view(-1)
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
        src = src.view(seq_len, t_bs, -1).type(torch.cuda.FloatTensor)
        # Run encoder forward      
        enc_out = self.encoder(src) 
        # extract image embeddings and reshape for decoder input
        img_embeddings = self.feature_extractor.get_vectors(imgs).view(seq_len, t_bs, -1) 
        # Run decoder forward
        output = self.decoder(img_embeddings.cuda(), enc_out)

        return output

    def training_step(self, batch, batch_idx):
        images, labels = batch
        targets = self.feature_extractor.get_vectors(images).view(seq_len, t_bs, -1)
        src_mask = self.generate_square_subsequent_mask(seq_len)
        output = self.forward(labels.unsqueeze(1), images, src_mask)
        noise = torch.cuda.FloatTensor(images.size(0), self.noise_size)
        noise.normal_(0,1)
        reconst_image = self.gen(noise, output.view(-1, emb_size))
        step_loss = self.KLDivergence(output, targets)
        self.step_loss = step_loss
        self.training_losses.append(self.step_loss.item())

        return step_loss

def main():
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    model = TransformerModel(emb_size, nhead, nhid, nlayers, dropout)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # input should be three-dimensional (Seq_len, N_batchs, Embedding)

    data_module = MNISTDataModule()
    logger = TensorBoardLogger('tb_logs', name='mnist_transformer')
    trainer = pl.Trainer(
        gpus=1, 
        progress_bar_refresh_rate=20, 
        logger=logger, 
        callbacks=[
            Logging(), 
            #TensorboardGenerativeModelImageSampler()
        ]
    )
    trainer.fit(model, data_module)
    
main()

