
from torch.nn import Transformer
import pytorch_lightning as pl
from log_utils import log_prediction
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
import math
from data_processing import get_targets, append_tokens, process_labels
import wandb

TOKENS = {
        "SOS": 0.0,
        "EOS": 1.0,
        "PAD": 0.5
    }

class ImageGenerator(nn.Module):
    def __init__(self, img_size, emb_size=1024, ngf=128, channels=3):
        super(ImageGenerator, self).__init__()

        assert len(img_size) == 2, "img_size has to be a tuple (h, w)"

        self.emb_size = emb_size
        self.resize = T.Resize(img_size)

        def block(in_feat, out_feat, kernel, stride, pad, last=False):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, kernel, stride, pad, bias=False)]
            return layers

        self.model = nn.Sequential(
            *block(emb_size, ngf * 8, 4, 1, 0),
            *block(ngf * 8, ngf * 4, 4, 2, 1),
            *block(ngf * 4, ngf * 2, 4, 2, 1),
            *block(ngf * 2, ngf, 4, 2, 1),
            *block(ngf, ngf // 2, 4, 2, 1),
            *block(ngf // 2, channels, 4, 2, 1, last=True)
        )

        #self.model = nn.Sequential(
        #    *block(emb_size, channels, 32, 1, 0, last=True),
        #    *block(ngf, channels, 4, 2, 1, last=True)
        #)

    def forward(self, x):
        x = x.view(x.size(0) * x.size(1), self.emb_size, 1, 1)
        x = self.model(x)
        return self.resize(x)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=10):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.to(self.pe.device)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class YTID(pl.LightningModule):
    
    def __init__(self,
                feature_extractor,
                embedding,
                pos_enc,
                img_gen,
                disc,
                args,
                criterion):
        super(YTID, self).__init__()
        self.automatic_optimization = False # disable automatic calling of backward()
        self.max_seq_len = args.model.max_seq_len
        self.feature_extractor = feature_extractor
        self.pos_enc = pos_enc
        self.embedding = embedding
        self.criterion = criterion

        # Variables for logging
        self.train_step = 1
        self.val_step = 1

        self.args = args
        
        # Transformer
        self.transformer = Transformer(
            args.model.emb_size,
            args.model.n_heads,
            args.model.n_layers,
            args.model.n_layers,
            args.model.ff_dim,
            args.model.dropout,
        )

        self.img_gen = img_gen
        self.discriminator = disc

        # self.init_weights()

    def forward(self, src, targets, src_key_padding_mask=None, tgt_key_padding_mask=None):

        r"""Take in and process masked source/target sequences.

        Args:
            src/tgt: the sequence to the encoder/decoder (required).
            [src/tgt]_mask: the additive mask for the src/tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            [src/tgt]_key_padding_mask: the ByteTensor mask for src/tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(S, N, E)`, `(N, S, E)` if batch_first.
            - tgt: :math:`(T, N, E)`, `(N, T, E)` if batch_first.
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.
        """

        # Create input embeddings
        in_seq = self.embedding(src.int())* math.sqrt(self.args.model.emb_size)
        in_seq = self.pos_enc(in_seq)

        # Forward transformer
        in_seq = in_seq.to(self.args.device)
        targets = targets.to(self.args.device)
        tgt_mask = self.transformer.generate_square_subsequent_mask(targets.shape[0]).to(self.args.device)
        trf_out = self.transformer(in_seq, targets, tgt_mask=tgt_mask)
        image_vector = trf_out[:,:,:-4]
        bbox = trf_out[:,:,-4:]

        # TODO Restore img_gen
        #out_imgs = self.img_gen(trf_out)
        out_imgs = image_vector.reshape(
            image_vector.shape[0] * image_vector.shape[1],
            self.args.data_loader.n_channels, 
            self.args.data_loader.img_w, 
            self.args.data_loader.img_h
        )

        return out_imgs, image_vector, bbox

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def training_step(self, batch, batch_idx, optimizer_idx):

        # Define Optimizers
        g_opt, d_opt, t_opt, b_opt = self.optimizers()

        # Data loading
        original_images, labels = batch
        images = original_images.view(self.args.model.seq_len, self.args.model.seq_bs, self.args.data_loader.n_channels, self.args.data_loader.img_h, self.args.data_loader.img_w)

        targets, tgt_imgs = get_targets(self.feature_extractor, images)
        #padded_tgt = get_padded_tgt(targets)

        # tgt = [<SOS>, [embeddings], <EOS> (, [<PAD>] ) ]
        # TODO Wrap in function
        in_seq = labels.reshape(self.args.model.seq_bs, self.args.model.seq_len)
        in_seq = process_labels(labels)
        
        # d_loss = self.discriminator_step(in_seq, targets, tgt_imgs)

        # d_opt.zero_grad()
        # self.manual_backward(d_loss)
        # d_opt.step()

        # self.log("d_loss", d_loss)

        # wandb.log({"d_loss": d_loss})

        #self.predictions, self.image_vectors, self.bbox = self(in_seq, targets[:-1])
        
        #g_loss = self.generator_step(tgt_imgs, targets)

        # g_opt.zero_grad()
        # self.manual_backward(g_loss, retain_graph=True)

        t_loss, box_loss = self.transformer_step(in_seq, targets)

        t_opt.zero_grad()
        b_opt.zero_grad()
        self.manual_backward(t_loss, retain_graph=True)
        self.manual_backward(box_loss)

        #g_opt.step()
        t_opt.step()
        b_opt.step()

        self.log_dict({'t_loss': t_loss, "box_loss": box_loss}, prog_bar=True)


        #TODO Wrap in function
        self.train_step += 1
        if self.global_step % self.args.trainer.log_every_n_steps == 0:
            target_imgs = tgt_imgs.view(tgt_imgs.shape[0] * tgt_imgs.shape[1], self.args.data_loader.n_channels, tgt_imgs.shape[3], tgt_imgs.shape[4])
            log_prediction(target_imgs, self.predictions, self.logger, title="Train Transformer Target and Ouptut")

    def discriminator_step(self, in_seq, targets, tgt_imgs):
        ##########################
        # Optimize Discriminator #
        ##########################
        # Measure discriminator's ability to classify real from generated samples

        # Transformer forward
        # tgt[:-1] (shifted right because the transformer has to predict based on previous output)
        # FIXME Use same prediction set or run multiple forward?
        # self.predictions, self.image_vectors, self.bbox = self(in_seq, targets[:-1])
        predictions, _, _ = self(in_seq, targets[:-1])

        sl, bs, c, h, w = tgt_imgs.shape
        tgt_imgs = tgt_imgs.reshape(sl * bs, c, h, w)

        # how well can it label as real?
        real = torch.ones(tgt_imgs.size(0), 1, device=self.args.device)

        real_loss = self.adversarial_loss(self.discriminator(tgt_imgs), real)

        # how well can it label as fake?
        fake = torch.zeros(predictions.size(0), 1, device=self.args.device)

        fake_loss = self.adversarial_loss(
            self.discriminator(predictions.detach()), fake)

        wandb.log({"d_real_loss": real_loss})
        wandb.log({"d_fake_loss": fake_loss})

        # discriminator loss is the sum of these
        return real_loss + fake_loss

    def generator_step(self, tgt_imgs, targets):
        ######################
        # Optimize Generator #
        ######################

        # adversarial loss is binary cross-entropy
        sl, bs, c, h, w = tgt_imgs.shape
        tgt_imgs = tgt_imgs.reshape(sl * bs, c, h, w)
        real = torch.ones(tgt_imgs.size(0), 1, device=self.args.device)

        g_loss = self.adversarial_loss(self.discriminator(self.predictions), real)

        wandb.log({"g_loss": g_loss})

        #self.log_dict({'t_loss': mse_loss, 'g_loss': t_adv_loss, 'd_loss': d_loss, 'd_real_loss': real_loss, 'd_fake_loss': fake_loss}, prog_bar=True)
        return g_loss

    def transformer_step(self, in_seq, targets):
        ########################
        # Optimize Transformer #
        ########################

        # TODO Check if processed input is the same as original input (no alteration has been done)
        self.predictions, image_vectors, bbox = self(in_seq, targets[:-1])

        tgt_vectors = targets[:,:,:-4]
        tgt_bboxes = targets[:,:,-4:]

        # tgt[1:] (shifted left to compare the real sequences, without the <SOS>)
        t_loss = self.transformer_loss(self.criterion, image_vectors, tgt_vectors[1:])
        # Exclude SOS and EOS
        box_loss = self.transformer_loss(self.criterion, bbox[:-1], tgt_bboxes[1:-1])

        # Compute loss
        wandb.log({"t_loss": t_loss})
        wandb.log({"box_loss": box_loss})

        return  t_loss, box_loss

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def transformer_loss(self, criterion,  outputs, targets):

        targets = targets.reshape(targets.shape[0] * targets.shape[1], -1).to(self.args.device)
        outputs = outputs.reshape(targets.shape).to(self.args.device)

        loss = criterion(outputs, targets)        
        return loss 

    def custom_histogram_adder(self):
        for name,params in self.named_parameters():
            self.logger.experiment.add_histogram(name,params,self.current_epoch)

    def training_epoch_end(self,outputs):
        self.custom_histogram_adder()


    #def validation_step(self, batch, batch_idx):
    #    images, labels = batch
    #    images = images.unsqueeze(1) # add the transformer sequence dimension
    #    # Reshape images to seq_len, batch_size
    #    images = images.view(self.args.model.seq_len, self.args.model.seq_bs, self.args.data_loader.n_channels, self.args.data_loader.img_h, self.args.data_loader.img_w)
    #    step_loss, predictions, target_imgs = self.transformer_step(labels, images)
    #    
    #    # Logging
    #    #self.logger.experiment.log({'Val t_loss': step_loss, 'val_step': self.val_step})
    #    self.logger.experiment.add_scalar('Loss/Val', step_loss, self.val_step)
    #    self.log("Default Transformer Loss", step_loss)
#
    #    self.val_step += 1
    #    if self.global_step % self.args.trainer.log_every_n_steps == 0:
    #        target_imgs = target_imgs.view(target_imgs.shape[0] * target_imgs.shape[1], self.args.data_loader.n_channels, target_imgs.shape[3], target_imgs.shape[4])
    #        log_prediction(target_imgs, predictions, self.logger, title="Validation Transformer Target and Ouptut")
    #    return step_loss
#
    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        # FIXME Fix optimizers
        g_optimizer = torch.optim.Adam(self.transformer.parameters(), lr=self.args.model.g_lr)
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.model.d_lr)
        t_optimizer = torch.optim.Adam(self.transformer.parameters(), lr=self.args.model.t_lr)
        b_optimizer = torch.optim.Adam(self.transformer.parameters(), lr=self.args.model.t_lr)

        g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, 1.0, gamma=0.95)
        d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, 1.0, gamma=0.95)
        t_scheduler = torch.optim.lr_scheduler.StepLR(t_optimizer, 1.0, gamma=0.95)
        b_scheduler = torch.optim.lr_scheduler.StepLR(b_optimizer, 1.0, gamma=0.95)


        return [g_optimizer, d_optimizer, t_optimizer, b_optimizer], [g_scheduler, d_scheduler, t_scheduler, b_scheduler]










    def transformer_step_old(self, labels, images):
        r'''
        Args:
        in_seq : (In_seq_len, N_batchs, Embedding)
        out_seq : (Out_seq_len, N_batchs, Embedding)
        [src/tgt]_mask : what elements to attend (triangular mask)
        [src/tgt]_key_padding_mask : what is padding (True) and what is value (False)

        images shape:  [B,C,H,W]
        '''
        
        # Create target images
        targets, tgt_imgs = get_targets(self.feature_extractor, images)
        #padded_tgt = get_padded_tgt(targets)

        # tgt = [<SOS>, [embeddings], <EOS> (, [<PAD>] ) ]
        in_seq = labels.reshape(self.args.model.seq_bs, self.args.model.seq_len)
        in_list = append_tokens(in_seq.tolist(), TOKENS['EOS'], TOKENS['SOS'])
        in_seq = torch.tensor(in_list, device=self.args.device)
        out_imgs = self(in_seq, targets)
        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(out_imgs.size(0), 1, device=self.args.device)

        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(self.discriminator(out_imgs), valid)

        # Compute loss
        # tgt[1:] (shifted left to compare the real sequences, without the <SOS>)
        #t_loss = self.transformer_loss(self.criterion, out_imgs, tgt_imgs)
        self.log("g_loss", g_loss)
        wandb.log({"g_loss": g_loss})
        return g_loss, out_imgs, tgt_imgs
