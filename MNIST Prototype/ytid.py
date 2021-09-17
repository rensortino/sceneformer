
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from data_processing import accuracy, extract_features, normalize, split_list
import numpy as np
from layers import NoamOpt
import pytorch_lightning as pl
from torchvision.utils import save_image
import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
import os
from torchvision.utils import make_grid
import wandb

class YTID(pl.LightningModule):
    
    def __init__(self,
                # hparams,
                img_transformer: nn.Module,
                img_gen: nn.Module,
                disc: nn.Module,
                backbone,
                logger,
                token_container,
                example_input,
                args):
        super(YTID, self).__init__()
        # self.hparams.update(hparams)
        self.automatic_optimization = False # disable automatic calling of backward()
        self.max_seq_len = args.max_seq_len
        self.example_input_array = example_input
        self.backbone = backbone
        self.token_container = token_container
        self.writer = logger
        self.phase = None

        # Variables for logging
        self.step = { p: 1 for p in ['train', 'val', 'test'] }

        self.args = args
        
        # Transformer
        self.img_transformer = img_transformer
        self.img_gen = img_gen
        self.discriminator = disc

    def forward(self, src, targets):

        # FIXME Transformer output should be handled
        img_features = self.img_transformer(src, targets)

        # predictions = self.img_gen(img_features)
        predictions = img_features

        return predictions

    def training_step(self, batch, batch_idx):

        self.phase = 'train'

        # Define Optimizers
        self.t_opt, self.g_opt, self.d_opt = self.optimizers()

        # Data loading
        images, src_seq, tgt_images, boxes = batch
        with torch.no_grad():
            tgt_features = extract_features(self.backbone, tgt_images)

        tgt_seq = torch.cat((tgt_features, boxes), dim=2)
        
        # Transformer Forward
        features, pred_boxes = self.transformer_step(src_seq, tgt_seq, self.t_opt)

        # Generator Forward
        pred_imgs = self.generator_step(features[:-1].detach(), tgt_images[1:-1], src_seq[1:-1], self.g_opt)
        # pred_imgs = images.unsqueeze(1)

        # Increase phase step (for logging)
        self.step[self.phase] += 1

        return {'images': pred_imgs, 'gt': tgt_images, 'gt_boxes': boxes[1:], 'pred_boxes': pred_boxes}

    def validation_step(self, batch, batch_idx):

        self.phase = 'val'
        
        # Data loading
        images, src_seq, tgt_images, boxes = batch
        with torch.no_grad():
            tgt_features = extract_features(self.backbone, tgt_images)

            tgt_seq = torch.cat((tgt_features, boxes), dim=2)

            features, pred_boxes = self.transformer_step(src_seq, tgt_seq)

            # Generator Forward
            pred_imgs = self.generator_step(features[:-1].detach(), tgt_images[1:-1], src_seq[1:-1], self.g_opt)

        # Increase phase step (for logging)
        self.step[self.phase] += 1

        return {'images': pred_imgs, 'gt': tgt_images, 'pred_boxes': pred_boxes, 'gt_boxes': boxes[1:]}

    def transformer_step(self, src, tgt, opt=None):
        # Transformer decoder output is standardized
        features, logits, boxes = self.img_transformer(src, tgt[:-1])

        tgt_features, tgt_boxes = split_list(tgt[1:], 4)

        # Compute losses
        cls_loss = self.classification_loss(logits, src[1:])
        feature_loss = self.reconstruction_loss(features, tgt_features)
        box_loss = self.box_loss(boxes, tgt_boxes)

        # cls_loss *= 0.1

        self.writer.log_metric(self.phase+'/cls_loss', cls_loss, self.step[self.phase])
        self.writer.log_metric(self.phase+'/feat_loss', feature_loss, self.step[self.phase])
        self.writer.log_metric(self.phase+'/box_loss', box_loss, self.step[self.phase])


        trf_loss = box_loss + feature_loss

        # probs = self.img_transformer.get_probs(logits)

        trf_acc = accuracy(
            rearrange(logits, 'seq b cl -> (seq b) cl'), 
            rearrange(src[1:], 'seq b -> (seq b)'), 
            topk=(1,)
        )

        # Log metrics
        self.writer.log_metric(self.phase+'/loss_trf', trf_loss, self.step[self.phase])
        self.writer.log_metric(self.phase+'/acc_trf', trf_acc, self.step[self.phase])

        # Backward
        if self.phase == 'train':
            # Clip Gradients
            # torch.nn.utils.clip_grad_norm_(self.img_transformer.parameters(), max_norm=1)
            opt.zero_grad()
            self.manual_backward(trf_loss)
            opt.step()

        return features, boxes

    def generator_step(self, feature_vec, tgt_images, labels, opt=None):

        # adversarial loss is binary cross-entropy
        batch_size = np.prod(feature_vec.shape[:2])
        real = torch.ones(batch_size, 1, device=self.args.device)

        feature_vec = rearrange(feature_vec, 'seq b emb -> (seq b) emb')

        with torch.no_grad():
            predictions = self.img_gen(feature_vec)

        tgt_images = rearrange(tgt_images, 'seq b c h w -> (seq b) c h w')
        gen_loss = self.reconstruction_loss(predictions, tgt_images)

        self.writer.log_metric(self.phase+'/loss_gen', gen_loss, self.step[self.phase])

        # if self.phase == 'train':
        #     opt.zero_grad()
        #     self.manual_backward(gen_loss)
        #     opt.step()

        return predictions

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def prob_match_loss(self, outputs, targets):

        criterion = nn.KLDivLoss(log_target=True)

        targets = F.log_softmax(targets, dim=2)
        outputs = F.log_softmax(outputs, dim=2)

        outputs = rearrange(outputs, 'seq b emb -> (seq b) emb')
        targets = rearrange(targets, 'seq b emb -> (seq b) emb')

        loss = criterion(outputs, targets)
        return loss

    def reconstruction_loss(self, outputs, targets):

        criterion = nn.MSELoss()

        loss = criterion(outputs, targets)

        # Take the mean over all dims but the sequence dim
        # Then, sum per sequence, to avoid the model to generate a result
        # mediated over sequence elements
        return loss

    def box_loss(self, outputs, targets):
        criterion = nn.MSELoss()
        loss = criterion(outputs, targets)
        return loss


    def gen_loss(self, outputs, targets):

        criterion = nn.L1Loss()

        # targets = targets.reshape(targets.shape[0] * targets.shape[1], -1).to(self.args.device)
        # outputs = outputs.reshape(targets.shape).to(self.args.device)

        # Normalize output
        # outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())

        loss = criterion(outputs, targets)
        # loss = loss.mean()
        return loss

    def classification_loss(self, outputs, targets):

        criterion = nn.CrossEntropyLoss()

        targets = rearrange(targets, 'seq b -> (seq b)')
        outputs = rearrange(outputs, 'seq b emb -> (seq b) emb')

        loss = criterion(outputs, targets)
        return loss 

    def training_epoch_end(self, outputs):
        if (self.current_epoch + 1) % self.args.log_every_n_steps == 0:
            self.writer.custom_histogram_adder(self.named_parameters(), self.current_epoch)

            self.writer.log_images(outputs, self.current_epoch, self.phase)


            # Zero metrics
            # phase_metrics = {name: 0.0 for name in self.metric_names}
            # self.metrics = { p: phase_metrics for p in ['train', 'val', 'test'] }
            # torch.save({
            #         'epoch': self.current_epoch,
            #         'model' : self.state_dict(),
            #         't_opt' : self.t_opt.state_dict(),
            #         'g_opt' : self.g_opt.state_dict()
            #         }, os.path.join(self.args.weight_dir, 'checkpoint.pt'))


    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        # return super().validation_epoch_end(outputs)
        if (self.current_epoch + 1) % self.args.log_every_n_steps == 0:
            self.custom_histogram_adder(self.named_parameters(), self.current_epoch)

            self.writer.log_images(outputs, self.logger.experiment, self.current_epoch, self.phase)
      

    def configure_optimizers(self):
        if self.args.opt == 'adam':
            t_optimizer = torch.optim.Adam(self.img_transformer.parameters(), lr=self.args.t_lr, betas=(0.9, 0.98), eps=1e-9)
            g_optimizer = torch.optim.Adam(self.img_gen.parameters(), lr=self.args.g_lr, betas=(0.9, 0.98), eps=1e-9)
            d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.d_lr)
        elif self.args.opt == 'sgd':
            t_optimizer = torch.optim.SGD(self.img_transformer.parameters(), lr=self.args.t_lr)
            g_optimizer = torch.optim.SGD(self.img_gen.parameters(), lr=self.args.g_lr)
            d_optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=self.args.d_lr)
        else:
            raise Exception(f'Optimizer {self.args.type} not supported')

        # t_scheduler = torch.optim.lr_scheduler.StepLR(t_optimizer, 1.0, gamma=0.95)
        g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, 1.0, gamma=0.95)
        d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, 1.0, gamma=0.95)


        return [t_optimizer, g_optimizer, d_optimizer], [g_scheduler, d_scheduler]