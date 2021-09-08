
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from data_processing import accuracy, extract_features, normalize, split_trf_output
import numpy as np
from layers import NoamOpt
import pytorch_lightning as pl
from torchvision.utils import save_image
from log_utils import get_data_stats, img_to_PIL, log_images, log_metric, log_prediction, save_weights, compare_weights
import torch
from torch import nn
import torch.nn.functional as F
import os
from torchvision.utils import make_grid
import wandb

class YTID(pl.LightningModule):
    
    def __init__(self,
                hparams,
                img_transformer: nn.Module,
                img_gen: nn.Module,
                disc: nn.Module,
                backbone,
                token_container,
                example_input,
                args):
        super(YTID, self).__init__()
        self.hparams.update(hparams)
        self.automatic_optimization = False # disable automatic calling of backward()
        self.max_seq_len = args.max_seq_len
        self.example_input_array = example_input
        self.backbone = backbone
        self.token_container = token_container
        self.phase = None

        # Variables for logging
        self.step = { p: 1 for p in ['train', 'val', 'test'] }

        self.args = args
        
        # Transformer
        self.img_transformer = img_transformer
        self.img_gen = img_gen
        self.discriminator = disc

        
        self.t_opt = torch.optim.Adam(img_transformer.parameters(), lr=args.t_lr, betas=(0.9, 0.98), eps=1e-9)

    def forward(self, src, targets):

        # FIXME Transformer output should be handled
        img_features = self.img_transformer(src, targets)

        # predictions = self.img_gen(img_features)
        predictions = img_features

        return predictions

    def training_step(self, batch, batch_idx):

        self.phase = 'train'

        # Define Optimizers
        self.g_opt, self.d_opt = self.optimizers()

        # Data loading
        images, src_seq, tgt_images, boxes = batch
        with torch.no_grad():
            tgt_features = extract_features(self.backbone, tgt_images)

        tgt_seq = torch.cat((tgt_features, boxes), dim=2)
        
        # Transformer Forward
        features, boxes = self.transformer_step(src_seq, tgt_seq, self.t_opt)

        # Discriminator Forward
        # self.discriminator_step(features.detach(), images, self.d_opt)

        # Generator Forward
        pred_imgs = self.generator_step(features[:-1].detach(), tgt_images[1:-1], src_seq[1:-1], self.g_opt)
        # pred_imgs = images.unsqueeze(1)

        # Increase phase step (for logging)
        self.step[self.phase] += 1

        return {'images': pred_imgs, 'gt': tgt_images}

    def validation_step(self, batch, batch_idx):

        self.phase = 'val'
        
        # Data loading
        images, src_seq, tgt_images, boxes = batch
        with torch.no_grad():
            tgt_features = extract_features(self.backbone, tgt_images)

            tgt_seq = torch.cat((tgt_features, boxes), dim=2)

            features, boxes = self.transformer_step(src_seq, tgt_seq)

            # Discriminator Forward
            # self.discriminator_step(features.detach(), images, self.d_opt)

            # Generator Forward
            pred_imgs = self.generator_step(features[:-1].detach(), tgt_images[1:-1], src_seq[1:-1], self.g_opt)

        # Increase phase step (for logging)
        self.step[self.phase] += 1

        return {'images': pred_imgs, 'gt': tgt_images}

    def transformer_step(self, src, tgt, opt=None):
        # Transformer decoder output is standardized
        features, logits, bbox = self.img_transformer(src, tgt[:-1])

        # Compute losses
        cls_loss = self.classification_loss(logits, src[1:])
        feature_loss = self.reconstruction_loss(features, tgt[1:, :, :-4])
        box_loss = self.reconstruction_loss(bbox, tgt[1:, :, -4:])

        trf_loss = 2*cls_loss + box_loss + 2*feature_loss

        # probs = self.img_transformer.get_probs(logits)

        #TODO Parametrize
        trf_acc = accuracy(logits.reshape(-1, 12), src[1:].reshape(-1), topk=(1,))[0].item()

        # Log metrics
        log_metric(self, self.phase+'/loss_trf', trf_loss, self.step[self.phase], True)
        log_metric(self, self.phase+'/acc_trf', trf_acc, self.step[self.phase], True)

        # Backward
        if self.phase == 'train':
            # Clip Gradients
            # torch.nn.utils.clip_grad_norm_(self.img_transformer.parameters(), max_norm=1)
            opt.zero_grad()
            self.manual_backward(trf_loss)
            opt.step()

        return features, bbox

    def discriminator_step(self, feature_vecs, real_imgs, opt):

        # Generate fake samples
        predictions = self.img_gen(feature_vecs)
        predictions = predictions[:-1]
        batched_pred = predictions.reshape(np.prod(predictions.shape[:2]), *predictions.shape[2:])

        # Normalize to [-1, 1] range
        tgt_imgs = 2 * ((real_imgs - real_imgs.min()) / real_imgs.max() - real_imgs.min()) - 1

        # how well can it label as real?
        real = torch.ones(tgt_imgs.size(0), 1, device=self.args.device)
        real_loss = self.adversarial_loss(self.discriminator(tgt_imgs), real)

        # how well can it label as fake?
        fake = torch.zeros(batched_pred.size(0), 1, device=self.args.device)
        fake_loss = self.adversarial_loss(
            self.discriminator(batched_pred.detach()), fake)

        disc_loss = real_loss + fake_loss

        log_metric(self, self.phase+'/loss_real', real_loss, self.step[self.phase])
        log_metric(self, self.phase+'/loss_fake', fake_loss, self.step[self.phase])
        log_metric(self, self.phase+'/loss_disc', disc_loss, self.step[self.phase], True)

        if self.phase == 'train':
            opt.zero_grad()
            self.manual_backward(disc_loss)
            opt.step()

    def generator_step(self, feature_vec, tgt_images, labels, opt=None):

        # adversarial loss is binary cross-entropy
        batch_size = np.prod(feature_vec.shape[:2])
        real = torch.ones(batch_size, 1, device=self.args.device)

        feature_vec = feature_vec.reshape(-1, feature_vec.shape[2])

        with torch.no_grad():
            predictions = self.img_gen(feature_vec)

        tgt_images = tgt_images.reshape(-1, *tgt_images.shape[2:])
        gen_loss = self.reconstruction_loss(predictions, tgt_images)
        # gen_loss = self.adversarial_loss(self.discriminator(batched_pred), real)
        # tgt_images = normalize(tgt_images, -1, 1)
        # gen_loss = self.gen_loss(predictions[-2], tgt_images[-2])
        log_metric(self, self.phase+'/loss_gen', gen_loss, self.step[self.phase], True)

        # if self.phase == 'train':
        #     opt.zero_grad()
        #     self.manual_backward(gen_loss)
        #     opt.step()

        return predictions

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def prob_match_loss(self, outputs, targets):

        criterion = nn.KLDivLoss()

        targets = F.softmax(targets, dim=2)
        outputs = F.log_softmax(outputs, dim=2)

        batch_size = self.args.seq_bs * (self.args.seq_len + 1)

        outputs = outputs.reshape(batch_size, -1)
        targets = targets.reshape(batch_size, -1)

        loss = criterion(outputs, targets)
        return loss

    def reconstruction_loss(self, outputs, targets):

        criterion = nn.MSELoss()

        loss = criterion(outputs, targets)

        # Take the mean over all dims but the sequence dim
        # Then, sum per sequence, to avoid the model to generate a result
        # mediated over sequence elements
        return loss


    def gen_loss(self, outputs, targets):

        criterion = nn.MSELoss()

        # targets = targets.reshape(targets.shape[0] * targets.shape[1], -1).to(self.args.device)
        # outputs = outputs.reshape(targets.shape).to(self.args.device)

        # Normalize output
        # outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())

        loss = criterion(outputs, targets)
        # loss = loss.mean()
        return loss

    def classification_loss(self, outputs, targets):

        criterion = nn.CrossEntropyLoss()

        targets = targets.reshape(-1)
        outputs = outputs.reshape(targets.shape[0], -1)

        loss = criterion(outputs, targets)
        return loss 

    def custom_histogram_adder(self):
        for name,params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def training_epoch_end(self, outputs):
        if (self.current_epoch + 1) % self.args.log_every_n_steps == 0:
            self.custom_histogram_adder()

            log_images(outputs, self.logger.experiment, self.current_epoch, self.phase)


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
            self.custom_histogram_adder()

            log_images(outputs, self.logger.experiment, self.current_epoch, self.phase)
      

    def configure_optimizers(self):
        if self.args.opt == 'Adam':
            # t_optimizer = torch.optim.Adam(self.img_transformer.parameters(), lr=self.args.t_lr, betas=(0.9, 0.98), eps=1e-9)
            g_optimizer = torch.optim.SGD(self.img_gen.parameters(), lr=self.args.g_lr)#, betas=(0.9, 0.98), eps=1e-9)
            d_optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=self.args.d_lr)
        else:
            raise Exception(f'Optimizer {self.args.type} not supported')

        # t_scheduler = torch.optim.lr_scheduler.StepLR(t_optimizer, 1.0, gamma=0.95)
        g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, 1.0, gamma=0.95)
        d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, 1.0, gamma=0.95)


        return [g_optimizer, d_optimizer], [g_scheduler, d_scheduler]