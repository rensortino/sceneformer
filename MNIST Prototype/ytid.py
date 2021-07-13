
from data_processing import extract_features, normalize
import numpy as np
from layers import NoamOpt
import pytorch_lightning as pl
from torchvision.utils import save_image
from log_utils import get_data_stats, img_to_PIL, log_metric, log_prediction, save_weights, compare_weights
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
                feature_extractor,
                example_input,
                args):
        super(YTID, self).__init__()
        self.hparams.update(hparams)
        self.automatic_optimization = False # disable automatic calling of backward()
        self.max_seq_len = args.data.max_seq_len
        self.example_input_array = example_input
        self.feature_extractor = feature_extractor
        self.phase = None

        # Variables for logging
        self.step = { p: 1 for p in ['train', 'val', 'test'] }

        self.args = args
        
        # Transformer
        self.img_transformer = img_transformer
        self.img_gen = img_gen
        self.discriminator = disc

        # Transformer Optimizer
        warmup = 400
        # self.t_opt = NoamOpt(img_transformer.emb_size, 1, warmup,
        #     torch.optim.Adam(img_transformer.parameters(), lr=args.optimizer.t_lr, betas=(0.9, 0.98), eps=1e-9))
        self.t_opt = torch.optim.Adam(img_transformer.parameters(), lr=args.optimizer.t_lr, betas=(0.9, 0.98), eps=1e-9)

        self.init_t_weights()
        self.init_g_weights()

    def forward(self, src, targets):

        img_features = self.img_transformer(src, targets)

        # predictions = self.img_gen(img_features)
        predictions = img_features

        return predictions

    def init_t_weights(self):
        for n, p in self.img_transformer.named_parameters():
            if p.dim() > 1 and 'tgt_embedding' not in n:
                nn.init.xavier_uniform(p)

    def init_g_weights(self):
        for n, p in self.img_gen.named_parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform(p)

    def training_step(self, batch, batch_idx):

        self.phase = 'train'

        # Define Optimizers
        self.g_opt, self.d_opt = self.optimizers()
        g_sch, d_sch = self.lr_schedulers()

        # Data loading
        images, src_seq, tgt_images = batch
        # with torch.no_grad():
        tgt_seq = extract_features(self.feature_extractor, tgt_images)
        
        # Transformer Forward
        feature_vecs = self.transformer_step(src_seq, tgt_seq, self.t_opt)
        # classes = self.feature_extractor.linear(feature_vecs)

        # Discriminator Forward
        # self.discriminator_step(feature_vecs.detach(), images, self.d_opt)

        # Generator Forward
        pred_imgs = self.generator_step(feature_vecs.detach(), tgt_images[1:], self.g_opt)
        # pred_imgs = images.unsqueeze(1)

        # Increase phase step (for logging)
        self.step[self.phase] += 1

        return {'images': pred_imgs, 'gt': tgt_images}

    # def validation_step(self, batch, batch_idx):

    #     self.phase = 'val'
        
    #     # Data loading
    #     images, src_seq, tgt_seq = batch
        
    #     # Transformer Forward
    #     feature_vecs = self.transformer_step(src_seq, tgt_seq, self.t_opt)

    #     # Discriminator Forward
    #     self.discriminator_step(feature_vecs.detach(), images, self.d_opt)

    #     # Generator Forward
    #     pred_imgs = self.generator_step(feature_vecs.detach(), src_seq[1:], self.g_opt)
        
    #     # Log generated images
    #     _, pil_img = img_to_PIL(pred_imgs)
    #     pil_img = pil_img.resize((64,64))
    #     wandb.log({self.phase+'/out_image': wandb.Image(pil_img)})

    #     # Increase phase step (for logging)
    #     self.step[self.phase] += 1

    def transformer_step(self, src, tgt, opt):

        # TODO Check if processed input is the same as original input (no alteration has been done)

        # Transformer decoder output is standardized
        trf_out = self.img_transformer(src, tgt[:-1])

        # Compute losses
        # cls_loss = self.classification_loss(out_classes, src[1:])
        trf_loss = self.reconstruction_loss(trf_out, tgt[1:])

        # Log metrics
        log_metric(self, self.phase+'/loss_trf', trf_loss, self.step[self.phase], True)

        # Backward
        if self.phase == 'train':
            # Clip Gradients
            # torch.nn.utils.clip_grad_norm_(self.img_transformer.parameters(), max_norm=1)
            opt.zero_grad()
            self.manual_backward(trf_loss)
            opt.step()

        return trf_out

    def discriminator_step(self, feature_vecs, real_imgs, opt):

        # Generate fake samples
        predictions = self.img_gen(feature_vecs)

        # Normalize to [-1, 1] range
        tgt_imgs = 2 * ((real_imgs - real_imgs.min()) / real_imgs.max() - real_imgs.min()) - 1

        # how well can it label as real?
        real = torch.ones(tgt_imgs.size(0), 1, device=self.args.device)
        real_loss = self.adversarial_loss(self.discriminator(tgt_imgs), real)

        # how well can it label as fake?
        fake = torch.zeros(predictions.size(0), 1, device=self.args.device)
        fake_loss = self.adversarial_loss(
            self.discriminator(predictions.detach()), fake)

        disc_loss = real_loss + fake_loss

        log_metric(self, self.phase+'/loss_real', real_loss, self.step[self.phase])
        log_metric(self, self.phase+'/loss_fake', fake_loss, self.step[self.phase])
        log_metric(self, self.phase+'/loss_disc', disc_loss, self.step[self.phase], True)

        if self.phase == 'train':
            opt.zero_grad()
            self.manual_backward(disc_loss)
            opt.step()

    def generator_step(self, feature_vec, tgt_images, opt):

        # adversarial loss is binary cross-entropy
        batch_size = np.prod(feature_vec.shape[:2])
        real = torch.ones(batch_size, 1, device=self.args.device)

        predictions = self.img_gen(feature_vec)
        
        # Normalize like the original MNIST images
        # predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())
        # pred_vectors = self.feature_extractor(predictions)
        # pred_classes = self.feature_extractor.linear(pred_vectors)
        # tgt_vectors = self.feature_extractor(tgt_images[1:].reshape(-1,1,32,32), True)

        # gen_loss = self.classification_loss(pred_classes, labels)
        # gen_loss = self.adversarial_loss(self.discriminator(predictions), real)
        tgt_images = normalize(tgt_images, -1, 1)
        gen_loss = self.gen_loss(predictions[-2], tgt_images[-2])
        log_metric(self, self.phase+'/loss_gen', gen_loss, self.step[self.phase], True)

        if self.phase == 'train':
            opt.zero_grad()
            self.manual_backward(gen_loss)
            opt.step()

        return predictions

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def prob_match_loss(self, outputs, targets):

        criterion = nn.KLDivLoss()

        targets = F.softmax(targets, dim=2)
        outputs = F.log_softmax(outputs, dim=2)

        batch_size = self.args.model.seq_bs * (self.args.model.seq_len + 1)

        outputs = outputs.reshape(batch_size, -1)
        targets = targets.reshape(batch_size, -1)

        loss = criterion(outputs, targets)
        return loss

    def reconstruction_loss(self, outputs, targets):

        criterion = nn.MSELoss()

        # targets = targets.reshape(targets.shape[0] * targets.shape[1], -1).to(self.args.device)
        # outputs = outputs.reshape(targets.shape).to(self.args.device)

        # Normalize output
        # outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())

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
        # FIXME Change
        if (self.current_epoch + 1) % self.args.trainer.log_every_n_steps == 0:
            self.custom_histogram_adder()

            out_image = outputs[-1]['images']
            gt = outputs[-1]['gt']

            # Log generated images
            _, pil_img = img_to_PIL(out_image[-2])
            pil_img = pil_img.resize((256,256))
            _, pil_gt = img_to_PIL(gt[-2])
            pil_gt = pil_gt.resize((256,256))
            wandb.log({self.phase+'/out_image': wandb.Image(pil_img)})
            wandb.log({self.phase+'/gt': wandb.Image(pil_gt)})

            grid = make_grid(out_image[-2,:6])
            self.logger.experiment.add_image('out images', grid, self.step[self.phase])


            # Zero metrics
            # phase_metrics = {name: 0.0 for name in self.metric_names}
            # self.metrics = { p: phase_metrics for p in ['train', 'val', 'test'] }
            # torch.save({
            #         'epoch': self.current_epoch,
            #         'model' : self.state_dict(),
            #         't_opt' : self.t_opt.state_dict(),
            #         'g_opt' : self.g_opt.state_dict()
            #         }, os.path.join(self.args.trainer.weight_dir, 'checkpoint.pt'))



    # def test_step(self, batch, batch_idx):
    #     self.phase = 'test'
    #     self.step[self.phase] += 1
        

    def configure_optimizers(self):
        if self.args.optimizer.type == 'Adam':
            # t_optimizer = torch.optim.Adam(self.img_transformer.parameters(), lr=self.args.optimizer.t_lr, betas=(0.9, 0.98), eps=1e-9)
            g_optimizer = torch.optim.SGD(self.img_gen.parameters(), lr=self.args.optimizer.g_lr)#, betas=(0.9, 0.98), eps=1e-9)
            d_optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=self.args.optimizer.d_lr)
        else:
            raise Exception(f'Optimizer {self.args.optimizer.type} not supported')

        # t_scheduler = torch.optim.lr_scheduler.StepLR(t_optimizer, 1.0, gamma=0.95)
        g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, 1.0, gamma=0.95)
        d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, 1.0, gamma=0.95)


        return [g_optimizer, d_optimizer], [g_scheduler, d_scheduler]