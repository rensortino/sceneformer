
import pytorch_lightning as pl
from log_utils import log_prediction, save_weights, compare_weights
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from data_processing import get_succession, get_target_images, append_tokens, process_labels
import wandb

TOKENS = {
        "SOS": 0.0,
        "EOS": 1.0,
        "PAD": 0.5
    }

class YTID(pl.LightningModule):
    
    def __init__(self,
                img_transformer: nn.Module,
                disc: nn.Module,
                args,
                criterion):
        super(YTID, self).__init__()
        self.automatic_optimization = False # disable automatic calling of backward()
        self.max_seq_len = args.model.max_seq_len
        self.criterion = criterion

        # Variables for logging
        self.train_step = 1
        self.val_step = 1

        self.args = args
        
        # Transformer
        self.img_transformer = img_transformer
        self.discriminator = disc

        # self.init_weights()

    def forward(self, src, targets):

        out = self.img_transformer(src, targets)
        # out_imgs, image_vector, bbox = self.img_transformer(src, targets)

        return out
        # return out_imgs, image_vector, bbox

    def training_step(self, batch, batch_idx, optimizer_idx):

        # TODO Parametrize
        log_weights_change = True

        # Define Optimizers
        d_opt, t_opt = self.optimizers()
        _, t_sch = self.lr_schedulers()

        # Data loading
        original_images, labels = batch
        with open('batch.txt', 'w') as f:
            f.write(f'{original_images}\n {labels}')
        images = original_images.view(self.args.model.seq_len, self.args.model.seq_bs, self.args.data.n_channels, self.img_transformer.image_size[0], self.img_transformer.image_size[1])

        if log_weights_change:
            old_weights = save_weights(self.img_transformer)
        
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

        t_loss = self.transformer_step(labels, images)

        t_opt.zero_grad()
        self.manual_backward(t_loss)

        torch.nn.utils.clip_grad_norm_(self.img_transformer.parameters(), max_norm=1)

        #g_opt.step()
        t_opt.step()
        # if self.current_epoch % 5 == 0:
        #     t_sch.step()

        if log_weights_change:
            new_weights = save_weights(self.img_transformer)
            compare_weights(self.current_epoch, old_weights, new_weights)

        self.log('t_loss', t_loss, prog_bar=True)


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
        # #####################
        # Optimize Generator #
        # #####################

        # adversarial loss is binary cross-entropy
        sl, bs, c, h, w = tgt_imgs.shape
        tgt_imgs = tgt_imgs.reshape(sl * bs, c, h, w)
        real = torch.ones(tgt_imgs.size(0), 1, device=self.args.device)

        g_loss = self.adversarial_loss(self.discriminator(self.predictions), real)

        wandb.log({"g_loss": g_loss})

        #self.log_dict({'t_loss': mse_loss, 'g_loss': t_adv_loss, 'd_loss': d_loss, 'd_real_loss': real_loss, 'd_fake_loss': fake_loss}, prog_bar=True)
        return g_loss

    def transformer_step(self, labels, images):

        #padded_tgt = get_padded_tgt(targets)

        # tgt = [<SOS>, [embeddings], <EOS> (, [<PAD>] ) ]
        labels = labels.reshape(self.args.model.seq_bs, self.args.model.seq_len)
        in_seq = process_labels(labels, 14, 15)
        tgt_images = get_target_images(images)
        # tgt_seq = process_labels(labels, 37, 38)

        # one_hot_targets = F.one_hot(tgt_seq, 512).float()

        # TODO Check if processed input is the same as original input (no alteration has been done)
        # self.predictions, image_vectors, bbox = self(in_seq, targets[:-1])

        out = self(in_seq, tgt_images[:-1])
        with open(f'output/{wandb.run.name}.txt', 'w') as o:
            o.write(f'Input:\t{in_seq}\n\nOutput:\t{out.argmax(2)}\n')
        # self.predictions, image_vectors, bbox = self(in_seq, one_hot_targets[:-1])

        # tgt_vectors = targets[:,:,:-4]
        # tgt_bboxes = targets[:,:,-4:]

        # TODO Parametrize
        # out_imgs = tgt_vectors[1:].reshape(-1, 1024).to(self.args.device)
        # logits = self.classifier(out_imgs)
        # t_loss = self.classification_loss(nn.CrossEntropyLoss(), logits, in_seq[1:])
        
        # tgt[1:] (shifted left to compare the real sequences, without the <SOS>)
        # t_loss = self.reconstruction_loss(self.criterion, out, tgt_images)
        t_loss = self.prob_match_loss(self.criterion, out, in_seq[1:])
        # Exclude SOS and EOS
        # box_loss = self.reconstruction_loss(self.criterion, bbox[:-1], tgt_bboxes[1:-1])

        # Compute loss
        wandb.log({"t_loss": t_loss})
        # wandb.log({"box_loss": box_loss})

        #TODO Wrap in function
        self.train_step += 1
        # if self.global_step % self.args.trainer.log_every_n_steps == 0:
        # target_imgs = out_imgs.view(out_imgs.shape[0] * out_imgs.shape[1], self.args.data.n_channels, out_imgs.shape[2], out_imgs.shape[3])
        # tgt_boxes = tgt_bboxes[1:-1].reshape(-1, 4)
        # boxes = bbox[:-1].reshape(-1, 4)
        # log_prediction(target_imgs, tgt_boxes, self.predictions, boxes, self.logger, title="Train Transformer Target and Ouptut")

        return  t_loss # + box_loss

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def prob_match_loss(self, criterion, outputs, targets):

        targets = targets.reshape(-1)
        outputs = outputs.reshape(-1,16)

        loss = criterion(outputs, targets)
        return loss


    def reconstruction_loss(self, criterion,  outputs, targets):

        targets = targets.reshape(targets.shape[0] * targets.shape[1], -1).to(self.args.device)
        outputs = outputs.reshape(targets.shape).to(self.args.device)

        loss = criterion(outputs, targets)        
        return loss

    def classification_loss(self, criterion, outputs, targets):

        targets = targets.reshape(-1)

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
    #    images = images.view(self.args.model.seq_len, self.args.model.seq_bs, self.args.data.n_channels, self.args.data.img_h, self.args.data.img_w)
    #    step_loss, predictions, target_imgs = self.transformer_step(labels, images)
    #    
    #    # Logging
    #    #self.logger.experiment.log({'Val t_loss': step_loss, 'val_step': self.val_step})
    #    self.logger.experiment.add_scalar('Loss/Val', step_loss, self.val_step)
    #    self.log("Default Transformer Loss", step_loss)
#
    #    self.val_step += 1
    #    if self.global_step % self.args.trainer.log_every_n_steps == 0:
    #        target_imgs = target_imgs.view(target_imgs.shape[0] * target_imgs.shape[1], self.args.data.n_channels, target_imgs.shape[3], target_imgs.shape[4])
    #        log_prediction(target_imgs, predictions, self.logger, title="Validation Transformer Target and Ouptut")
    #    return step_loss
#
    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        if self.args.optimizer.type == 'Adam':
            # FIXME Fix optimizers
            #g_optimizer = torch.optim.Adam(self.img_gen.parameters(), lr=self.args.optimizer.g_lr)
            d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.optimizer.d_lr)
            t_optimizer = torch.optim.Adam(self.img_transformer.parameters(), lr=self.args.optimizer.t_lr)
        else:
            raise Exception(f'Optimizer {self.args.optimizer.type} not supported')

        #g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, 1.0, gamma=0.95)
        d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, 1.0, gamma=0.95)
        t_scheduler = torch.optim.lr_scheduler.StepLR(t_optimizer, 1.0, gamma=0.95)


        return [d_optimizer, t_optimizer], [d_scheduler, t_scheduler]