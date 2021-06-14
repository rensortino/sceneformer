
import pytorch_lightning as pl
from log_utils import img_to_PIL, log_prediction, save_weights, compare_weights
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import make_grid
from data_processing import get_succession, get_target_images, append_tokens, process_labels
import wandb

# FIXME DRY
tokens = {
    'src': {
        "SOS": 14,
        "EOS": 15,
        "PAD": 16
    },
    'tgt': {
        "SOS": 0.0,
        "EOS": 1.0,
        "PAD": 0.5 # TODO Modify 
    }
}

class YTID(pl.LightningModule):
    
    def __init__(self,
                img_transformer: nn.Module,
                img_gen: nn.Module,
                disc: nn.Module,
                feature_extractor,
                args,
                criterion):
        super(YTID, self).__init__()
        self.automatic_optimization = False # disable automatic calling of backward()
        self.max_seq_len = args.model.max_seq_len
        self.criterion = criterion
        self.feature_extractor = feature_extractor

        # Variables for logging
        self.train_step = 1
        self.val_step = 1

        self.args = args
        
        # Transformer
        self.img_transformer = img_transformer
        self.img_gen = img_gen
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
        t_opt, g_opt, d_opt = self.optimizers()
        t_sch, g_sch, d_sch = self.lr_schedulers()

        # Data loading
        original_images, labels = batch
        with open('batch.txt', 'w') as f:
            f.write(f'{original_images}\n {labels}')
        images = original_images.view(self.args.model.seq_len, self.args.model.seq_bs, self.args.data.n_channels, self.img_transformer.image_size[0], self.img_transformer.image_size[1])

        # Data Processing
        # tgt = [<SOS>, [embeddings], <EOS> (, [<PAD>] ) ]
        labels = labels.reshape(self.args.model.seq_bs, self.args.model.seq_len)
        src_seq = process_labels(labels, 14, 15)
        tgt_images = get_target_images(images, tokens['tgt'])

        if log_weights_change:
            old_weights = save_weights(self.img_transformer)
        
        t_loss, feature_vecs = self.transformer_step(src_seq, tgt_images)

        # Optimize Transformer
        t_opt.zero_grad()
        self.manual_backward(t_loss)
        t_opt.step()

        # d_loss = self.discriminator_step(feature_vecs.detach(), tgt_images[1:])

        # Optimize Discriminator
        # d_opt.zero_grad()
        # self.manual_backward(d_loss)
        # d_opt.step()


        # g_loss, out_images = self.generator_step(feature_vecs.detach(), src_seq[1:])

        # g_opt.zero_grad()
        # self.manual_backward(g_loss, retain_graph=True)

        torch.nn.utils.clip_grad_norm_(self.img_transformer.parameters(), max_norm=1)

        # g_opt.step()
        # if self.current_epoch % 5 == 0:
        #     t_sch.step()

        if log_weights_change:
            new_weights = save_weights(self.img_transformer)
            compare_weights(self.current_epoch, old_weights, new_weights)

        self.log('t_loss', t_loss, prog_bar=True)
        # self.log('g_loss', g_loss, prog_bar=True)
        # self.log('d_loss', d_loss, prog_bar=True)
        wandb.log({"t_loss": t_loss})#, "g_loss": g_loss, "d_loss": d_loss})

        #TODO Wrap in function
        self.train_step += 1
        # if self.global_step % self.args.trainer.log_every_n_steps == 0:
        # tgt_boxes = tgt_bboxes[1:-1].reshape(-1, 4)
        # boxes = bbox[:-1].reshape(-1, 4)

        # log_prediction(tgt_images[1:], None, out_images, None, self.args.data.n_channels, self.args.model.seq_bs, self.logger, title="Train Transformer Target and Ouptut")


    def discriminator_step(self, feature_vec, real_imgs):
        # Measure discriminator's ability to classify real from generated samples

        # Transformer forward
        # tgt[:-1] (shifted right because the transformer has to predict based on previous output)
        # FIXME Use same prediction set or run multiple forward?
        # self.predictions, self.image_vectors, self.bbox = self(in_seq, targets[:-1])
        predictions = self.img_gen(feature_vec)

        sl, bs, c, h, w = real_imgs.shape
        tgt_imgs = real_imgs.reshape(sl * bs, c, h, w)

        tgt_imgs = 2 * ((tgt_imgs - tgt_imgs.min()) / tgt_imgs.max() - tgt_imgs.min()) - 1

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

    def generator_step(self, feature_vec, labels):
        # #####################
        # Optimize Generator #
        # #####################

        # adversarial loss is binary cross-entropy
        batch_size = feature_vec.shape[0] * feature_vec.shape[1]
        real = torch.ones(batch_size, 1, device=self.args.device)

        predictions = self.img_gen(feature_vec)
        predictions = (predictions - predictions.min()) /  (predictions.max() - predictions.min())
        
        pred_vectors = self.feature_extractor(predictions)
            # tgt_vectors = self.feature_extractor(tgt_images[1:].reshape(-1,1,32,32), True)

        g_loss = self.classification_loss(nn.CrossEntropyLoss(), pred_vectors, labels)
        # g_loss = self.adversarial_loss(self.discriminator(predictions), real)

        wandb.log({"g_loss": g_loss})

        #self.log_dict({'t_loss': mse_loss, 'g_loss': t_adv_loss, 'd_loss': d_loss, 'd_real_loss': real_loss, 'd_fake_loss': fake_loss}, prog_bar=True)
        return g_loss, predictions

    def transformer_step(self, src_seq, tgt_images):

        # TODO Check if processed input is the same as original input (no alteration has been done)
        # self.predictions, image_vectors, bbox = self(in_seq, targets[:-1])

        out_vectors = self(src_seq, tgt_images[:-1])
        
        # tgt[1:] (shifted left to compare the real sequences, without the <SOS>)
        # t_loss = self.reconstruction_loss(self.criterion, out_imgs, tgt_images[1:])
        tgt_vectors = self.feature_extractor(tgt_images[1:].reshape(-1,1,32,32), True)
        # TODO Change variable names
        out_features = self.feature_extractor.linear(out_vectors)
        tgt_features = self.feature_extractor.linear(tgt_vectors)
        with open(f'output/{wandb.run.name}.txt', 'w') as o:
            o.write(f'Input:\t{src_seq}\n\Target:\t{tgt_features.reshape(5,8,16).argmax(1)}\n\nOutput:\t{out_features.argmax(2)}\n')
        
        t_loss = self.prob_match_loss(self.criterion, out_vectors, tgt_vectors)
        # t_loss = self.classification_loss(self.criterion, out_vectors, in_seq[1:])
        # t_loss = self.criterion(out_vectors.reshape(-1, 512), tgt_vectors)
        # Exclude SOS and EOS
        # box_loss = self.reconstruction_loss(self.criterion, bbox[:-1], tgt_bboxes[1:-1])
        # wandb.log({'gt': wandb.Image(img_to_PIL(make_grid(tgt_images[1:].reshape(40,1,32,32)))[1], caption="GT")})
        # wandb.log({'pred': wandb.Image(img_to_PIL(make_grid(out_imgs.repeat(1,3,1,1)))[1], caption="Pred")})
        # Compute loss
        # wandb.log({"box_loss": box_loss})

        return  t_loss, out_vectors # + box_loss

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def prob_match_loss(self, criterion, outputs, targets):

        targets = F.softmax(targets, dim=1)
        outputs = F.softmax(outputs, dim=1)

        outputs = outputs.reshape(40, 512)

        loss = criterion(outputs, targets)
        return loss


    def reconstruction_loss(self, criterion,  outputs, targets):

        targets = targets.reshape(targets.shape[0] * targets.shape[1], -1).to(self.args.device)
        outputs = outputs.reshape(targets.shape).to(self.args.device)

        # Normalize output
        outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())

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
            g_optimizer = torch.optim.Adam(self.img_gen.parameters(), lr=self.args.optimizer.g_lr)
            d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.optimizer.d_lr)
            t_optimizer = torch.optim.Adam(self.img_transformer.parameters(), lr=self.args.optimizer.t_lr)
        else:
            raise Exception(f'Optimizer {self.args.optimizer.type} not supported')

        g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, 1.0, gamma=0.95)
        d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, 1.0, gamma=0.95)
        t_scheduler = torch.optim.lr_scheduler.StepLR(t_optimizer, 1.0, gamma=0.95)


        return [t_optimizer, g_optimizer, d_optimizer], [t_scheduler, g_scheduler, d_scheduler]