
import pytorch_lightning as pl
from log_utils import get_data_stats, img_to_PIL, log_metric, log_prediction, save_weights, compare_weights
import torch
from torch import nn
import torch.nn.functional as F
from datetime import datetime
import os
import torchvision.transforms as T
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from data_processing import get_embedding_from_vocab, get_succession, get_target_images, append_tokens, process_labels
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

current_date = datetime.now()

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
        self.phase = None

        # Variables for logging
        self.step = {
            'train': 1,
            'val': 1,
            'test': 1
        }

        self.args = args
        
        # Transformer
        self.img_transformer = img_transformer
        self.img_gen = img_gen
        self.discriminator = disc

        # self.init_weights()

    def forward(self, src, targets):

        out = self.img_transformer(src, targets)

        return out

    def training_step(self, batch, batch_idx, optimizer_idx):

        self.phase = 'train'

        self.tb_writer = SummaryWriter(os.path.join('tb_logs', wandb.run.name, f'{current_date.year}-{current_date.month}-{current_date.day}-{current_date.hour}h{current_date.minute}'))
        
        # TODO Parametrize
        log_weights_change = False

        # Define Optimizers
        self.t_opt, g_opt, d_opt = self.optimizers()
        t_sch, g_sch, d_sch = self.lr_schedulers()

        # Data loading
        original_images, labels = batch
        images = original_images.view(self.args.model.seq_len, self.args.model.seq_bs, self.args.data.n_channels, self.img_transformer.image_size[0], self.img_transformer.image_size[1])

        # Data Processing
        # tgt = [<SOS>, [embeddings], <EOS> (, [<PAD>] ) ]
        labels = labels.reshape(self.args.model.seq_bs, self.args.model.seq_len)
        src_seq = process_labels(labels, 14, 15)
        tgt_images = get_target_images(images, tokens['tgt'])
        vocab = torch.load('vocab.pth')
        tgt_embeddings = get_embedding_from_vocab(src_seq, vocab)

        if log_weights_change:
            old_weights = save_weights(self.img_transformer)
        
        trf_loss, feature_vecs = self.transformer_step(src_seq, tgt_embeddings)

        # Optimize Transformer
        self.t_opt.zero_grad()
        self.manual_backward(trf_loss)
        self.t_opt.step()

        torch.nn.utils.clip_grad_norm_(self.img_transformer.parameters(), max_norm=1)

        if log_weights_change:
            new_weights = save_weights(self.img_transformer)
            compare_weights(self.current_epoch, old_weights, new_weights)

        #TODO Wrap in function
        self.step[self.phase] += 1
        # if self.global_step % self.args.trainer.log_every_n_steps == 0:
        # tgt_boxes = tgt_bboxes[1:-1].reshape(-1, 4)
        # boxes = bbox[:-1].reshape(-1, 4)

        # log_prediction(tgt_images[1:], None, out_images, None, self.args.data.n_channels, self.args.model.seq_bs, self.logger, title="Train Transformer Target and Ouptut")


    def discriminator_step(self, feature_vec, real_imgs):
        # Measure discriminator's ability to classify real from generated samples

        # Transformer forward
        # tgt[:-1] (shifted right because the transformer has to predict based on previous output)
        # FIXME Use same prediction set or run multiple forward?
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

    def transformer_step(self, src, tgt):

        # TODO Check if processed input is the same as original input (no alteration has been done)

        out_embeddings, out = self(src, tgt[:-1])
        
        # tgt[1:] (shifted left to compare the real sequences, without the <SOS>)
        # with torch.no_grad():
        out_classes = self.feature_extractor.linear(out_embeddings)
        tgt_features = self.feature_extractor.linear(tgt)
        with open(f'output/{wandb.run.name}.txt', 'w') as o:
            o.write(f'Input:\n{src}\n\Target:\n{tgt_features.argmax(2)}\n\nOutput:\n{out_classes.argmax(2)}\n')
            o.write(f'\n\nOut Embedding;\n{tgt_features}\n\nStats:\n{get_data_stats(tgt_features)}')

            acc_out = (tgt_features[1:].argmax(2) == out_classes.argmax(2)).sum() / (out_classes.shape[0] * out_classes.shape[1])
            log_metric(self, self.phase+'/acc/out', acc_out, self.step[self.phase])
        
        cls_loss = self.classification_loss(nn.CrossEntropyLoss(), out_classes, src[1:])
        emb_loss = self.prob_match_loss(self.criterion, out_classes, tgt_features[1:])

        log_metric(self, self.phase+'/loss/emb', emb_loss, self.step[self.phase])
        log_metric(self, self.phase+'/loss/cls', cls_loss, self.step[self.phase])
        trf_loss = emb_loss + cls_loss
        log_metric(self, self.phase+'/loss/trf', trf_loss, self.step[self.phase])

        # Exclude SOS and EOS
        # box_loss = self.reconstruction_loss(self.criterion, bbox[:-1], tgt_bboxes[1:-1])
        # wandb.log({'gt': wandb.Image(img_to_PIL(make_grid(tgt_images[1:].reshape(40,1,32,32)))[1], caption="GT")})
        # wandb.log({'pred': wandb.Image(img_to_PIL(make_grid(out_imgs.repeat(1,3,1,1)))[1], caption="Pred")})
        # Compute loss
        # wandb.log({"box_loss": box_loss})

        return  trf_loss, out_classes # + box_loss

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def prob_match_loss(self, criterion, outputs, targets):

        targets = F.softmax(targets, dim=2)
        outputs = F.log_softmax(outputs, dim=2)

        outputs = outputs.reshape(40, -1)
        targets = targets.reshape(40, -1)

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
        outputs = outputs.reshape(-1, 16)

        loss = criterion(outputs, targets)
        return loss 

    def custom_histogram_adder(self):
        for name,params in self.named_parameters():
            self.logger.experiment.add_histogram(name,params,self.current_epoch)

    def training_epoch_end(self,outputs):
        self.custom_histogram_adder()
        os.makedirs('weights', exist_ok=True)
        torch.save({
                'epoch': self.current_epoch,
                'model' : self.state_dict(),
                'opt' : self.t_opt.state_dict()
                }, os.path.join('weights', 'checkpoint.pt'))


    def validation_step(self, batch, batch_idx):
        self.phase = 'val'
        # Data loading
        original_images, labels = batch
        images = original_images.view(self.args.model.seq_len, self.args.model.seq_bs, self.args.data.n_channels, self.img_transformer.image_size[0], self.img_transformer.image_size[1])

        # Data Processing
        # tgt = [<SOS>, [embeddings], <EOS> (, [<PAD>] ) ]
        labels = labels.reshape(self.args.model.seq_bs, self.args.model.seq_len)
        src_seq = process_labels(labels, 14, 15)
        tgt_images = get_target_images(images, tokens['tgt'])
        vocab = torch.load('vocab.pth')
        tgt_embeddings = get_embedding_from_vocab(src_seq, vocab)
        
        trf_loss, feature_vecs = self.transformer_step(src_seq, tgt_embeddings)

        self.step[self.phase] += 1
        # if self.global_step % self.args.trainer.log_every_n_steps == 0:
        #     target_imgs = target_imgs.view(target_imgs.shape[0] * target_imgs.shape[1], self.args.data.n_channels, target_imgs.shape[3], target_imgs.shape[4])
        #     log_prediction(target_imgs, predictions, self.logger, title="Validation Transformer Target and Ouptut")
        return trf_loss

    # def test_step(self, batch, batch_idx):
    #     self.phase = 'test'
    #     self.step[self.phase] += 1
        

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