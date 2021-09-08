from torchvision.utils import make_grid
from torch.utils.data import DataLoader, random_split
from tokens import *
from torchvision import datasets
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from data_processing import extract_features, get_bboxes, get_img_grid, process_labels

class GenericDataModule(pl.LightningDataModule):
    def __init__(self, args, image_size, backbone, token_container, data_dir="data", debug=False):
        super(GenericDataModule, self).__init__()
        self.data_dir = data_dir
        self.args = args
        self.image_size = image_size
        self.debug = debug
        self.backbone = backbone
        self.token_container = token_container

    def grid_collate_fn(self, batch):
        # tgt = [<SOS>, [embeddings], <EOS> (, [<PAD>] ) ]
        images, labels = torch.utils.data._utils.collate.default_collate(batch)
        data = images.reshape(self.args.seq_len, self.args.seq_bs, self.args.data.n_channels, self.args.data.img_h, self.args.data.img_w)
        labels = labels.reshape(self.args.seq_len, self.args.seq_bs)

        src_seq = process_labels(labels, self.token_container.src_token['sos'], self.token_container.src_token['eos'])

        for i in range(data.shape[1]):
            img_grid = get_img_grid(data[:,i], self.args.data.n_channels)
            img_grid = img_grid.unsqueeze(1)
            if i == 0:
                img_grids = img_grid
            else:
                img_grids = torch.cat((img_grids, img_grid), dim=1)

        tgt_seq = torch.cat((self.token_container.sos_img_token, img_grids, self.token_container.eos_img_token))

        return images, src_seq, tgt_seq

    def bbox_collate_fn(self, batch):
        images, labels = torch.utils.data._utils.collate.default_collate(batch)
        data = images.reshape(self.args.seq_len, self.args.seq_bs, self.args.data.n_channels, self.args.data.img_h, self.args.data.img_w)
        labels = labels.reshape(self.args.seq_len, self.args.seq_bs)

        src_seq = process_labels(labels, self.token_container.src_token['sos'], self.token_container.src_token['eos'])

        for i in range(data.shape[1]):
            img_w_boxes = get_bboxes(data[:,i])
            img_w_boxes = img_w_boxes.unsqueeze(1)
            if i == 0:
                boxes = img_w_boxes
            else:
                boxes = torch.cat((boxes, img_w_boxes), dim=1)

        tgt_seq = torch.cat((self.token_container.sos_img_token, data, self.token_container.eos_img_token))
        tgt_boxes = torch.cat((self.token_container.sos_box_token, boxes, self.token_container.eos_box_token))

        return images, src_seq, tgt_seq, tgt_boxes

    def get_example_batch(self):
        src_seq, tgt_images, tgt_boxes = next(iter(DataLoader(self.train_dataset, batch_size=self.args.batch_size, drop_last=True, collate_fn=self.bbox_collate_fn)))[1:]
        tgt_feats = extract_features(self.backbone, tgt_images).cpu()
        tgt_seq = torch.cat((tgt_feats, tgt_boxes), dim=2)
        return src_seq, tgt_seq

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, persistent_workers=self.args.persistent_workers, drop_last=True, collate_fn=self.bbox_collate_fn)

    def val_dataloader(self):
        if not self.debug:
            return DataLoader(self.val_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=False, drop_last=True, collate_fn=self.bbox_collate_fn)

    def test_dataloader(self):
        if not self.debug:
            return DataLoader(self.test_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=False, drop_last=True, collate_fn=self.bbox_collate_fn)


class MNISTDataModule(GenericDataModule):

    def __init__(self, args, image_size, backbone, token_container, data_dir="data", debug=False):
        super(MNISTDataModule, self).__init__(args, image_size, backbone, token_container, data_dir, debug)

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        mnist = datasets.MNIST(download=False, train=True, root="data").data.float()
        self.transforms = T.Compose([ 
            T.Resize((self.image_size)),
            T.ToTensor()
        ])

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist = datasets.MNIST(self.data_dir, train=True, transform=self.transforms)
            self.train_dataset, self.val_dataset = random_split(mnist, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.MNIST(self.data_dir, train=False, transform=self.transforms)



class CIFAR10DataModule(GenericDataModule):
    def __init__(self, args, backbone, token_container, data_dir="data", debug=False):
        super(CIFAR10DataModule, self).__init__(args, 32, backbone, token_container, data_dir, debug)
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)


    def setup(self, stage=None):
        self.train_transforms = T.Compose(
            [
                # T.RandomCrop(32, padding=4),
                T.Resize((self.args.data.img_h, self.args.data.img_w)),
                # T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        self.test_transforms = T.Compose(
            [
                T.Resize((self.args.data.img_h, self.args.data.img_w)),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            cifar_train = datasets.CIFAR10(self.data_dir, download=True, train=True, transform=self.train_transforms)
            self.train_dataset, self.val_dataset = random_split(cifar_train, [46000, 4000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.CIFAR10(self.data_dir, download=True, train=False, transform=self.test_transforms)
