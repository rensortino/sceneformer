from torchvision.utils import make_grid
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import pytorch_lightning as pl
import torch
import os
import torchvision.transforms as T
from data_processing import process_labels, get_embedding_from_vocab, build_vocab, wrong_process_labels

tokens = {
    'src': {
        "sos": 14,
        "eos": 15,
        "pad": 16
    },
    'tgt': {
        "sos": 0.0,
        "eos": 1.0
    }
}

class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, args, image_size, data_dir="data"):
        super(MNISTDataModule, self).__init__()
        self.data_dir = data_dir
        self.args = args
        self.image_size = image_size
        
        self.vocab = torch.load('vocab.pth')

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def custom_collate_fn(self, batch):
        # tgt = [<SOS>, [embeddings], <EOS> (, [<PAD>] ) ]
        images, labels = torch.utils.data._utils.collate.default_collate(batch)
        # data = images.reshape(self.args.seq_len, self.args.seq_bs, self.args.n_channels, self.args.img_h, self.args.img_w)
        labels = labels.reshape(self.args.seq_len, self.args.seq_bs)

        src_seq = process_labels(labels, tokens['src']['sos'], tokens['src']['eos'])
        tgt_seq = get_embedding_from_vocab(src_seq, self.vocab)

        return images, src_seq, tgt_seq

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

        # Create vocab if doesn't exist
        # FIXME Fix embeddings for eos and sos
        if os.path.exists('vocab.pth'):
            self.vocab = torch.load('vocab.pth')
        else:
            self.vocab = build_vocab(self.train_dataloader(), tokens)


        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.MNIST(self.data_dir, train=False, transform=self.transforms)

    def get_example_batch(self):
        return next(iter(DataLoader(self.train_dataset, batch_size=self.args.batch_size, drop_last=True, collate_fn=self.custom_collate_fn)))[1:]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, persistent_workers=self.args.persistent_workers, drop_last=True, collate_fn=self.custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=False, drop_last=True, collate_fn=self.custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=False, drop_last=True, collate_fn=self.custom_collate_fn)




class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, args, data_dir="data"):
        super().__init__()
        self.hparams = args
        self.data_dir = data_dir
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)

    def setup(self, stage=None):
        self.transforms = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.Resize((self.hparams.img_h, self.hparams.img_w)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            cifar_train = datasets.CIFAR10(self.data_dir, train=True, transform=self.transforms)
            self.train_dataset, self.val_dataset = random_split(cifar_train, [46000, 4000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.CIFAR10(self.data_dir, train=False, transform=self.transforms)

    def train_dataloader(self):
        
        dataset = datasets.CIFAR10(root=self.data_dir, train=True, transform=self.transforms)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
                T.Resize((self.hparams.img_h, self.hparams.img_w)),
            ]
        )
        dataset = datasets.CIFAR10(root=self.data_dir, train=False, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            # pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()
