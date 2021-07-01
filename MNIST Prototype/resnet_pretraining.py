import torch

from torchvision.models import resnet18
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data
from sklearn.metrics import classification_report
from torchvision.transforms import ToTensor
from tqdm.autonotebook import tqdm

train_ds = MNIST("data", train=True, download=True, transform=ToTensor())
test_ds = MNIST("data", train=False, download=True, transform=ToTensor())

train_dl = DataLoader(train_ds, batch_size=64, num_workers=8, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=64, num_workers=8)

class ResNetMNIST(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = resnet18(num_classes=16)
    self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    self.loss = nn.CrossEntropyLoss()

  @auto_move_data
  def forward(self, x):
    return self.model(x)
  
  def training_step(self, batch, batch_no):
    x, y = batch
    logits = self(x)
    loss = self.loss(logits, y)
    return loss
  
  def configure_optimizers(self):
    return torch.optim.RMSprop(self.parameters(), lr=0.005)

model = ResNetMNIST()

trainer = pl.Trainer(
    gpus=1,
    max_epochs=20,
    progress_bar_refresh_rate=20
)

trainer.fit(model, train_dl)

trainer.save_checkpoint("resnet18_mnist.ptl")

def convert_weights_pl_to_pt(w_path, out_path):
    state_dict = torch.load(w_path)['state_dict']
    new_sd = {}
    for k in state_dict.keys():
        new_k = k[6:] #strip the model string
        new_sd[new_k] = state_dict[k]
    torch.save(new_sd, out_path)

convert_weights_pl_to_pt('resnet18_mnist.ptl', 'resnet18_16classes.pt')

def get_prediction(x, model: pl.LightningModule):
  model.freeze() # prepares model for predicting
  probabilities = torch.softmax(model(x), dim=1)
  predicted_class = torch.argmax(probabilities, dim=1)
  return predicted_class, probabilities

inference_model = ResNetMNIST.load_from_checkpoint("resnet18_mnist.ptl", map_location="cuda")

true_y, pred_y = [], []
for batch in tqdm(iter(test_dl), total=len(test_dl)):
  x, y = batch
  true_y.extend(y)
  preds, probs = get_prediction(x, inference_model)
  pred_y.extend(preds.cpu())

print(classification_report(true_y, pred_y, digits=3))