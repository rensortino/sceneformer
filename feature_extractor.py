import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self, w_path, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(num_classes=num_classes)
        if "mnist" in w_path:
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif "cifar" in w_path:
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.load_state_dict(torch.load(w_path))
        self.out_layer = self.model._modules.get('avgpool')
        self.embedding_size = self.model._modules.get('fc').in_features
        # FIXME del self.model.fc ?

    def get_vectors(self, images):
        self.model.eval()
        # Create a vector of zeros that will hold our feature vector
        # The 'avgpool' layer has an output size of 512
        visual_embedding = torch.zeros(images.shape[0], self.embedding_size).cuda()

        # Define a function that will copy the output of a layer
        def copy_data(module, input, output):
            visual_embedding.copy_(output.squeeze(3).squeeze(2))

        # Attach that function to our selected layer
        #hook = self.out_layer.register_forward_hook(copy_data)
        hook = self.out_layer.register_forward_hook(copy_data)
        # Run the model on our transformed image
        with torch.no_grad():
            self.model(images.cuda())
        # Detach our copy function from the layer
        hook.remove()
        # Return the feature vector
        return visual_embedding

    def forward(self, imgs):
        output = self.model(imgs)
        return output



def get_seq_targets(self, img_seq):
        '''
        img_ seq = [seq_len, h, w] = [4, 16, 16]
        '''
        img_seq = img_seq.unsqueeze(1) # Restore Channel info
        seq_len, c, h, w = img_seq.shape
        grid = torch.zeros(seq_len, c, h, w)
        img_embeddings = []
        for i, img in enumerate(img_seq):
            # Place the image in the right quadrant
            grid[i] = img
            # Construct the grid from the batch of images
            img_grid = make_grid(grid.cpu(), nrow=2, padding=0)
            #show_image(img_grid)
            # Take the first channel (Grayscale images, the channels are all the same)
            img_embedding = img_grid[0].flatten()
            img_embedding = feature_extractor.get_vectors(img_grid[0], hp['batch_size'])
            img_embeddings.append(img_embedding.unsqueeze(0)) # Unsqueeze to cat along dim 0 later
        return torch.cat(img_embeddings)