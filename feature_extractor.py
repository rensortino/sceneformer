import torch

w_path = 'ckpt/resnet18_mnist.pt'


class ResNet18(nn.Module):
    def __init__(self, w_path, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.load_state_dict(torch.load(w_path)['model_state_dict'])
        self.out_layer = self.model._modules.get('avgpool')
        self.embedding_size = self.model._modules.get('fc').in_features

    def get_vectors(self, images):
        # Create a vector of zeros that will hold our feature vector
        # The 'avgpool' layer has an output size of 512
        visual_embedding = torch.zeros(self.embedding_size * batch_size).cuda()

        # Define a function that will copy the output of a layer
        def copy_data(module, input, output):
            visual_embedding.copy_(output.flatten())

        # Attach that function to our selected layer
        #hook = self.out_layer.register_forward_hook(copy_data)
        hook = self.out_layer.register_full_backward_hook(copy_data)
        # Run the model on our transformed image
        self.model(images)
        # Detach our copy function from the layer
        hook.remove()
        # Return the feature vector
        return visual_embedding

    def forward(self, imgs):
        output = self.model(imgs)
        return output
