import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()
        
        self.model = nn.Sequential(
            #
            
        
            ##Backbone
        
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),    #->224*224
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),    #->114*114
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),    #->56*56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),    #->28*28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),    #->14*14
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        
            ##flatten layer
        
            nn.Flatten(),  #->256*7*7
        
            ##Head
        
            nn.Linear(256*7*7, 500),  # -> 500
            nn.Dropout(0.5),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, num_classes),
        )
            

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
