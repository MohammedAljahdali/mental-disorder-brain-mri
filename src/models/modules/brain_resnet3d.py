from torch import nn
import torchvision


class BrainResNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.model = torchvision.models.video.r3d_18(pretrained=False, progress=True)
        # self.up_c = nn.Sequential(
        #     nn.Conv3d(1, 3, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
        #     nn.ReLU(),
        #     nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
        # )
        self.model.fc = nn.Linear(self.model.fc.in_features, hparams["num_classes"])

    def forward(self, x):
        # x = x.unsqueeze(1)
        # x = self.up_c(x)
        x = self.model(x)
        return x
