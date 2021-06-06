from torch import nn
import torchvision
import torch

class BrainResNetV1(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        in_features = self.model.fc.in_features
        self.indv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(64),
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(64, 32),
                nn.ReLU(),
                # nn.Linear(hparams["linear_out_dim"], hparams["linear_out_dim"] // 2),
                # nn.ReLU(),
            ) for _ in range(hparams["num_dim"])
        ])
        # self.model.fc = nn.Linear(in_features, hparams["linear_out_dim"])
        self.model.avgpool = nn.Identity()
        self.model.fc = nn.Identity()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(32 * hparams["num_dim"], hparams["linear_out_dim"]),
            nn.ReLU(),
            nn.Linear(hparams["linear_out_dim"], hparams["linear_out_dim"] // 4),
            nn.ReLU(),
            nn.Linear(hparams["linear_out_dim"] // 4, hparams["num_classes"]),
            # nn.Linear(hparams["linear_out_dim"] * hparams["num_dim"], hparams["num_classes"]),
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear((hparams["linear_out_dim"] // 32) * hparams["num_dim"], hparams["linear_out_dim"]),
        #     nn.ReLU(),
        #     nn.Linear(hparams["linear_out_dim"], hparams["linear_out_dim"] // 2),
        #     nn.ReLU(),
        #     nn.Linear(hparams["linear_out_dim"] // 2, hparams["linear_out_dim"] // 4),
        #     nn.ReLU(),
        #     nn.Linear(hparams["linear_out_dim"] // 4, hparams["num_classes"]),
        #     # nn.Linear(hparams["linear_out_dim"] * hparams["num_dim"], hparams["num_classes"]),
        # )

    def forward(self, x):
        xs = []
        for s in x:
            s = self.model.conv1(s)
            s = self.model.bn1(s)
            s = self.model.relu(s)
            s = self.model.maxpool(s)
            s = self.model.layer1(s)
            s = self.model.layer2(s)
            s = self.model.layer3(s)
            s = self.model.layer4(s)
            xs.append(s)
        x = torch.stack(xs)
        x = x.permute(1, 0, 2, 3, 4)
        xs = []
        for i, s in enumerate(x):
            xs.append(self.indv_layers[i](s))
        x = torch.stack(xs)
        x = x.permute(1, 0, 2)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

class BrainResNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:6])
        # self.shared = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(num_features=32, momentum=0.9),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(0.3),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(num_features=64, momentum=0.9),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(0.3),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )
        self.indv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=128, momentum=0.9),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.3),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=64, momentum=0.9),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.3),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(64*2*2, 64),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.5),
                nn.Linear(64, 16),
                nn.ReLU(inplace=True),
                # nn.Linear(hparams["linear_out_dim"], hparams["linear_out_dim"] // 2),
                # nn.ReLU(),
            ) for _ in range(hparams["num_dim"])
        ])
        # for l in self.indv_layers:
        #     l[0].fc = nn.Linear(512, 32)
        # self.model.fc = nn.Linear(in_features, hparams["linear_out_dim"])
        # self.model.avgpool = nn.Identity()
        # self.model.fc = nn.Identity()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(16 * hparams["num_dim"], hparams["linear_out_dim"]),
            nn.ReLU(inplace=True),
            nn.Linear(hparams["linear_out_dim"], hparams["linear_out_dim"] // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hparams["linear_out_dim"] // 4, hparams["num_classes"]),
        )


    def forward(self, x):
        xs = []
        for s in x:
            s = self.model(s)
            # s = self.model.conv1(s)
            # s = self.model.bn1(s)
            # s = self.model.relu(s)
            # s = self.model.maxpool(s)
            # s = self.model.layer1(s)
            # s = self.model.layer2(s)
            # s = self.model.layer3(s)
            # s = self.model.layer4(s)
            xs.append(s)
        x = torch.stack(xs)
        x = x.permute(1, 0, 2, 3, 4)
        xs = []
        for i, s in enumerate(x):
            xs.append(self.indv_layers[i](s))
        x = torch.stack(xs)
        x = x.permute(1, 0, 2)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

class ResidualBlock(nn.Module):
    """
    A residual block as defined by He et al.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ResidualBlock, self).__init__()
        self.conv_res1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, bias=False)
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        self.conv_res2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, bias=False)
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)

        if stride != 1:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu(out)
        out += residual
        return out