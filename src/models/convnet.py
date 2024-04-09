from torch import nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.conv3(x)))
        return x
    
class FullyConnectedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32 * 4 * 4, 512)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 100)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)

        return x
    
class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.convnet = ConvNet()
        self.fcnet = FullyConnectedNet()
    
    def forward(self, x):

        x = self.convnet(x)
        x = x.view(-1, 512)
        x = self.fcnet(x)
        return x


if __name__ == "__main__":
    from torchsummary import summary

    convnet = ConvNet().to("cuda")

    model = Model().to("cuda")

    summary(model, input_size=(3, 32, 32))

