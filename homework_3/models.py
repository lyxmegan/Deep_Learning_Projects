import torch
import torch.nn.functional as F


class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Dropout()
            )

            self.downsample = None
            if stride !=1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(
                    torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride),
                    torch.nn.BatchNorm2d(n_output)
                )

        def forward(self, x):

            identity = x
            if self.downsample is not None:
                identity = self.downsample(identity)
            return self.net(x) + identity

    def __init__(self, layers=[32, 64, 128], n_input_channels=3, n_output_channels=6):
        super().__init__()
        L = [
            torch.nn.Conv2d(n_input_channels, 32, kernel_size=7, padding=3, stride=2, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding =1)
        ]

        c = 32
        for l in layers:
            L.append(self.Block(c, l, stride=2))
            c = l

        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, n_output_channels)

    def forward(self, x):
        z = self.network(x)
        z = z.mean(dim=[2, 3])
        return self.classifier(z)


class FCN(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, index, stride=2):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(inplace=True)
            )

            self.downsample = None
            if stride != 1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(
                    torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride),
                    torch.nn.BatchNorm2d(n_output)
                )

        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(identity)
            return self.net(x) + identity

    def __init__(self, layers=[64,128], n_input_channels=3, n_output_channels =5):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """

        L = [torch.nn.Conv2d(n_input_channels, layers[0], kernel_size=7, padding=3, stride=2, bias=False),
             torch.nn.BatchNorm2d(layers[0]),
             torch.nn.ReLU(inplace=True)
             ]
        c = layers[0]
        for i, l in enumerate(layers):
            L.append(self.Block(c, l, i, stride=2))
            c = l

        # output channels layer
        L.append(torch.nn.Conv2d(layers[-1], n_output_channels, kernel_size=1, stride=1))
        L.append(torch.nn.BatchNorm2d(n_output_channels))

        # final layer
        L.append(torch.nn.ConvTranspose2d(5, 5, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), output_padding=1))
        L.append(torch.nn.BatchNorm2d(n_output_channels))
        L.append(torch.nn.ConvTranspose2d(5, 5, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=1))
        L.append(torch.nn.BatchNorm2d(n_output_channels))
        L.append(torch.nn.ConvTranspose2d(5, 5, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=1))

        self.net = torch.nn.Sequential(*L)


    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,6,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        z = self.net(x)
        H = x.shape[2]
        W = x.shape[3]
        z = z[:, :, :H, :W]
        return z



model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
