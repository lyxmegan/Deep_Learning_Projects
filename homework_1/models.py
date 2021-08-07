import torch
import torch.nn.functional as F



input_size = 3 * 64 * 64
out_features = 6
class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        loss = F.cross_entropy(input, target)
        return loss


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Linear(input_size, out_features)
        torch.nn.init.normal_(self.network.weight, std = 0.01)
        torch.nn.init.normal_(self.network.bias, std=0.01)

    def forward(self, x):
        return self.network(x.view(x.size(0), -1))


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, out_features)
        )

    def forward(self, x):
        return self.network(x.view(x.size(0), -1))


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
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