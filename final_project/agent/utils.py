import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from . import dense_transforms


class DetectionSuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=dense_transforms.ToTensor()):
        from PIL import Image
        from glob import glob
        from os import path
        self.data = []
        self.labels = []

        for j in range(4):
            for f in glob(path.join(dataset_path, str(j), '*_pos_ball.npz')):
                self.labels.append(f)
                self.data.append(f.replace('_pos_ball.npz', '_img.png'))
                # label = np.load(f)['arr_0']
                # i = Image.open(f.replace('_pos_ball.npz', '_img.png'))
                # i.load()
                # self.data.append((i, [label]))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = np.load(self.labels[idx])["arr_0"]
        img = Image.open(self.data[idx])
        img.load()
        res = [img, [label]]
        res = self.transform(*res)
        return  res[0], res[1]


def load_detection_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = DetectionSuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


if __name__ == '__main__':
    dataset = DetectionSuperTuxDataset('dense_data/train/')
    import torchvision.transforms.functional as F
    from pylab import show, subplots
    import matplotlib.patches as patches
    import numpy as np

    fig, axs = subplots(1, 2)
    for i, ax in enumerate(axs.flat):
        im, puck = dataset[100+i]
        ax.imshow(F.to_pil_image(im), interpolation=None)
        print(puck)
        for p in puck:
            ax.add_patch(
                patches.Rectangle((p[0] - 0.55, p[1] - 0.55), 0.05, 0.05, fc='none', ec='r', lw=2))
        ax.axis('off')
    dataset = DetectionSuperTuxDataset('dense_data/train/',
                                       transform=dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(0.5),
                                                                           dense_transforms.ToTensor(),
                                                                           dense_transforms.to_heatmap]))
    fig.tight_layout()

    fig, axs = subplots(1, 2)
    for i, ax in enumerate(axs.flat):
        im, hm = dataset[100+i]
        ax.imshow(F.to_pil_image(im), interpolation=None)
        hm = hm.numpy().transpose([1, 2, 0])
        alpha = 0.25*hm.max(axis=2) + 0.75
        r = 1 - np.maximum(hm[:, :, 0], hm[:, :, 0])
        g = 1 - np.maximum(hm[:, :, 0], hm[:, :, 0])
        b = 1 - np.maximum(hm[:, :, 0], hm[:, :, 0])
        ax.imshow(np.stack((r, g, b, alpha), axis=2), interpolation=None)
        ax.axis('off')
    fig.tight_layout()
    show()

