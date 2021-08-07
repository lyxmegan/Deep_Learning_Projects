from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv
from os import path

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        self.data = []
        tensor = transforms.ToTensor()

        with open(path.join(dataset_path, 'labels.csv')) as csvfile:
            ImageReader = csv.reader(csvfile, delimiter=',')
            next(ImageReader)
            for file, label, _ in ImageReader:
                imageId = Image.open(path.join(dataset_path, file))
                image_tensor = tensor(imageId)
                labelName = LABEL_NAMES.index(label)
                self.data.append((image_tensor, labelName))

    def __len__(self):
        return len(self.data)
        # raise NotImplementedError('SuperTuxDataset.__len__')

    def __getitem__(self, idx):
        return self.data[idx]
        # raise NotImplementedError('SuperTuxDataset.__getitem__')


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
