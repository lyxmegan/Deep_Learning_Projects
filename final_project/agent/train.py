import torch
import torchvision
import numpy as np

from .models import Detector, save_model, FocalLoss, load_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb
from torchsummary import summary
from torch import optim
from torch.autograd import Variable


def train(args):
    from os import path
    model = Detector()
    train_logger, valid_logger = None, None

    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    """
    lr = float(args.learning_rate)
    num_epochs = int(args.epochs)
    best_loss = 100000

    cuda = torch.cuda.is_available()

    if args.pretrain:
        model = load_model()
    else:
        model = Detector()

    if cuda:
        model.cuda()

    print(summary(model, (3, 64, 64)))

    train_dataloader = load_detection_data("dense_data/train", num_workers=8, batch_size=64, 
                                           transform=dense_transforms.Compose([
                                               dense_transforms.RandomHorizontalFlip(0.5),
                                               dense_transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.5),
                                               dense_transforms.ToTensor(),
                                               dense_transforms.ToHeatmap()]))

    val_dataloader = load_detection_data("dense_data/valid", num_workers=8, batch_size=64, 
                                         transform=dense_transforms.Compose([dense_transforms.ToTensor(),
                                                                             dense_transforms.ToHeatmap()]))
    dataloader = {}
    dataloader["train"] = train_dataloader
    dataloader["val"] = val_dataloader

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, verbose=1, factor=0.5)
    
    criterion = FocalLoss()

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)

        for phase in ['train', 'val']:

            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0

            for data in dataloader[phase]:
                inputs, labels = data

                if cuda:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                # forward
                output = model(inputs)
                loss = criterion(output, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(dataloader[phase])

            if phase == "val":
                for param_group in optimizer.param_groups:
                    print(f"current learning rate: {param_group['lr']}")
                scheduler.step(epoch_loss)
                valid_logger.add_scalar('loss', epoch_loss, epoch)
            else:
                train_logger.add_scalar('loss', epoch_loss, epoch)

            print(f"{phase}: Loss: {epoch_loss:.6f}")

        if phase == 'val' and epoch_loss < best_loss:
            print(f"saving model .... , loss improve from {best_loss:.6f} to {epoch_loss:.6f}")
            best_loss = epoch_loss
            save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default="logs/")
    # Put custom arguments here
    parser.add_argument('--learning_rate', default=0.01)
    parser.add_argument('--pretrain', default=False)
    parser.add_argument('--epochs', default=100)

    args = parser.parse_args()
    train(args)