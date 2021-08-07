import torch
import numpy as np
from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb
import inspect


def train(args):
    from os import path
    model = Detector()
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    device = torch.device(args.device)
    model.to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    transform = dense_transforms.Compose(
        [dense_transforms.RandomHorizontalFlip(), dense_transforms.ColorJitter(0.9, 0.9, 0.9, 0.1), dense_transforms.ToTensor(), dense_transforms.ToHeatmap()])
    train_data = load_detection_data('dense_data/train', num_workers=4, transform=transform)
    #valid_data = load_detection_data('dense_data/valid', num_workers=2, transform=dense_transforms.Compose([dense_transforms.ToTensor(), dense_transforms.ToHeatmap()]))

    global_step = 0
    for epoch in range(n_epochs):
        model.train()
        avg_train_loss = []
        for img, det, _ in train_data:
            img, det = img.to(device), det.to(device)
            output = model(img)

            pre_det = torch.sigmoid(output * (1 - 2 * det))
            loss_val = (loss(output, det) * pre_det).mean() / pre_det.mean()
            avg_train_loss.append(loss_val.item())

            if train_logger is not None and global_step % 100 == 0:
                train_logger.add_images('image', img[:16], global_step)
                train_logger.add_images('label', det[:16], global_step)
                train_logger.add_images('pred', torch.sigmoid(output[:16]), global_step)

            if train_logger:
                train_logger.add_scalar('det_loss', loss_val, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
        print("train loss:", sum(avg_train_loss) / len(avg_train_loss), "epoch", epoch)

        model.eval()
        """
        avg_valid_loss = []
        for image, det, _ in valid_data:
            image, det = image.to(device), det.to(device)
            logit = model(image)
            pre_det = torch.sigmoid(logit * (1 - 2 * det))
            loss_value = (loss(logit, det) * pre_det).mean() / pre_det.mean()
            avg_valid_loss.append(loss_value.item())

            if valid_logger is not None:
                valid_logger.add_scalar('valid_loss', loss_value, global_step)

        print("Valid loss: ",sum(avg_valid_loss) / len(avg_valid_loss) , "epoch :", epoch)
        """
    save_model(model)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-d', '--device', defualt='cuda')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-e', '--n_epochs', type=int, default=30)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    #parser.add_argument('-sl', '--schedule_lr', action='store_true')
    #parser.add_argument('-t', '--transform', default='Compose([RandomHorizontalFlip(), ColorJitter(0.9, 0.9, 0.9, 0.1), ToTensor(), ToHeatmap()])')
    #parser.add_argument('-g', '--gamma', type=float, default=0, help="class dependent weight for cross entropy")
    args = parser.parse_args()
    train(args)
