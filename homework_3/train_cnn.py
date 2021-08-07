from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES, accuracy
import torch
import torchvision
import torch.utils.tensorboard as tb
import numpy as np
from torchvision import transforms
import inspect

def train(args):
    from os import path
    model = CNNClassifier()
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.n_epochs
    device = torch.device(args.device)
    model.to(device)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    transform = eval(args.transform,{t: c for t, c in inspect.getmembers(torchvision.transforms) if inspect.isclass(c)})
    train_data = load_data('data/train', batch_size=args.batch_size, transform=transform)
    valid_data = load_data('data/valid', batch_size=args.batch_size)

    # create the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum = 0.9, weight_decay=1e-5)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)

    # create the loss
    loss = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=50)
    # start training
    global_step = 0
    for epoch in range(epochs):
        model.train()
        loss_values = []
        train_accuracy = []
        for image, label in train_data:
            image = image.to(device)
            label = label.to(device)

            logit = model(image)
            # compute the loss
            loss_value = loss(logit, label)
            loss_values.append(loss_value.detach().cpu().numpy())
            # compute the accuracy
            accuracy_value = accuracy(logit, label)
            train_accuracy.append(accuracy_value.detach().cpu().numpy())
            train_logger.add_scalar("loss", loss_value, global_step=global_step)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            global_step += 1

        avg_loss = sum(loss_values) / len(loss_values)
        avg_train_accuracy = sum(train_accuracy) / len(train_accuracy)
        train_logger.add_scalar("accuracy", avg_train_accuracy, global_step=global_step)

        model.eval()
        valid_accuracy = []
        for image, label in valid_data:
            image = image.to(device)
            label = label.to(device)
            # compute the accuracy
            acc = accuracy(model(image), label)
            valid_accuracy.append(acc.detach().cpu().numpy())
        avg_valid_accuracy = sum(valid_accuracy) / len(valid_accuracy)

        valid_logger.add_scalar("valid accuracy", avg_valid_accuracy, global_step=global_step)
        scheduler.step(avg_valid_accuracy)
        print('epoch %-3d \t loss = %0.3f \t train accuracy = %0.3f \t val accuracy = %0.3f'
              % (epoch, avg_loss, avg_train_accuracy, avg_valid_accuracy))
        
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-d', '--device', defualt='cuda')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-sl', '--schedule_lr', action='store_true')
    parser.add_argument('-t', '--transform', default='Compose([ RandomHorizontalFlip(), ColorJitter(0.5, 0.5, 0.5, 0.1), ToTensor()])')

    args = parser.parse_args()
    train(args)
