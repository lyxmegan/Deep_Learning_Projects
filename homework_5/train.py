from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):
    from os import path
    model = Planner()
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    device = torch.device(args.device)
    model.to(device)
    
    train_logger =  None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    """
    Your code here, modify your HW4 code
    
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    loss = torch.nn.MSELoss()

    
    transform = dense_transforms.Compose(
        [dense_transforms.RandomHorizontalFlip(), dense_transforms.ColorJitter(0.9, 0.9, 0.9, 0.1), dense_transforms.ToTensor()]
    )
    train_data = load_data('drive_data', num_workers=4, batch_size = args.batch_size, transform=transform)

    global_step = 0
    for epoch in range(n_epochs):
        model.train()
        loss_train_vals = []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)
            logit = model(img)
            loss_val = loss(logit, label)
            loss_train_vals.append(float(loss_val))

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            global_step += 1

        if train_logger:
            print("epoch: ", epoch, "train loss value: ", np.mean(loss_train_vals))
            train_logger.add_scalar('avg_loss_value', np.mean(loss_train_vals), global_step)

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-e', '--n_epochs', type=int, default=20)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    args = parser.parse_args()
    train(args)
