import torch
import numpy as np

from .models import FCN, save_model, model_factory
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb
import inspect


def train(args):
    from os import path
    model = FCN()
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    schedule_lr = args.schedule_lr
    device = torch.device(args.device)
    model.to(device)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    """

    transform= eval(args.transform, {t: c for t, c in inspect.getmembers(dense_transforms) if inspect.isclass(c)})
    train_data = load_dense_data('dense_data/train', batch_size=args.batch_size, num_workers=2, transform=transform)
    valid_data = load_dense_data('dense_data/valid', batch_size=args.batch_size, num_workers=2)

    # create the optimizer
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    # create the loss
    w= torch.as_tensor(DENSE_CLASS_DISTRIBUTION)**(-args.gamma)
    loss = torch.nn.CrossEntropyLoss(weight=w/w.mean()).to(device)
    if schedule_lr:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')

    # start training
    global_step = 0
    for epoch in range(n_epochs):
        model.train()
        matrix = ConfusionMatrix()
        for image, label in train_data:
            image = image.to(device)
            label = label.to(device).long()
            logit = model(image)
            # compute the loss
            loss_value = loss(logit, label)
            if train_logger:
                train_logger.add_scalar('loss', loss_value, global_step)

            matrix.add(logit.argmax(1), label)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            global_step += 1

        if train_logger:
            train_logger.add_scalar('global accuracy',matrix.global_accuracy, global_step )
            train_logger.add_scalar('IOU',matrix.iou, global_step )

        model.eval()
        val_matrix = ConfusionMatrix()
        for image, label in valid_data:
            image = image.to(device)
            label = label.to(device).long()
            img=model(image)
            val_matrix.add(img.argmax(1), label)

        if valid_logger:
            valid_logger.add_scalar('global_accuracy', val_matrix.global_accuracy, global_step)
            valid_logger.add_scalar('iou', val_matrix.iou, global_step)


        scheduler.step(val_matrix.global_accuracy)

        print('epoch %-3d \t train acc = %0.3f \t valid acc = %0.3f \t train iou = %0.3f \t valid iou = %0.3f' %
              (epoch, matrix.global_accuracy, val_matrix.global_accuracy, matrix.iou, val_matrix.iou))
        save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('-d', '--device', defualt='cuda')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-e', '--n_epochs', type=int, default=30)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-sl', '--schedule_lr', action='store_true')
    parser.add_argument('-t', '--transform', default='Compose([RandomHorizontalFlip(), ColorJitter(0.9, 0.9, 0.9, 0.1), ToTensor()])')
    parser.add_argument('-g', '--gamma', type=float, default=0, help="class dependent weight for cross entropy")
    args = parser.parse_args()
    train(args)
