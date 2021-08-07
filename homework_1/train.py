from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch


def train(args):
    model = model_factory[args.model]()
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.n_epochs
    device = torch.device(args.device)
    model.to(device)

    train_data = load_data('data/train', batch_size=args.batch_size)
    valid_data = load_data('data/valid', batch_size=args.batch_size)

    #create the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    #create the loss
    loss = ClassificationLoss()
    #start training
    for epoch in range(epochs):
        model.train()
        loss_values = []
        train_accuracy= []
        for image, label in train_data:
            image = image.to(device)
            label = label.to(device)

            logit = model(image)
            #compute the loss
            loss_value = loss(logit, label)
            loss_values.append(loss_value.detach().cpu().numpy())
            # compute the accuracy
            acc_val = accuracy(logit, label)
            train_accuracy.append(acc_val.detach().cpu().numpy())

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        avg_loss = sum(loss_values) / len(loss_values)
        avg_train_accuracy = sum(train_accuracy) / len(train_accuracy)

        model.eval()
        valid_accuracy = []
        for image, label in valid_data:
            image = image.to(device)
            label = label.to(device)
            #compute the accuracy
            acc = accuracy(model(image), label)
            valid_accuracy.append(acc.detach().cpu().numpy())
        avg_valid_accuracy = sum(valid_accuracy) / len(valid_accuracy)

        print('epoch %-3d \t loss = %0.3f \t train accuracy = %0.3f \t val accuracy = %0.3f'
              % (epoch, avg_loss, avg_train_accuracy, avg_valid_accuracy))

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default ='linear')
    parser.add_argument('-d', '--device', defualt= 'cuda')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-e', '--epochs', type=int, default=50)


    args = parser.parse_args()
    train(args)
