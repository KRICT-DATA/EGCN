import torch
import copy
import numpy as np
import torch.optim as optim
import torch.utils.data as data_utils

def train(model, criterion, optimizer, train_data_loader, max_epochs):
    model.train()

    for epoch in range(0, max_epochs):
        train_loss = 0

        for data, target in train_data_loader:
            pred = model(data)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

        train_loss /= len(train_data_loader.dataset)

        print('Epoch {}, train loss {:.4f}'.format(epoch + 1, train_loss))


def test(model, criterion, test_data_loader, accs=None):
    preds = None
    model.eval()

    with torch.no_grad():
        test_loss = 0
        correct = 0

        for bg, target in test_data_loader:
            pred = model(bg)
            loss = criterion(pred, target)
            test_loss += loss.detach().item()

            if preds is None:
                preds = pred.clone().detach()
            else:
                preds = torch.cat((preds, pred), dim=0)

            if accs is not None:
                correct += torch.eq(torch.max(pred, dim=1)[1], target).sum().item()

        test_loss /= len(test_data_loader.dataset)

        print('Test loss: ' + str(test_loss))

    if accs is not None:
        accs.append(correct / len(test_data_loader.dataset) * 100)
        print('Test accuracy: ' + str((correct / len(test_data_loader.dataset) * 100)) + '%')

    return test_loss, preds


def cross_validation(dataset, model, criterion, num_folds, batch_size, max_epochs, train, test, accs=None):
    num_data_points = len(dataset)
    size_fold = int(len(dataset) / float(num_folds))
    folds = []
    models = []
    optimizers = []
    test_losses = []

    for k in range(0, num_folds - 1):
        folds.append(dataset[k * size_fold:(k + 1) * size_fold])

    folds.append(dataset[(num_folds - 1) * size_fold:num_data_points])

    for k in range(0, num_folds):
        models.append(copy.deepcopy(model))
        optimizers.append(optim.Adam(models[k].parameters(), weight_decay=0.01))

    for k in range(0, num_folds):
        print('--------------- fold {} ---------------'.format(k + 1))

        train_dataset = []
        test_dataset = folds[k]

        for i in range(0, num_folds):
            if i != k:
                train_dataset += folds[i]

        train_data_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_data_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train(models[k], criterion, optimizers[k], train_data_loader, max_epochs)
        test_loss, pred = test(models[k], criterion, test_data_loader, accs)
        test_losses.append(test_loss)

    if accs is None:
        return np.mean(test_losses)
    else:
        return np.mean(test_losses), np.mean(accs)
