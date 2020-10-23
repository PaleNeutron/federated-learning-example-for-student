import torch
import torch.nn.functional as F

from preprocess import CompDataset


def user_round_train(X, Y, model, device, debug=False):
    data = CompDataset(X=X, Y=Y)
    train_loader = torch.utils.data.DataLoader(
        data,
        batch_size=320,
        shuffle=True,
    )

    model.train()

    correct = 0
    prediction = []
    real = []
    total_loss = 0
    model = model.to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # import ipdb
        # ipdb.set_trace()
        # print(data.shape, target.shape)
        output = model(data)
        loss = F.nll_loss(output, target)
        total_loss += loss
        loss.backward()
        pred = output.argmax(
            dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        prediction.extend(pred.reshape(-1).tolist())
        real.extend(target.reshape(-1).tolist())

        if batch_idx % 100 == 0:
            print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


    grads = {'n_samples': data.shape[0], 'named_grads': {}}
    correct_rate = correct / len(train_loader.dataset)
    for name, param in model.named_parameters():
        grads['named_grads'][name] = param.grad.detach().cpu().numpy() * correct_rate

    if debug:
        print('Training Loss: {:<10.2f}, accuracy: {:<8.2f}'.format(
            total_loss, 100. * correct_rate))

    # better result return larger grad

    return grads
