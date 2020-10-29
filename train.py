import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import WeightedRandomSampler

from preprocess import CompDataset, ATTACK_TYPES


def user_round_train(X, Y, model, device, debug=False, client_name=""):
    data = CompDataset(X=X, Y=Y)
    unique, counts = np.unique(Y, return_counts=True)
    target_weights = {i: 1/ c for i, c in zip(unique, counts)}
    weights = [target_weights[i] for i in Y]
    sample_size = 100000
    sampler = WeightedRandomSampler(weights, num_samples=sample_size, replacement=True)
    train_loader = torch.utils.data.DataLoader(
        data,
        batch_size=3200,
        shuffle=True,
        # sampler=sampler,
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

        # if batch_idx % 100 == 0:
        #     print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))


    grads = {'n_samples': data.shape[0], 'named_grads': {}}
    correct_rate = correct / len(real)
    if correct_rate > 0.9:
        accelerate_rate = 1 + (1 - correct_rate) * 8
    else:
        accelerate_rate = 1 + (1 - correct_rate) * 2

    for name, param in model.named_parameters():
        grads['named_grads'][name] = param.grad.detach().cpu().numpy()

    if debug:
        print('client: {:<32}  Training Loss: {:<10.2f}  accuracy: {:<8.2f} on tags: {}'.format(client_name,
            total_loss, 100. * correct_rate, " ".join(["{:>2}".format(str(i)) for i in set(real)])))

    # better result return larger grad

    return correct_rate, grads
