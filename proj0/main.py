import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('../tensorboard_log/mobilenet_cifar10')

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    total, correct = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        # loss = F.nll_loss(output, target)
        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        total += target.size(0)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            writer.add_scalar('train/loss', running_loss / args.log_interval, epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train/acc', 100. * correct / total, epoch * len(train_loader) + batch_idx)
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
            running_loss = 0.0


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    writer.add_scalar('test/loss', test_loss / len(test_loader), epoch)
    writer.add_scalar('test/acc', 100. * correct / len(test_loader.dataset), epoch)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='EML HW 0')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--model', type=str, default='Net', help='which model to use')
    parser.add_argument('--resume', type=str, help='path to state dict')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])
    dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                                transform=transform_train)
    dataset2 = datasets.CIFAR10('../data', train=False,
                                transform=transform_test)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    if args.model == 'Net':
        from mnist import Net
        model = Net()
    elif args.model == 'Resnet18':
        from resnet import resnet18
        model = resnet18()
    elif args.model == 'MobileNet':
        from mobilenet_v2 import MobileNetV2
        model = MobileNetV2(10, 1)
    else:
        raise Exception(f'model {args.model} not defined')
    model = model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr,
    #                       momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    epoch_start = 1
    epoch_end = args.epochs
    if args.resume:
        ckpt = torch.load(f'../checkpoints/{args.resume}')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        prev_epoch = ckpt['epoch']
        epoch_start += prev_epoch
        epoch_end += prev_epoch

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    scheduler = CosineAnnealingLR(optimizer, epoch_end)
    for epoch in range(epoch_start, epoch_end + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        writer.add_scalar('train/learning rate', torch.tensor(scheduler.get_last_lr()), epoch)
        test(model, device, test_loader, epoch)
        scheduler.step()

    if args.save_model:
        torch.save({
            'epoch': epoch_end,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            f"../checkpoints/cifar_sgd_cosine100_lr_dot1_{epoch_end}.pt")


if __name__ == '__main__':
    main()