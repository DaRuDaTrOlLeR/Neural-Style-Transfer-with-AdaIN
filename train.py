import datetime
import argparse
import torch
import torch.nn as nn
from pathlib import Path
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import transforms
import AdaIN_net as net
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import custom_dataset

content_weight = 1
style_weight = 1
lr = 5e-5


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('-content_dir', type=str, required=True,
                    help='Directory path to a batch of content images')
parser.add_argument('-style_dir', type=str, required=True,
                    help='Directory path to a batch of style images')
parser.add_argument('-gamma', type=float, default=1.0)

# training options
parser.add_argument('-e', type=int, default=10,
                    help='Epochs')
parser.add_argument('-b', type=int, default=10, help='Batch Size')
parser.add_argument('-l', type=str, help='Encoder Weight File')
parser.add_argument('-s', type=str, help='Decoder Weight File')
parser.add_argument('-p', type=str, help='Image File')
parser.add_argument('-cuda', type=str, help='[y/N]')
args = parser.parse_args()



def train(n_epochs, n_batches, model, content_dataset, style_dataset, optimizer, device):#, writer):
    model.train()
    print('training...')
    losses = []
    losses_c = []
    losses_s = []
    for epoch in range(n_epochs):
#        scheduler.step()
        adjust_learning_rate(optimizer, iteration_count=epoch, lr=lr)
        content_dataloader = DataLoader(content_dataset, batch_size=n_batches, shuffle=True)
        style_dataloader = DataLoader(style_dataset, batch_size=n_batches, shuffle=True)
        loss_t = 0
        loss_c_t = 0
        loss_s_t = 0
        print('epochs ', epoch)
        for batch in range(n_batches):
            content_images = next(iter(content_dataloader)).to(device)
            style_images = next(iter(style_dataloader)).to(device)
            loss_c, loss_s = model(content_images, style_images)
            loss_c = content_weight * loss_c
            loss_s = style_weight * loss_s
            loss = loss_c + loss_s

            optimizer.zero_grad()
            optimizer.step()

            loss_t += loss.item()
            loss_c_t += loss_c.item()
            loss_s_t += loss_s.item()

        losses += [loss_t * args.gamma/n_batches]
        losses_c += [loss_c_t * args.gamma/n_batches]
        losses_s += [loss_s_t * args.gamma/n_batches]
    plt.plot(range(n_epochs), losses_s, label='Style')
    plt.plot(range(n_epochs), losses_c, label='Content')
    plt.plot(range(n_epochs), losses, label='Content+Style')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc=1)
    plt.show()



def adjust_learning_rate(optimizer, iteration_count, lr):
    """Imitating the original implementation"""
    learning_rate = lr / (1.0 + lr * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def main():
    device = torch.device('cpu')

    decoder = net.encoder_decoder.decoder
    encoder = net.encoder_decoder.encoder
    encoder.load_state_dict(torch.load('encoder.pth'))
    encoder = nn.Sequential(*list(encoder.children())[:31])
    model = net.AdaIN_net(encoder, decoder)

    content_tf = train_transform()
    style_tf = train_transform()

    content_dataset = custom_dataset.custom_dataset(args.content_dir, content_tf)
    style_dataset = custom_dataset.custom_dataset(args.style_dir, style_tf)

    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    train(n_epochs=args.e, n_batches=args.b, model=model, content_dataset=content_dataset,
          style_dataset=style_dataset, optimizer=optimizer, device=device) #,scheduler=scheduler , writer=writer)
    torch.save(model.state_dict(), args.s)


if __name__ == '__main__':
    main()
