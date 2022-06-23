import os
import csv
import random

import numpy as np
import pandas as pd
import torch
from torchvision.utils import save_image
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from IPython import display

import config


def save_some_examples(gen, val_loader, epoch, folder, show_images=False):
    iterator = iter(val_loader)
    x, y = next(iterator)
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    if not os.path.isdir(folder):
        os.makedirs(folder)

    with torch.no_grad():
        y_fake = gen(x)
        save_image(y_fake, folder + f"/y_gen_{epoch:03d}.png")
        if epoch == 0:
            save_image(x * 1 + 0, folder + f"/input.png")
            save_image(y * 1 + 0, folder + f"/label.png")
    gen.train()

    if show_images:
        f, ax = plt.subplots(1, 3, figsize=(12, 8))
        x = x.squeeze().cpu().permute(1, 2, 0)
        y = y.squeeze().cpu().permute(1, 2, 0)
        y_fake = y_fake.squeeze().cpu().permute(1, 2, 0)

        ax[0].imshow(x)
        ax[1].imshow(y)
        ax[2].imshow(y_fake)
        plt.show()


def plot_loss_lr(D_loss, G_loss, L1_loss, L2_loss,
                 G_LR_plot, D_LR_plot, epoch=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    ax_lr = ax.twinx()
    ax.plot(D_loss, label='Discriminator loss (x10)')
    ax.plot(G_loss, label='Generator loss')
    ax.plot(L1_loss, label='L1 validation')
    ax.plot(L2_loss, label='L2 validation')
    ax_lr.plot(G_LR_plot, color='y', label='Generator LR',
               linewidth=0.5, alpha=0.7)
    ax_lr.plot(D_LR_plot, color='g', label='Discriminator LR',
               linewidth=0.5, alpha=0.7)

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax_lr.set_ylabel('Learning rate')

    ax.legend(loc=0)
    ax_lr.legend(loc=0, bbox_to_anchor=(1, 0.84))
    if not os.path.isdir('evaluation'):
        os.makedirs('evaluation')
    plt.savefig('evaluation/output.png')
    plt.show()


def concat_two_images(img0, img1):
    img_new = Image.new('RGB', (img0.width + img1.width, img0.height))
    img_new.paste(img0, (0, 0))
    img_new.paste(img1, (img0.width, 0))
    return img_new


def save_gif(dir_to_imgs, L1_loss, L2_loss):
    generated_images = []
    img_label = Image.open(dir_to_imgs + '/label.png')
    filenames = [filename for filename in os.listdir(dir_to_imgs)
                 if filename.startswith('y')]
    for filename, l1, l2 in zip(filenames, L1_loss, L2_loss):
        img = Image.open(dir_to_imgs+'/'+filename)
        img = concat_two_images(img_label, img)
        img_number = int(filename[6:9])
        draw = ImageDraw.Draw(img).text(
            (0, 0),
            f"epoch: {img_number+1}/{config.NUM_EPOCHS}",
            (255, 255, 255)
            )
        draw = ImageDraw.Draw(img).text((0, 10), f"L1: {l1:.3f}")
        draw = ImageDraw.Draw(img).text((0, 20), f"L2: {l2:.3f}")
        generated_images.append(img)

    generated_images[0].save(dir_to_imgs+"/out.gif", save_all=True,
                             append_images=generated_images[1:],
                             duration=300, loop=1)


def display_gif(dir_to_gif):
    filename = dir_to_gif + '/out.gif'
    try:
        with open(filename, 'rb') as f:
            display.display(display.Image(data=f.read(), format='png'))
    except FileNotFoundError:  # if there is no file
        pass


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning
    # rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_metrics(L1, L2, disc_accuracy, folder, plot=False):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    header = ['mae', 'mse', 'D accuracy']
    with open(folder+'/metrics.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(zip(L1, L2, disc_accuracy))
    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        df = pd.read_csv(folder+'/metrics.csv')
        ax.plot(df['mae'], label='mae')
        ax.plot(df['mse'], label='mse')
        ax.plot(df['D accuracy'], label='D accuracy')
        plt.savefig('evaluation/metrics.png')
        plt.show()


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()
