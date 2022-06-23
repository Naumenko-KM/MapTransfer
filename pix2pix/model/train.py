import torch
from dataset import MapDataset
import sys
from utils import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler, epoch):
    loop = tqdm(loader, leave=True)
    D_loss_epoch = []
    G_loss_epoch = []
    L1_loss_epoch = []
    disc.train()
    gen.train()
    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
                epoch=epoch+1
            )
            
        D_loss_epoch.append(D_loss)
        G_loss_epoch.append(G_loss)
    
    D_loss_epoch = torch.FloatTensor(D_loss_epoch).mean()
    G_loss_epoch = torch.FloatTensor(G_loss_epoch).mean()

    return D_loss_epoch, G_loss_epoch


def validation_fn(gen, disc, loader):
    gen.eval()
    disc.eval()
    loop = tqdm(loader, leave=True)
    L1_losses = []
    L2_losses = []
    disc_accuracy = []
    L1 = nn.L1Loss()
    L2 = nn.MSELoss()
    Sigmoid = nn.Sigmoid()
    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        with torch.no_grad():
            y_fake = gen(x)
            preds = disc(x, y_fake.detach())
            preds = Sigmoid(preds).round()
            accuracy = (preds == torch.zeros_like(preds)).sum() / torch.numel(preds)

            disc_accuracy.append(accuracy)
            L1_losses.append(L1(y_fake, y))
            L2_losses.append(L2(y_fake, y))

    L1_loss = torch.FloatTensor(L1_losses).mean()
    L2_loss = torch.FloatTensor(L2_losses).mean()
    disc_accuracy = torch.FloatTensor(disc_accuracy).mean()
    # Чем хуже дискриминатор отличает фэйковые картинки, тем лучше
    disc_accuracy = 1 - disc_accuracy
    return L1_loss, L2_loss, disc_accuracy


def main():
    D_loss =  []
    G_loss = []
    L1_loss_val = []
    L2_loss_val = []
    disc_accuracy_val = []
    D_LR_plot = []
    G_LR_plot = []
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.D_LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.G_LEARNING_RATE, betas=(0.5, 0.999))
    sheduler_disc = optim.lr_scheduler.StepLR(opt_disc, step_size=config.D_SCHEDULER_STEP, gamma=config.D_SCHEDULER_GAMMA)
    sheduler_gen = optim.lr_scheduler.StepLR(opt_gen, step_size=config.G_SCHEDULER_STEP, gamma=config.G_SCHEDULER_GAMMA)

    BCE = nn.BCEWithLogitsLoss()
    
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.G_LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.D_LEARNING_RATE,
        )

    train_dataset = MapDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        NUM_WORKERS=config.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = MapDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        if epoch != 0:  # to plot LR properly (scaler works inside train func)
            sheduler_disc.step()
            sheduler_gen.step()

        D_loss_epoch, G_loss_epoch = train_fn(disc, gen, train_loader, opt_disc,
                                                opt_gen, L1_LOSS, BCE, g_scaler, d_scaler, epoch)
        L1_val, L2_val, disc_accuracy = validation_fn(gen, disc, val_loader)

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
        
        D_loss.append(D_loss_epoch.cpu().detach().numpy())
        G_loss.append(G_loss_epoch.cpu().detach().numpy())
        L1_loss_val.append(L1_val.cpu().detach().numpy())
        L2_loss_val.append(L2_val.cpu().detach().numpy())
        disc_accuracy_val.append(disc_accuracy.cpu().detach().numpy())
        D_LR_plot.append(sheduler_disc.get_last_lr())
        G_LR_plot.append(sheduler_gen.get_last_lr())

        display.clear_output(True)
        plot_loss_lr(D_loss, G_loss, L1_loss_val, L2_loss_val, G_LR_plot, D_LR_plot, epoch)
        save_some_examples(gen, val_loader, epoch, folder=config.EVAL_DIR, show_images=True)
        if epoch % 5 == 0 or epoch + 1 == config.NUM_EPOCHS:
            save_gif(config.EVAL_DIR, L1_loss_val, L2_loss_val)
        display_gif(config.EVAL_DIR)
        save_metrics(L1_loss_val, L2_loss_val, disc_accuracy, config.EVAL_DIR, plot=True)

    # print(f'Best L1: {max(L1_loss)/config.L1_LAMBDA}, epoch: {np.asarray(L1_loss).argmax()}')


if __name__ == "__main__":
    main()