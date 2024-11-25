import os
import time
import socket
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from argparse import ArgumentParser as argparse

from utils.utils import write_json
from utils.embedding_size import get_embedding_size
from utils.training import epoch_time, initialize_log
from dataloaders.data_augmentation import data_aug_selector

import torch
from torchsummary import summary
from torch.utils.data import DataLoader

from test_siamese import test_model
from make_templates import make_templates
from loss_fn.triplet_loss import TripletLoss
from models.siamese import siamese_embeddings
from dataloaders.Image_Dataset import Image_Dataset

def train_loop(model, device, data_loader, criterion, optimizer):
    losses = []
    model.train()
    for images, labels, ind in tqdm(data_loader, desc="Train loop"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        features = model(images)

        loss = criterion(features, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().numpy())

    return np.mean(losses)

def validation_loop(model, device, data_loader, criterion):
    losses = []
    model.eval()
    with torch.no_grad():
        for images, labels, ind in tqdm(data_loader, desc="Val  loop "):
            images = images.to(device)
            labels = labels.to(device)

            features = model(images)

            loss = criterion(features, labels)

            losses.append(loss.detach().cpu().numpy())

    return np.mean(losses)


def main(args):
    # Visdom Visualization
    if args.visdom:
        print('Initializing Visdom')
        import visdom
        from utils.linePlotter import VisdomLinePlotter
        vis = visdom.Visdom()
        plotter = VisdomLinePlotter()
    else:
        vis = None

    # Set image name
    if 'ip' in socket.gethostname():
        pc_name = 'AWS'
    else:
        pc_name = socket.gethostname()
    timestamp  = datetime.today().strftime('%Y%m%d_%H%M%S')
    args.model_name = '{}_siam_{}_{}'.format(args.backbone,timestamp,pc_name)

    # Get image size
    img_size = (args.img_size, args.img_size)

    # Check GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Working with: {}".format(device))

    # Create output paths
    model_path = "./checkpoints/{}/".format(args.model_name)
    os.makedirs(model_path, exist_ok=True)
    model_save_path_best = os.path.join(model_path, "best_model.pth")
    model_save_path_last = os.path.join(model_path, "last_model.pth")
    json_log_path = os.path.join(model_path, "log.json")
    loss_fig_path = os.path.join(model_path, "loss.svg")
    figure_title = "{} model loss evolution".format(args.model_name)

    # Train Transform and DA
    transform = data_aug_selector(args)

    # Train loader
    print("\nLoading training set ...")
    train_list = os.path.join(args.dataset, 'train.txt')
    train_dataset = Image_Dataset(train_list, args=args, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.jobs)

    # Validation loader
    print("\nLoading validation set ...")
    validation_list = os.path.join(args.dataset, 'validation.txt')
    validation_dataset = Image_Dataset(validation_list, img_size=img_size, transform=None)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.jobs)

    # Test loader
    test_list = os.path.join(args.dataset, 'test.txt')

    # Get pretrained model
    model = siamese_embeddings(args.backbone, args.weights)
    model.to(device)

    # Resume training
    if args.weights not in ["", "None", None, "imagenet", "legacy"]:
        model.load_state_dict(torch.load(args.weights, map_location=device))

    # Get embedding size
    args.embedding_size = get_embedding_size(model, img_size, device)

    # Print info
    print(" ")
    print("Model architecture:")
    summary(model, input_size=(3, args.img_size, args.img_size))
    print(" ")
    print("Dataset: {}".format(args.dataset))
    print("Train images: {:d}".format(len(train_dataset)))
    print("Validation images: {:d}".format(len(validation_dataset)))
    print("DA Library: {}".format(args.da_library))
    print("DA Level: {}".format(args.da_level))
    print(" ")
    print("Model name: {}".format(args.model_name))
    print("Margin: {}".format(args.margin))
    print("Backbone: {}".format(args.backbone))
    print("Weights: {}".format(args.weights))
    print("Image size: {}".format(img_size))
    print("Embedding size: {}".format(args.embedding_size))
    print("Epochs: {:d}".format(args.epochs))
    print("bs: {:d}".format(args.batch_size))
    print("lr: {:f}".format(args.learning_rate))
    print("lr update freq: {:d}".format(args.lr_update_freq))
    print("jobs: {:d}".format(args.jobs))
    print(" ")

    # Optimizer
    criterion = TripletLoss(args.margin, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.lr_update_freq:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_update_freq, gamma=0.1)

    # Loop variables
    log_dict = initialize_log(args)
    log_dict["training_images"] = len(train_dataset)
    log_dict["validation_images"] = len(validation_dataset)
    train_loss_history = []
    val_loss_history = []
    epochs = []
    epoch_dt = []
    best_loss = 1000
    best_epoch = 0

    # Train model
    T0 = time.time()
    for e in range(1,args.epochs+1):
        print('\nepoch : {:d}'.format(e))
        epochs.append(e)
        t0 = time.time()

        # Train loop
        train_loss = train_loop(model, device, train_loader, criterion, optimizer)

        # Validation loop
        val_loss = validation_loop(model, device, validation_loader, criterion)

        # Update scheduler
        if args.lr_update_freq:
            scheduler.step()

        # Store los values
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        # Stop timer
        t1 = time.time()
        epoch_dt.append(t1-t0)

        # Print epoch info
        print('training  : loss={:.5f}'.format(train_loss))
        print('validation: loss={:.5f}'.format(val_loss))
        print(epoch_time(t0, t1))

        # Save latest model
        torch.save(model.state_dict(), model_save_path_last)

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = e
            torch.save(model.state_dict(), model_save_path_best)

        # Visualize on Visdom
        if args.visdom:
            plotter.plot('loss', 'training  loss ', figure_title, e, train_loss)
            plotter.plot('loss', 'validation loss', figure_title, e, val_loss)

        # Plot loss
        fig, ax = plt.subplots(1,1, figsize=(8,5))
        ax.plot(epochs, train_loss_history, label='training loss')
        ax.plot(epochs, val_loss_history, label='validation loss')
        ax.set_title(figure_title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        plt.savefig(loss_fig_path, format="svg")

        # Measure total training time
        T1 = time.time()

        # Update log_dict
        log_dict["epoch"] = e
        log_dict["val_loss"] = float(val_loss)
        log_dict["best_epoch"] = best_epoch
        log_dict["best_val_loss"] = float(best_loss)
        log_dict["Training_Time"] = epoch_time(T0, T1)
        log_dict["Avg_Epoch_Time"] = epoch_time(0, np.mean(epoch_dt))
        write_json(log_dict, json_log_path)


    # Load best model
    print("\nLoading best model")
    print("Epoch: {:d}".format(best_epoch))
    print("Path: {}".format(model_save_path_best))
    model.load_state_dict(torch.load(model_save_path_best, map_location=device))
    model.eval()

    # Get templates
    templates = make_templates(train_list, model_path, 4, model)

    # Evaluate model on test set
    print("\nEvaluating on test set ...")
    dists, labels = test_model(test_list, model_path, args.batch_size, args.jobs, model, templates)
    '''
    # Comput test accuracy
    acc = sum(y_true == y_pred) / len(y_pred)
    print("Test set accuracy: {:0.4f}".format(acc))
    print(" ")
    '''

if __name__ == '__main__':
    parser = argparse()
    parser.add_argument('-d', '--dataset', type=str,
                        help='Path to the lists of the dataset.')
    parser.add_argument('-b', '--backbone', type=str, default="vgg16",
                        help='Conv-Net backbone.')
    parser.add_argument('-w', '--weights', type=str, default="",
                        help="Model's initial Weights: < none | imagenet | /path/to/weights/ >")
    parser.add_argument('-sz', '--img_size', type=int, default=224,
                        help='Image size.')
    parser.add_argument('-e', '--epochs', type=int, default=2,
                        help='Number of epochs.')
    parser.add_argument('-bs', '--batch_size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('-j', '--jobs', type=int, default=8,
                        help="Number of workers for dataloader's parallel jobs.")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001,
                        help='Learning Rate.')
    parser.add_argument('-lrf', '--lr_update_freq', type=int, default=0,
                        help='Learning rate update frequency in epochs.')
    parser.add_argument('-ma', '--margin', type=float, default=1.0,
                        help='Triplet loss margin parameter.')
    parser.add_argument('-da', '--da_library', type=str, default="albumentations",
                        help='Data Augmentation library: < albumentations | torchvision >')
    parser.add_argument('-lvl', '--da_level', type=str, default="heavy",
                        help='Data Augmentation level: < light | medium | heavy >')
    parser.add_argument('-vis', '--visdom', action='store_true',
                        help='Visualize training on visdom.')
    args = parser.parse_args()

    main(args)
