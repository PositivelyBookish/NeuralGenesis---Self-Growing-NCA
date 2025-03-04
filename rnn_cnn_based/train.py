import argparse
import yaml
from tqdm import tqdm
import sys
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import os
sys.path.append("/home/Souvik.Roy/rnn_nca/")
from model import NCA
from helper import load_image, pad_image, make_seed, L2, make_circle_masks, rgba_to_rgb
import sys


def setup_device(device=None):
    """
    Set up device for training. If no device is specified, use cuda if available,
        otherwise use mps or cpu.

    Args:
        device (str): device to use for training (defaults to None)

    Returns:
        device (torch.device): device to use for training
    """

    if device is not None:
        device = torch.device(device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def plot_loss(losses, save_path):
    """
    Plot the loss.

    Args:
        losses (list): list of losses during training

    Returns:
        None
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.plot(losses)
    plt.title("Loss during training")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(save_path)
    plt.show()


def train(config):

    device = setup_device(config["device"])
    print(f"Using device: {device}")

    # load target image, pad it and repeat it batch_size times
    target = load_image(config["target_path"], config["img_size"])
    target = pad_image(target, config["padding"])
    target = target.to(device)
    target_batch = target.repeat(config["batch_size"], 1, 1, 1)

    model = NCA(n_channels=config["n_channels"], filter=config["filter"], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # initialize pool with seed cell state
    seed = make_seed(config["img_size"], config["n_channels"])
    seed = pad_image(seed, config["padding"])
    seed = seed.to(device)
    pool = seed.clone().repeat(config["pool_size"], 1, 1, 1)

    losses = []

    for iter in tqdm(range(config["iterations"])):
        
        # randomly select batch_size cell states from pool
        batch_idxs = np.random.choice(
            config["pool_size"], config["batch_size"], replace=False
        ).tolist()

        cs = pool[batch_idxs]

        # run model for random number of iterations 
        for i in range(np.random.randint(64, 96)):
            cs = model(cs)

        # calculate loss for each image in batch
        if config["loss"] == "L1":
            loss_batch, loss = L1(target_batch, cs)
        elif config["loss"] == "L2":
            loss_batch, loss = L2(target_batch, cs)
        elif config["loss"] == "Manhattan":
            loss_batch, loss = Manhattan(target_batch, cs)
        elif config["loss"] == "Hinge":
            loss_batch, loss = Hinge(target_batch, cs)

        losses.append(loss.item())

        # backpropagate loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # index of cell state with highest loss in batch
        argmax_batch = loss_batch.argmax().item()
        argmax_pool = batch_idxs[argmax_batch]

        # indices of cell states in batch that are not the cell state with highest loss
        remaining_batch = [i for i in range(config["batch_size"]) if i != argmax_batch]
        remaining_pool = [i for i in batch_idxs if i != argmax_pool]

        # replace cell state with highest loss in pool with seed image
        pool[argmax_pool] = seed.clone()
        # update cell states of selected batch with cell states from model output
        pool[remaining_pool] = cs[remaining_batch].detach()

        # damage cell states in batch if config["damage"] is True
        if config["damage"]:
            # get indicies of the 3 best losses in batch
            best_idxs_batch = np.argsort(loss_batch.detach().cpu().numpy())[:3]
            # get the corresponding indicies in the pool
            best_idxs_pool = [batch_idxs[i] for i in best_idxs_batch]

            # replace the 3 best cell states in the batch with damaged versions of themselves
            for n in range(3):
                # create damage mask
                damage = 1.0 - make_circle_masks(config["img_size"]).to(device)
                damage = nn.functional.pad(damage, (16, 16, 16, 16), mode="constant", value=1)
                # apply damage mask to cell state
                pool[best_idxs_pool[n]] *= damage

    model_dir = os.path.dirname(config["model_path"])
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), config["model_path"])

    plot_loss(losses, "/home/Souvik.Roy/rnn_nca/models/lizard/L2/loss.png" )


if __name__ == "__main__":

    sys.argv = [sys.argv[0]]  

    parser = argparse.ArgumentParser(description="Train a NCA model.")
    parser.add_argument("-c", "--config", type=str, default="/home/Souvik.Roy/rnn_nca/train_config.yaml", help="Path to config file.")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, "r"))

    # train model
    train(config)
