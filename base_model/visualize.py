import argparse
import yaml
from tqdm import tqdm
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from model import NCA 
from helper import *


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train a NCA model.")
    parser.add_argument("-c", "--config", type=str, default="/home/Souvik.Roy/base_nca/visualize_config.yaml", help="Path to config file.")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, "r"))

    model = NCA(n_channels=16, filter=config["filter"], device=torch.device("cpu"))
    model.load_state_dict(torch.load(config["model_path"], map_location="cpu"))
    model.eval()

    target_img = load_image(config["target_path"], size=config["img_size"])
    target_img = pad_image(target_img, config["padding"])

    px = 1/plt.rcParams["figure.dpi"]  # pixel in inches
    fig = plt.figure(frameon=False)
    fig.set_size_inches(config["img_size"]*px, config["img_size"]*px)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # initialize cell state
    cs = make_seed(config["img_size"], config["n_channels"])
    cs = pad_image(cs, config["padding"])

    # store frames for animation
    frames = []

    # run model
    for i in tqdm(range(config["iterations"])):
        cs = model(cs)
        frame = ax.imshow(rgba_to_rgb(cs[:, :4].detach().cpu())[0].permute(1, 2, 0), animated=True) 
        frames.append([frame])

        # damage cell states if config["damage"] is True
        if config["damage"] and i % (config["iterations"] / 10) == 0 and i != 0:
            damage = 1.0 - make_circle_masks(config["img_size"])
            damage = nn.functional.pad(damage, (16, 16, 16, 16), mode="constant", value=1)
            cs *= damage

    if config["loss"] == "L1":
        loss, _ = L1(target_img, cs)
    elif config["loss"] == "L2":
        loss, _ = L2(target_img, cs)
    elif config["loss"] == "Manhattan":
        loss, _ = Manhattan(target_img, cs)
    elif config["loss"] == "Hinge":
        loss, _ = Hinge(target_img, cs)

    print("Final loss: {:.4f}".format(loss.item()))

    # save animation
    ani_dir = os.path.dirname(config["ani_path"])
    os.makedirs(ani_dir, exist_ok=True)
    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
    ani.save(config["ani_path"], writer="imagemagick")

    fig.set_size_inches(5, 5)
    plt.show() 