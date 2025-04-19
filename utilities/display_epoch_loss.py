import logging
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
import ast

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    epochLossFilepath,
    title,
    outputDirectory,
    limits
):
    logging.info("display_epoch_loss.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    epoch_loss_df = pd.read_csv(epochLossFilepath)
    epochs = epoch_loss_df['epoch']
    print(epochs)
    total_losses = epoch_loss_df['loss']
    initial_losses = epoch_loss_df['initial_loss']
    boundary_losses = epoch_loss_df['boundary_loss']
    diff_eqn_losses = epoch_loss_df['diff_eqn_loss']

    fig, ax = plt.subplots(1)
    ax.plot(epochs, diff_eqn_losses, label='Differential equation residual loss')
    ax.plot(epochs, initial_losses, label='Initial condition loss')
    ax.plot(epochs, boundary_losses, label='Boundary loss')
    ax.plot(epochs, total_losses, label='Total loss')
    ax.set_yscale('log')
    ax.set_ylim(*limits)
    ax.set_title(title)
    ax.grid(True)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')

    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('epochLossFilepath', help="The filepath to the epoch loss file")
    parser.add_argument('title', help="The figure title")
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_display_epoch_loss'",
                        default='./output_display_epoch_loss')
    parser.add_argument('--limits', help="The y limits. Default: '[1e-2, 1e5]'", default='[1e-2, 1e5]')


    args = parser.parse_args()
    args.limits = ast.literal_eval(args.limits)
    main(
        args.epochLossFilepath,
        args.title,
        args.outputDirectory,
        args.limits
    )