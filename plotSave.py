import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import seaborn as sns

if __name__ == '__main__':

    with open('rewards_losses.pkl', 'rb') as handle:
        rewards, losses = pkl.load(handle)

    # Smoothes out the plot a bit
    rewards = np.convolve(rewards, np.ones((5,))/5, mode='valid')

    # Generate the figure
    sns.set(style='darkgrid')
    plt.figure()
    plt.plot(range(len(rewards)), rewards)
    plt.xlabel('Rollout')
    plt.ylabel('Accuracy')

    # Save instead of show
    plt.savefig("rewards_plot.png", dpi=300, bbox_inches="tight")
    plt.close()  # optional: frees memory if you generate many plots
