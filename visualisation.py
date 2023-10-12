import matplotlib.pyplot as plt
import torch
import os


def create_fig_training(checkpoint_path, plot_name=''):
    """
    :param checkpoint_path: str path of the checkpoint .ckpt file
    :param plot_name: (optional) name of the file must be .png default it's the checkpoint name
    :return: (None): create the .png file with the plots
    """
    checkpoint = torch.load(checkpoint_path)
    ax1 = plt.subplot(211)
    num_epochs = len(checkpoint['train_losses'])
    ax1.plot(list(range(num_epochs)), checkpoint['train_losses'], label='training')
    ax1.plot(list(range(num_epochs)), checkpoint['val_losses'], label='validation')
    ax1.legend(loc="upper left")
    ax1.set_ylabel('loss')
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.set_ylabel('accuracy')
    ax2.plot(list(range(num_epochs)), checkpoint['train_accs'], label='training')
    ax2.plot(list(range(num_epochs)), checkpoint['val_accs'], label='validation')
    ax2.legend(loc="upper left")
    name = plot_name or os.path.splitext(os.path.basename(checkpoint_path))[0] + '.png'
    plt.savefig(name)


if __name__ == '__main__':
    create_fig_training('model.ckpt')