import os
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple


def show_top_n_image(data_path, file_names, similarities, n: int=5):
    n = max(n, len(file_names))
    fig, axes = plt.subplots(1, n)
    fig.set_figheight(1)
    fig.set_figwidth(n)
    for i, (file_name, similarity) in enumerate(zip(file_names[:n], similarities[:n])):
        axes[i].axis("off")
        if similarities:
            axes[i].set_title(f"{similarity:.3f}")
        else:
            axes[i].set_title(f"{file_name}")
        path = os.path.join(data_path, str(file_name))
        axes[i].imshow(Image.open(path).resize((224, 224)))
    plt.show()


def show_single_image(data_path, file_name):
    fig, axes = plt.subplots(1, 1)
    fig.set_figheight(1)
    fig.set_figwidth(1)
    axes.axis("off")
    axes.set_title(f"query")
    path = os.path.join(data_path, str(file_name))
    axes.imshow(Image.open(path).resize((224, 224)))
    plt.show()


def show_full_outfit(top_data_path, bottom_data_path, top_filenames, bottom_filenames, compatibilities, style):
    fig, axes = plt.subplots(2, 10)
    fig.set_figheight(2)
    fig.set_figwidth(10)
    axes.axis("off")
    axes.set_title(f"a photo of {style} style clothes")
    
    for i in range(10):
        axes[0][i].set_title(f"{compatibilities[i]:.3f}")
        
        top_path = os.path.join(top_data_path, str(top_filenames[i]))
        axes[0][i].imshow(Image.open(top_path).resize((224, 224)))
        
        bottom_path = os.path.join(bottom_data_path, str(bottom_filenames[i]))
        axes[0][i].imshow(Image.open(bottom_path).resize((224, 224)))
    
    plt.show()
    plt.savefig(f"{style}")
    plt.close()