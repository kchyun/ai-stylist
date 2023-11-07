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


