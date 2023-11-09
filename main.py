import os
import random
from embed_generator.generator import *
from embed_generator.visualization import *
from recommender.model.fashion_mlp import *
from inference import *
from PIL import Image

MODEL_PATH = 'C:/Users/owj04/Desktop/Projects/ai-stylist/recommender/model'
MODEL_NAME = 'tmp'

def show_images(top_path, bottoms):
    fig, axes = plt.subplots(1, 1 + len(bottoms))
    fig.set_figheight(3)
    fig.set_figwidth(3 * (len(bottoms) + 1))

    axes[0].axis("off")
    axes[0].imshow(Image.open(top_path).resize((224, 224)))

    for i, (score, file_path) in enumerate(bottoms, start=1):
        axes[i].axis("off")
        axes[i].set_title(f'{score:.3f}')
        axes[i].imshow(Image.open(file_path).resize((224, 224)))
    plt.show()


def infer():
    model = FashionMLP(512)
    model_path = os.path.join(MODEL_PATH, f'{MODEL_NAME}.pth')
    model.load_state_dict(torch.load(model_path))
    print(f'Model successfully loaded')

    embed_generator = FashionEmbeddingGenerator()
    print(f'Embed Generator successfully loaded')

    recommender = FashionRecommender(model, embed_generator, device='cuda')

    TOP_DIR = 'C:/Users/owj04/Desktop/Projects/ai-stylist/datasets/FashionVCdata/FashionVCdata/top/'
    BOTTOM_DIR = 'C:/Users/owj04/Desktop/Projects/ai-stylist/datasets/FashionVCdata/FashionVCdata/bottom/'
    top_path = os.listdir(TOP_DIR)
    top_path = os.path.join(TOP_DIR, random.choice(top_path))
    top_image = Image.open(top_path)

    bottom_paths = os.listdir(BOTTOM_DIR)
    bottom_paths = [os.path.join(BOTTOM_DIR, random.choice(bottom_paths)) for _ in range(9)]


    scores = []
    for bottom_path in bottom_paths:
        bottom_image = Image.open(bottom_path)

        style_dict = {
            'top': top_image,
            'bottom': bottom_image
            }
        scores.append(recommender.single_infer(style_dict).item())
    bottoms = sorted(list(zip(scores, bottom_paths)), key=lambda x: x[0], reverse=True)
    show_images(top_path, bottoms)

if __name__ == '__main__':
    infer()