import os
import random
from embed_generator.generator import *
from embed_generator.visualization import *
from style_aware_net.model.style_aware_net import *
from inference import *
from PIL import Image

import pandas as pd

MODEL_PATH = 'F:/Projects/ai-stylist/style_aware_net/model/'
MODEL_NAME = '20231119'
DIR = 'F:/Projects/ai-stylist/data/polyvore_outfits/images'

model_args = ModelArgs(
    n_conditions = 7
)

def show_images(top, bottoms):
    fig, axes = plt.subplots(1, 1 + len(bottoms))
    fig.set_figheight(2)
    fig.set_figwidth(2 * (len(bottoms) + 1))

    axes[0].axis("off")
    axes[0].imshow(Image.open(os.path.join(DIR, str(top) + '.jpg')).resize((224, 224)))

    for i, (score, id) in enumerate(bottoms, start=1):
        axes[i].axis("off")
        axes[i].set_title(f'{score:.3f}')
        axes[i].imshow(Image.open(os.path.join(DIR, str(id) + '.jpg')).resize((224, 224)))
    plt.show()


def infer():
    model = StyleAwareNet(model_args)
    model_path = os.path.join(MODEL_PATH, f'{MODEL_NAME}.pth')
    model.load_state_dict(torch.load(model_path))
    print(f'Model successfully loaded')

    embed_generator = FashionEmbeddingGenerator()
    print(f'Embed Generator successfully loaded')

    recommender = FashionRecommender(model, embed_generator, device='cuda')

    top_ids = pd.read_json('F:/Projects/ai-stylist/data/polyvore_cleaned/top_embeds.json').index
    bottom_ids = pd.read_json('F:/Projects/ai-stylist/data/polyvore_cleaned/bottom_embeds.json').index

    top_id_choosed = 10240692# random.choice(top_ids)
    top_image = Image.open(os.path.join(DIR, str(top_id_choosed) + '.jpg'))

    bottom_ids_choosed = bottom_ids[0:256]# [random.choice(bottom_ids) for _ in range(64)]

    '''
    0:      "formal, dandy, modern and minimal",
    1:      "athletic and sports",
    2:      "casual and classic", 
    3:      "ethnic, hippie and maximalism", 
    4:      "hip-hop and street gangster", 
    5:      "preppy and classic", 
    6:      "feminine and girlish"
    '''

    scores = []
    for bottom_id in bottom_ids_choosed:
        bottom_image = Image.open(os.path.join(DIR, str(bottom_id) + '.jpg'))

        scores.append(recommender.single_infer(top_image, bottom_image, torch.LongTensor([6])).item())

    bottoms = sorted(list(zip(scores, bottom_ids_choosed)), key=lambda x: x[0], reverse=False)
    bottoms = bottoms[:5] + bottoms[-5:]
    show_images(top_id_choosed, bottoms)

if __name__ == '__main__':
    infer()