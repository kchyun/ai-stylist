import os
import random
from embed_generator.generator import *
from embed_generator.visualization import *
from style_aware_net.model.style_aware_net import *
from inference import *
from PIL import Image
from tqdm import tqdm
import pandas as pd
from style_aware_net.model.style_classifier import *

MODEL_PATH = 'F:/Projects/2023-ai-stylist/style_aware_net/model/saved_model'
MODEL_NAME = '2023-11-29_0_10.026'
DIR = 'F:/Projects/2023-ai-stylist/data/polyvore_outfits/images'

model_args = ModelArgs(
    n_conditions = 6
)

def show_images(top, pos_bottoms, neg_bottoms):
    fig, axes = plt.subplots(3, len(pos_bottoms))
    fig.set_figheight(3 * 3)
    fig.set_figwidth(3 * (len(pos_bottoms)))

    axes[0][0].axis("off")
    axes[0][0].imshow(Image.open(os.path.join(DIR, str(top) + '.jpg')).resize((224, 224)))

    for i, (score, id) in enumerate(pos_bottoms, start=0):
        axes[1][i].axis("off")
        axes[1][i].set_title(f'{score:.3f}')
        axes[1][i].imshow(Image.open(os.path.join(DIR, str(id) + '.jpg')).resize((224, 224)))

    for i, (score, id) in enumerate(neg_bottoms, start=0):
        axes[2][i].axis("off")
        axes[2][i].set_title(f'{score:.3f}')
        axes[2][i].imshow(Image.open(os.path.join(DIR, str(id) + '.jpg')).resize((224, 224)))
    plt.show()


def infer():
    # styles = [
    #    "office working and business meeting",
    #    "athletic, sports and physical activity",
    #    "casual day and picnic", 
    #    "ethnic, hippie, and maximalism", 
    #    "hip-hop", 
    #    "party", 
    #    "vacation and travel",
    #    ]

    styles = [
       "formal business meeting",
       "athletic running",
       "casual day", 
       "hip-hop", 
       "party", 
       "beach vacation",
       ]

    model = StyleAwareNet(model_args)
    model_path = os.path.join(MODEL_PATH, f'{MODEL_NAME}.pth')
    model.load_state_dict(torch.load(model_path))
    print(f'Model successfully loaded')

    embed_generator = FashionEmbeddingGenerator()
    print(f'Embed Generator successfully loaded')


    style_classifier = StyleClassifier(embed_generator=embed_generator, styles=styles)

    recommender = FashionRecommender(model, embed_generator, device='cuda')

    top_ids = pd.read_json('F:/Projects/2023-ai-stylist/data/polyvore_cleaned/top_ids.json').index
    bottom_ids = pd.read_json('F:/Projects/2023-ai-stylist/data/polyvore_cleaned/bottom_ids.json').index


    top, bottom = pd.read_json('F:/Projects/2023-ai-stylist/data/polyvore_cleaned/top_ids.json'), pd.read_json('F:/Projects/2023-ai-stylist/data/polyvore_cleaned/bottom_ids.json')

    top_id_choosed = 47609# 6661187# random.choice(top_ids)
    top_image = Image.open(os.path.join(DIR, str(top_id_choosed) + '.jpg'))

    bottom_ids_choosed = bottom_ids[0:2048]# [random.choice(bottom_ids) for _ in range(64)]

    scores = []
    for bottom_id in tqdm(bottom_ids_choosed):
        bottom_image = Image.open(os.path.join(DIR, str(bottom_id) + '.jpg'))

        # top = embed_generator.img2embed(top_image)
        # bottom = embed_generator.img2embed(bottom_image)

        scores.append(recommender.single_infer([top.loc[top_id_choosed, 'embed']], [bottom.loc[bottom_id, 'embed']], torch.LongTensor([5])).item())

    bottoms = sorted(list(zip(scores, bottom_ids_choosed)), key=lambda x: x[0], reverse=True)

    for b in bottoms[:10]:
        print(style_classifier.forward(torch.Tensor([top.loc[top_id_choosed, 'embed']]), torch.Tensor([bottom.loc[b[1], 'embed']]), torch.device('cpu')))
        print()

    show_images(top_id_choosed, bottoms[:10], bottoms[-10:])

if __name__ == '__main__':
    infer()