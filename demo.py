import os
import random
from embed_generator.generator import *
from embed_generator.visualization import *
from style_aware_net.model.style_aware_net import *
from inference import *
from PIL import Image
from tqdm import tqdm
import pandas as pd


MODEL_PATH = 'C:/KU/ai-stylist/ai-stylist/style_aware_net/model/saved_model'
MODEL_NAME = '0_0.683'
DIR = 'C:/KU/ai-stylist/ai-stylist/data/polyvore_outfits/images'


model_args = ModelArgs(
    n_conditions = 9
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

    torch.no_grad()
    top_ids = pd.read_json('C:/KU/ai-stylist/ai-stylist/data/polyvore_cleaned/top_embeds.json').index
    bottom_ids = pd.read_json('C:/KU/ai-stylist/ai-stylist/data/polyvore_cleaned/bottom_embeds.json').index

    top_id_choosed = random.choice(top_ids)
    top_image = Image.open(os.path.join(DIR, str(top_id_choosed) + '.jpg'))

    bottom_ids_choosed = bottom_ids[0:256]# [random.choice(bottom_ids) for _ in range(64)]

    '''
    styles = ["formal and minimal",
          "athletic and sports",
          "casual and daily", 
          "ethnic, hippie, and maximalism", 
          "hip-hop and street", 
          "preppy and classic", 
          "feminine and girlish"]
    '''

    # get top bottom embeddings first
    top_image = Image.open(os.path.join(DIR, str(top_id_choosed) + '.jpg'))
    top_embed = embed_generator.img2embed([top_image])
    
    bottom_embeds = []
    for bottom_id in bottom_ids_choosed:
        bottom_image = Image.open(os.path.join(DIR, str(bottom_id) + '.jpg'))
        bottom_embeds.append(embed_generator.img2embed(bottom_image))

    for i in range(7):
        scores = []
        for bottom_embed in tqdm(bottom_embeds):


            scores.append(recommender.single_infer(top_embed, bottom_embed, torch.LongTensor([i])).item())

        bottoms = sorted(list(zip(scores, bottom_ids_choosed)), key=lambda x: x[0], reverse=False)
        bottoms = bottoms[:5] + bottoms[-5:]
        show_images(top_id_choosed, bottoms)

if __name__ == '__main__':
    infer()