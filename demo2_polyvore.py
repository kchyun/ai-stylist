import os
import random
from embed_generator.generator import *
from embed_generator.visualization import *
from style_aware_net.model.style_aware_net import *
#from inference import *
from inference2 import *
from PIL import Image
from tqdm import tqdm
from test_style_classifier import *
import pandas as pd


MODEL_PATH = 'C:/KU/ai-stylist/ai-stylist/style_aware_net/model/saved_model'
MODEL_NAME = '2023-11-29_0_2.060'
DIR = 'C:/KU/ai-stylist/ai-stylist/data/polyvore_outfits/images'

model_args = ModelArgs()


def show_images(fashion_sets, style):
    fig, axes = plt.subplots(2, 10)
    fig.set_figheight(2)
    fig.set_figwidth(10)
    
    for i, fashion_set in enumerate(fashion_sets):
        top = fashion_set['top_id']
        bottom = fashion_set['bottom_id']
        score = fashion_set['score']
        print(f'top : {top} / bottom : {bottom} / score : {score}')
        
        axes[0, i].axis("off")
        axes[1, i].axis("off")
        axes[0, i].set_title(f'{score:.3f}')
        axes[0][i].imshow(Image.open(os.path.join(DIR, str(top) + '.jpg')).resize((224, 224)))
        axes[1][i].imshow(Image.open(os.path.join(DIR, str(bottom) + '.jpg')).resize((224,224)))

    plt.show()
    fig.savefig(f"./category_compatibility/{style}.png")
    plt.close()


def infer():
    model = StyleAwareNet(model_args)
    model_path = os.path.join(MODEL_PATH, f'{MODEL_NAME}.pth')
    model.load_state_dict(torch.load(model_path))

    styles = ["formal and minimal",
          "athletic and sports",
          "casual and daily", 
          "ethnic, hippie, and maximalism", 
          "hip-hop and street", 
          "preppy and classic", 
          "feminine and girlish"]

    print(f'Model successfully loaded')

    embed_generator = FashionEmbeddingGenerator()
    print(f'Embed Generator successfully loaded')

    recommender = FashionRecommender(model, embed_generator, device='cuda')
    styleclassifier = StyleClassifier(embed_generator, styles=styles)
    torch.no_grad()
    #top_ids = pd.read_json('C:/KU/ai-stylist/ai-stylist/data/polyvore_cleaned/top_embeds.json').index
    #bottom_ids = pd.read_json('C:/KU/ai-stylist/ai-stylist/data/polyvore_cleaned/bottom_embeds.json').index

    top_item = pd.read_json('./data/polyvore_cleaned/top_embeds.json')
    bottom_item = pd.read_json('./data/polyvore_cleaned/bottom_embeds.json')

    record_top = []
    for id, embed in tqdm(zip(top_item.index, torch.stack([torch.Tensor(list(e)) for e in top_item.values]).squeeze(1))):
        record_top.append({'id' : id, 'embed':embed, 'score' : styleclassifier.forward(embed, 0, 'cuda')})

    record_bottom = []
    for id, embed in tqdm(zip(bottom_item.index, torch.stack([torch.Tensor(list(e)) for e in bottom_item.values]).squeeze(1))):
            record_bottom.append({'id' : id, 'embed' : embed, 'score' : styleclassifier.forward(embed, 0, 'cuda')})

    for i in range(len(styles)):
        category = i

        record_top = sorted(record_top, key=lambda x: x['score'][i])
        record_bottom = sorted(record_bottom, key=lambda x: x['score'][i])
        
        pick_top = record_top[-200:]
        pick_bottom = record_bottom[-200:]

        print("Picked top and bottom")
        # import pdb; pdb.set_trace()
        score = recommender.single_infer(pick_top, pick_bottom)
        print("Scoring complete")
        show_images(score[:10], styles[category])

if __name__ == '__main__':
    infer()