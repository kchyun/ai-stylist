import os
import sys
sys.path.append("C:/KU/ai-stylist/ai-stylist/")
sys.path.append("C:/KU/ai-stylist/ai-stylist/OutfitTransformer/")
sys.path.append("C:/KU/ai-stylist/ai-stylist/OutfitTransformer/outfit-transformer")

from embed_generator.generator import *
from embed_generator.visualization import *
from cp_inference import Predictor
from utils.data import MusinsaDataset
from PIL import Image
from tqdm import tqdm
from style_classifier import *
import pandas as pd


DATA_PATH = "C:/KU/ai-stylist/ai-stylist/data/musinsa-en/"


def show_images(fashion_sets, style):
    fig, axes = plt.subplots(3, 10)
    fig.set_figheight(3)
    fig.set_figwidth(10)
    
    for i, fashion_set in enumerate(fashion_sets):
        # top = fashion_set['top_id']
        # bottom = fashion_set['bottom_id']
        # score = fashion_set['score']
        # print(f'top : {top} / bottom : {bottom} / score : {score}')
        
        axes[0, i].axis("off")
        axes[1, i].axis("off")
        axes[2, i].axis("off")
        # axes[0, i].set_title(f'{score:.3f}')
        axes[0, i].imshow(Image.open(os.path.join(DATA_PATH, fashion_set['top']['image_path'][2:])).resize((224, 224)))
        axes[1, i].imshow(Image.open(os.path.join(DATA_PATH, fashion_set['bottom']['image_path'][2:])).resize((224,224)))
        axes[2, i].imshow(Image.open(os.path.join(DATA_PATH, fashion_set['shoes']['image_path'][2:])).resize((224,224)))

    plt.show()
    fig.savefig(f"./{style}.png")
    plt.close()

def get_embeds(items: List, 
               embed_generator: FashionEmbeddingGenerator):
    print("Getting embeds...")
    images = []
    for item in items:
        images.append(Image.open(os.path.join(DATA_PATH, item['img_path'][2:])).convert("RGB"))
        
    return embed_generator.img2embed(images)

def get_outfits(top, bottom, shoes):
    outfits = []

    for t in top:
        for b in bottom:
            for s in shoes:
                outfit = {'top': t, 'bottom': b, 'shoes': s}
                outfits.append(outfit)
                
    return outfits

def infer():
    
    styles = ["formal and minimal",
          "athletic and sports",
          "casual and daily", 
          "ethnic, hippie, and maximalism", 
          "hip-hop and street", 
          "preppy and classic", 
          "feminine and girlish"]

    embed_generator = FashionEmbeddingGenerator()
    print(f'Embed Generator successfully loaded')

    styleclassifier = StyleClassifier(embed_generator, styles=styles)
    torch.no_grad()
    print(f'Style Classifier successfully loaded')
    
    # Get data
    data = MusinsaDataset(DATA_PATH)
    
    
    # Embed data
    top_embeds = get_embeds(data.top_items, embed_generator)
    bottom_embeds = get_embeds(data.bottom_items, embed_generator)
    shoes_embeds = get_embeds(data.shoes_items, embed_generator)
    
    # Make item dict
    record_top = []
    for item, embed in tqdm(zip(data.top_items, top_embeds)):
        record_top.append({'image_path' : os.path.join(DATA_PATH, item['img_path'][2:]), 'desc': item["desc"], 'score' : styleclassifier.forward(embed, 0, 'cuda')})

    record_bottom = []
    for item, embed in tqdm(zip(data.bottom_items, bottom_embeds)):
        record_bottom.append({'image_path' : os.path.join(DATA_PATH, item['img_path'][2:]), 'desc': item["desc"], 'score' : styleclassifier.forward(embed, 0, 'cuda')})

    record_shoes = []
    for item, embed in tqdm(zip(data.shoes_items, shoes_embeds)):
        record_shoes.append({'image_path' : os.path.join(DATA_PATH, item['img_path'][2:]), 'desc': item["desc"], 'score' : styleclassifier.forward(embed, 0, 'cuda')})

    # 모델 정의
    predictor = Predictor()
    print("Model successfully loaded")

    for i in range(len(styles)):
        category = i
        
        # 각 스타일에 어울리는 순서대로 아이템 정렬
        record_top = sorted(record_top, key=lambda x: x['score'][i])
        record_bottom = sorted(record_bottom, key=lambda x: x['score'][i])
        record_shoes = sorted(record_shoes, key=lambda x: x['score'][i])
        
        # top N개 선택
        N = 5
        pick_top = record_top[-5:]
        pick_bottom = record_bottom[-5:]
        pick_shoes = record_shoes[-5:]

        print("Picked top and bottom")

        # outfit 형태로 조합
        
        outfits = get_outfits(pick_top, pick_bottom, pick_shoes)

        score = []
        for o in outfits:
            score.append(predictor.predict(o))
        
        topk = torch.topk(torch.Tensor(score), 10)
        reranked_outfits = [outfits[idx] for idx in topk.indices]

        print("Scoring complete")
        
        # visualize top 10 outfits
        show_images(reranked_outfits[-10:], styles[category])

if __name__ == '__main__':
    infer()