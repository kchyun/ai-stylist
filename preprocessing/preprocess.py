import PIL.Image as Image
import numpy as np
import cv2
import argparse
import os

from process import load_seg_model, get_palette, generate_mask
    

def preprocess_image(args, model, image, image_id, device):
    
    palette = get_palette(4)
    alpha_masks, cloth_seg = generate_mask(image, net=model, palette=palette, device=device)
    # print(alpha_masks)
    results = apply_masking_and_centering(image, alpha_masks, cloth_seg)
    
    category = ["", "upper", "lower", "dress"]
    
    for c in category:
        save_path = os.path.join(args.save_path, c)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
    
    save_image(results, image_id, category, args.save_path)

    
def apply_masking_and_centering(image, alpha_masks, cloth_seg):
    
    results = {}
    for i in alpha_masks.keys():
        
        masked = cv2.copyTo(np.array(image), np.array(alpha_masks[i]), np.full_like(image, 0.))
        nonzero_coords = np.nonzero(np.array(masked))

        # get bounding box
        height = masked.shape[0]
        width = masked.shape[1]

        margin = 5

        min_x = max(min(nonzero_coords[0]) - margin, 0) 
        min_y = max(min(nonzero_coords[1]) - margin, 0)

        max_x = min(max(nonzero_coords[0]) + margin, height)
        max_y = min(max(nonzero_coords[1]) + margin, width)

        results[i] = Image.fromarray(cv2.copyTo(np.array(image), np.array(alpha_masks[i]), np.full_like(image, 255.))).crop((min_y, min_x, max_y, max_x))

    return results

def save_image(results, image_id, category, save_path):

    # save image to the corresponding category
    for i in results.keys():
        filename = save_path + "/" + category[i] + "/" + image_id + ".png"
        results[i].save(filename, "png")
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Help to set arguments for Cloth Segmentation.')
    parser.add_argument('--input_path', type=str, help='Path to the input image')
    parser.add_argument('--cuda', action='store_true', help='Enable CUDA (default: False)')
    parser.add_argument('--checkpoint_path', type=str, default='model/cloth_segm.pth', help='Path to the checkpoint file')
    parser.add_argument('--save_path', type=str, default='preprocessed_image/', help='Path to save preprocessed image')
    args = parser.parse_args()
    
    device = 'cuda:0' if args.cuda else 'cpu'
    
    # load model
    model = load_seg_model(args.checkpoint_path, device=device)

    targets = []
    # process all images in directory
    if os.path.isdir(args.input_path):
        targets = os.listdir(args.input_path)
    # open one image
    else:
        targets.append(args.input_path)
    
    # preprocess image one by one
    for t in targets:
        image = Image.open(os.path.join(args.input_path, t)).convert('RGB')
        image_id = os.path.splitext(os.path.basename(t))[0]
        print(image_id)
        preprocess_image(args, model, image, image_id, device)
    