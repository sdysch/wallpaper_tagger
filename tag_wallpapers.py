import os
import argparse
import torch
import clip
from PIL import Image
import csv

CATEGORIES = ['nature', 'city', 'space', 'abstract', 'animals', 'technology']


def process_images(folder_path, top_k=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    model, preprocess = clip.load('ViT-B/32', device=device)
    text_tokens = clip.tokenize(CATEGORIES).to(device)

    tags_dict = {}

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            image_path = os.path.join(folder_path, filename)
            try:
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

                with torch.no_grad():
                    image_features = model.encode_image(image)
                    text_features = model.encode_text(text_tokens)
                    similarity = (image_features @ text_features.T).softmax(dim=-1)

                    values, indices = similarity.topk(top_k)
                    image_tags = [CATEGORIES[i] for i in indices[0]]
                    tags_dict[filename] = image_tags

                    # ===== RENAME FILE =====
                    name, ext = os.path.splitext(filename)
                    new_name = f'{name}_{'_'.join(image_tags)}{ext}'
                    new_path = os.path.join(folder_path, new_name)
                    os.rename(image_path, new_path)

                    print(f'{filename} -> {image_tags} | Renamed to: {new_name}')

            except Exception as e:
                print(f'Failed to process {filename}: {e}')

    return tags_dict


def save_csv(tags_dict, output_csv):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'tags'])
        for filename, tags in tags_dict.items():
            writer.writerow([filename, ', '.join(tags)])
    print(f'Tags saved to {output_csv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tag and optionally rename wallpapers using CLIP.')
    parser.add_argument('folder', help='Path to folder containing wallpapers')
    parser.add_argument('--tag', help='Save tags CSV to this file', default=None)
    parser.add_argument('--top-k', type=int, default=1, help='Number of top tags to include in filename')

    args = parser.parse_args()

    tags_dict = process_images(args.folder, top_k=args.top_k)

    if args.tag:
        save_csv(tags_dict, args.tag)

