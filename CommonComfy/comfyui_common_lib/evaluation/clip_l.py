import torch
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


class CLIP_L:
    def __init__(self, model_ID):
        self.model_ID = model_ID
        self.model = CLIPModel.from_pretrained(model_ID).to('cuda')
        self.preprocess = CLIPImageProcessor.from_pretrained(model_ID)

    def load_and_preprocess_image(self, img):
        # Check if img is a batch (4D tensor) or single image (3D tensor)
        if len(img.shape) == 3:
            # Single image case
            img = img.unsqueeze(0)  # Add batch dimension   
        elif len(img.shape) != 4:
            raise ValueError("Input image must be a 3D or 4D tensor")
        
        # Convert to PIL image and preprocess
        images = [transforms.ToPILImage()(img[i]) for i in range(img.shape[0])]
        preprocessed_images = self.preprocess(images, return_tensors="pt")
        return preprocessed_images
    
    def clip_image_similarity(self, tensor1, tensor2):
        # Load the two images and preprocess them for CLIP
        image_a = self.load_and_preprocess_image(tensor1)["pixel_values"]
        image_b = self.load_and_preprocess_image(tensor2)["pixel_values"]
        
        # Calculate the embeddings for the images using the CLIP model
        with torch.no_grad():
            embedding_a = self.model.get_image_features(image_a.to('cuda'))
            embedding_b = self.model.get_image_features(image_b.to('cuda'))

        # Calculate the cosine similarity between the embeddings
        similarity_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b, dim=-1)
        
        return similarity_score

if __name__ == "__main__":
    # Load the CLIP model
    model_ID = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_ID).to('cuda')
    preprocess = CLIPImageProcessor.from_pretrained(model_ID)
    
    path1 = f"image1.jpg"
    path2 = f"image2.jpg"
    
    # Load original and generated images as tensors
    img1 = torch.tensor(np.array(Image.open(path1))).permute(2,0,1).unsqueeze(0)
    img2 = torch.tensor(np.array(Image.open(path2))).permute(2,0,1).unsqueeze(0)
    
    # Calculate the cosine similarity between the images
    clip_l = CLIP_L(model_ID)
    similarity_score = clip_l.clip_image_similarity(img1, img2)
    print(f"Cosine similarity: {similarity_score}")