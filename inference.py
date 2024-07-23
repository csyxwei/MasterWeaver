import torch
import os
import dlib
from PIL import Image
from masterweaver.pipeline import MasterWeaverPipeline
import numpy as np
import sys

sys.path.append("./data_scripts/")
from face_parsing import evaluate, BiSeNet
from face_alignment import image_align
import cv2

MAX_SEED = np.iinfo(np.int32).max

base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "openai/clip-vit-large-patch14"


encoder_path = './pretrained/adapter_final.pt'

num_tokens = 16
dtype = torch.float16
device = "cuda"

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

pipe = MasterWeaverPipeline(base_model_path, vae_model_path, image_encoder_path, encoder_path, device, num_tokens=num_tokens, dtype=dtype)

####### load parsing
n_classes = 19
net = BiSeNet(n_classes=n_classes)
net.cuda()
net.load_state_dict(torch.load('./pretrained/79999_iter.pth'))
net.eval()

#### load dlib
model_path = os.path.join('./pretrained/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(model_path)

image_paths = ['./test_data/hinton.jpg']

input_id_images = []
for img_idx, img_path in enumerate(image_paths):
    image = Image.open(img_path)
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    dets = detector(gray, 1)
    if len(dets) > 0:
        face_landmarks = [(item.x, item.y) for item in shape_predictor(gray, dets[0]).parts()]
        aligned_image = image_align(image_np, face_landmarks, transform_size=512)
        aligned_image = Image.fromarray(aligned_image.astype('uint8'))
        input_id_images.append(evaluate(aligned_image, net, mode='face'))
    else:
        input_id_images.append(evaluate(image, net, mode='face'))

prompt='a man with red hair'
images = pipe.generate(pil_image=input_id_images, num_samples=4, num_inference_steps=30, seed=42, guidance_scale=5, id_scale=1, prompt=prompt, negative_prompt='')

grid = image_grid(input_id_images + images, 1, len(input_id_images) + 4)
grid.save('./output.jpg')
