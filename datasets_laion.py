from PIL import Image
from torchvision import transforms
import os
import PIL
from torch.utils.data import Dataset
import numpy as np
import torch
import random
import cv2

class FilteredLaionFaceDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        stylegan_aug=False
    ):

        self.i_drop_rate = 0.05
        self.t_drop_rate = 0.05
        self.ti_drop_rate = 0.05

        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size

        self.image_dir = os.path.join(data_root, 'images_cropped')
        self.face_dir = os.path.join(data_root, 'images_cropped_face')
        self.caption_dir = os.path.join(data_root, 'captions')
        self.mask_dir = os.path.join(data_root, 'images_cropped_face_mask')
        self.aug_face_dir = os.path.join(data_root, 'images_cropped_face_aug')

        image_files = os.listdir(self.image_dir)
        image_files = [file for file in image_files if file.endswith('jpg')]
        image_files = [os.path.join(self.image_dir, file) for file in image_files]

        self.image_files = image_files
        self._length = len(image_files)

        if stylegan_aug:
            self.datatype_list = [0, 1, 1, 1, 2, 2]
        else:
            self.datatype_list = [0, 0, 0, 0, 0, 0]

    def __len__(self):
        return self._length
        # return len(self.image_files_laion)

    def get_tensor_clip(self, normalize=True, toTensor=True, aug=False):

        #### augmentation
        transform_list = []

        transform_list.append(transforms.RandomHorizontalFlip(0.5))

        if aug:
            if random.uniform(0, 1) > 0.5:
                expand = random.uniform(0, 1) > 0.5
                transform_list.append(transforms.RandomRotation(45, interpolation=transforms.InterpolationMode.BILINEAR, expand=expand))

            if random.uniform(0, 1) > 0.5:
                transform_list.append(transforms.Resize(332, interpolation=transforms.InterpolationMode.BILINEAR))
                transform_list.append(transforms.RandomCrop(224))
            else:
                transform_list.append(transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR))
        else:
            transform_list.append(transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR))

        if toTensor:
            transform_list += [transforms.ToTensor()]
        if normalize:
            transform_list += [transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))]

        return transforms.Compose(transform_list)

    def process(self, image):
        img = np.array(image)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        img = np.array(img).astype(np.float32)
        img = img / 127.5 - 1.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def extract_ids(self, prompt):
        input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return input_ids

    def __getitem__(self, i):

        example = {}

        image_path = self.image_files[i]
        image = Image.open(image_path).convert("RGB")

        face_id = os.path.basename(image_path)[:-4]

        face_dir = os.path.join(self.face_dir, face_id)
        mask_dir = os.path.join(self.mask_dir, face_id)
        aug_face_dir = os.path.join(self.aug_face_dir, face_id)

        if os.path.exists(os.path.join(self.caption_dir, face_id + '.txt')):
            f = open(os.path.join(self.caption_dir, face_id + '.txt')).readlines()
            prompt = f[0].strip()
        else:
            prompt = ''

        face_files = [file for file in os.listdir(face_dir) if file.endswith('jpg')]

        face_images = []
        aug_face_images = []
        face_bbox = np.zeros_like(np.array(image))

        datatype = random.choice(self.datatype_list)
        for face_idx, face_file in enumerate(face_files):
            face_image = np.array(Image.open(os.path.join(face_dir, face_file)).resize((512, 512), PIL.Image.BICUBIC).convert("RGB"))
            if os.path.exists(os.path.join(mask_dir, face_file[:-4] + '.png')):
                parsing = np.array(Image.open(os.path.join(mask_dir, face_file[:-4] + '.png')).resize((512, 512), PIL.Image.NEAREST))
                parsing = np.where(parsing == 14, 0, parsing)
                parsing = np.where(parsing == 15, 0, parsing)
                parsing = np.where(parsing == 16, 0, parsing)
                parsing = np.where(parsing == 18, 0, parsing)
                if datatype == 1:
                    parsing = np.where(parsing == 17, 0, parsing)
                    parsing = np.where(parsing == 11, 0, parsing)
                    parsing = np.where(parsing == 12, 0, parsing)
                    parsing = np.where(parsing == 13, 0, parsing)
                face_mask = np.where(parsing > 0, 1, 0)
                face_mask = np.array(Image.fromarray(face_mask.astype('uint8')).convert('RGB'))
                face_image = face_image * face_mask

            face_images.append(face_image)

            lm_path = os.path.join(face_dir, face_file.replace('jpg', 'npy'))
            if os.path.exists(lm_path):
                face_lm = np.load(lm_path)
                x_min, x_max = int(np.min(face_lm[:, 1])), int(np.max(face_lm[:, 1]))
                y_min, y_max = int(np.min(face_lm[:, 0])), int(np.max(face_lm[:, 0]))
                x_c, y_c = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
                x_r, y_r = int((x_max - x_min) / 2), int((y_max - y_min) / 2)
                ratio = 1.5
                x_min = max(0, int(x_c - x_r * 2.2))
                x_max = int(x_c + x_r * ratio)
                y_min = max(0, int(y_c - y_r * ratio))
                y_max = int(y_c + y_r * ratio)
                face_bbox[x_min:x_max, y_min:y_max] = 1

        face_bbox = Image.fromarray(face_bbox.astype('uint8')).resize((512, 512), PIL.Image.NEAREST)
        face_bbox = torch.from_numpy(np.array(face_bbox)[:, :, 0]).unsqueeze(0).float()

        try:
            aug_face_files = os.listdir(os.path.join(aug_face_dir, face_file[:-4]))
            aug_face_file = random.choice(aug_face_files)
            aug_face_image = np.array(Image.open(os.path.join(aug_face_dir, face_file[:-4], aug_face_file)).resize((512, 512), PIL.Image.BICUBIC).convert("RGB"))
            aug_face_images.append(aug_face_image)
        except:
            aug_face_images = []

        if datatype == 1:
            face_images = face_images + aug_face_images
        elif datatype == 2:
            face_images = aug_face_images

        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            prompt = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            prompt = ""
            drop_image_embed = 1

        example["text_input_ids"] = self.extract_ids(prompt)
        example["texts"] = prompt
        example["drop_image_embeds"] = drop_image_embed
        example["images"] = self.process(image)

        ref_images = []
        for face_image in face_images:
            ref_image_tensor = self.get_tensor_clip()(Image.fromarray(face_image))
            ref_images.append(ref_image_tensor)

        if len(ref_images) < 3:
            if len(face_images) > 0 and random.uniform(0, 1) > 0.9:
                ref_images += [self.get_tensor_clip(aug=True)(Image.fromarray(face_images[0])) for _ in range(3 - len(ref_images))]
            else:
                ref_images += [torch.zeros(3, 224, 224) for _ in range(3 - len(ref_images))]

        ref_images = ref_images[:3]

        example["face_bbox"] = face_bbox
        example["clip_images"] = torch.cat(ref_images, dim=0)

        return example
