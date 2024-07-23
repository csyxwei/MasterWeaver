import torch
import random
import os
import spaces
import gradio as gr
from diffusers.utils import load_image
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

port = 7860
encoder_path = f"./pretrained/adapter_final.pt"

num_tokens = 16
dtype = torch.float16
device = "cuda"

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

@spaces.GPU
def generate_image(upload_images, prompt, negative_prompt, num_steps, num_outputs,
                   guidance_scale, id_scale, seed, progress=gr.Progress(track_tqdm=True)):

    if upload_images is None:
        raise gr.Error(f"Cannot find any input face image!")

    input_id_images = []
    for img in upload_images:
        input_id_images.append(load_image(img))

    aligned_id_images = []
    for img_idx, img in enumerate(input_id_images):
        image_np = np.array(img)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        dets = detector(gray, 1)
        if len(dets) > 0:
            face_landmarks = [(item.x, item.y) for item in shape_predictor(gray, dets[0]).parts()]
            aligned_image = image_align(image_np, face_landmarks, transform_size=512)
            aligned_image = Image.fromarray(aligned_image.astype('uint8'))
            aligned_id_images.append(evaluate(aligned_image, net, mode='face'))
        else:
            print('not detected faces')
            aligned_id_images.append(evaluate(img, net, mode='face'))

    input_id_images = aligned_id_images
    print("Start inference...")
    print(f"[Debug] {len(input_id_images)} images, Prompt: {prompt}, \n[Debug] Neg Prompt: {negative_prompt}")

    images = pipe.generate(pil_image=input_id_images, num_samples=num_outputs, num_inference_steps=num_steps, seed=seed, guidance_scale=guidance_scale,
                           id_scale=id_scale, prompt=prompt, negative_prompt=negative_prompt)

    return images + input_id_images, gr.update(visible=True)


def swap_to_gallery(images):
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)


def upload_example_to_gallery(images, prompt, seed, id_scale):
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)


def remove_back_to_files():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)


def remove_cite():
    return gr.update(visible=False)


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def get_image_path_list(folder_name):
    image_basename_list = os.listdir(folder_name)
    image_path_list = sorted([os.path.join(folder_name, basename) for basename in image_basename_list])
    return image_path_list

def get_example():
    case = [
        [
            ['./test_data/hinton.jpg'],
            "photo of a man with red hair",
            42,
            1.0,
        ],
        [
            ['./test_data/lecun.jpg'],
            "a man in Iron man suit",
            42,
            1.5,
        ],
        [
            ['./test_data/musk.jpg'],
            "Manga drawing of a man",
            42,
            1.3,
        ],
        [
            ['./test_data/taylor.jpg'],
            "a woman in the snow",
            42,
            1.0,
        ],
        [
            ['./test_data/dilireba.jpg'],
            "photo of a woman with curly hair",
            42,
            0.9,
        ],
    ]
    return case


title = r"""
<h1 align="center">MasterWeaver:  Taming Editability and Identity for Personalized Text-to-Image Generation</h1>
Code: <a href='https://github.com/csyxwei/MasterWeaver' target='_blank'>GitHub</a>. Paper: <a href='https://arxiv.org/abs/2405.05806' target='_blank'>ArXiv</a>.
"""

description = r"""
❗️❗️❗️ Tips:<br>
- Upload images of the person you want to customize and enter a text prompt, then click the <b>Submit</b> button to start customizing.
- A single ID image is usually sufficient, but you can also supplement with additional auxiliary images.
- You can adjust the ID Scale to control the scale of identity. The higher the number, the greater the ID fidelity.
"""

article = r"""
If MasterWeaver is helpful, please help to ⭐ the <a href='https://github.com/csyxwei/MasterWeaver' target='_blank'>Github Repo</a>. Thanks! 
---
📝 **Citation**
<br>
If our work is useful for your research, please consider citing:

```bibtex
@inproceedings{wei2024masterweaver,
  title={MasterWeaver: Taming Editability and Face Identity for Personalized Text-to-Image Generation},
  author={Wei, Yuxiang and Ji, Zhilong and Bai, Jinfeng and Zhang, Hongzhi and Zhang, Lei and Zuo, Wangmeng},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```
📋 **License**
<br>
Apache-2.0 LICENSE.

📧 **Contact**
<br>
If you have any questions, please feel free to reach me out at <b>yuxiang.wei.cs@gmail.com</b>.

**Acknowledgements** 
<br>
This demo page is built on [PhotoMaker demo](https://huggingface.co/spaces/TencentARC/PhotoMaker). We thank the authors for sharing the demos.
"""


css = '''
.gradio-container {width: 85% !important}
'''
with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            files = gr.File(
                label="Drag (Select) 1 or more photos of your face",
                file_types=["image"],
                file_count="multiple"
            )
            uploaded_files = gr.Gallery(label="Your images", visible=False, columns=5, rows=1, height=200)
            with gr.Column(visible=False) as clear_button:
                remove_and_reupload = gr.ClearButton(value="Remove and upload new ones", components=files, size="sm")
            prompt = gr.Textbox(label="Prompt",
                                info="Try something like 'a photo of a man/woman'.",
                                placeholder="A photo of a [man/woman]...")
            submit = gr.Button("Submit")

            with gr.Accordion(open=False, label="Advanced Options"):
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="low quality",
                    value="nsfw, low quality, multiple faces, small face, watermark, text, missing fingers",
                )
                num_steps = gr.Slider(
                    label="Number of sample steps",
                    minimum=20,
                    maximum=100,
                    step=1,
                    value=50,
                )
                num_outputs = gr.Slider(
                    label="Number of output images",
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=4,
                )
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1,
                    value=5,
                )
                id_scale = gr.Slider(
                    label="ID Scale",
                    minimum=0.0,
                    maximum=3.0,
                    step=0.1,
                    value=0.9,
                )
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=False)
        with gr.Column():
            gallery = gr.Gallery(label="Generated Images")
            cite = gr.Markdown(label="Usage tips of MasterWeaver", value=article, visible=False)

        files.upload(fn=swap_to_gallery, inputs=files, outputs=[uploaded_files, clear_button, files])
        remove_and_reupload.click(fn=remove_back_to_files, outputs=[uploaded_files, clear_button, files])

        submit.click(
            fn=remove_cite,
            outputs=cite,
        ).then(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=generate_image,
            inputs=[files, prompt, negative_prompt, num_steps, num_outputs, guidance_scale, id_scale, seed],
            outputs=[gallery, cite]
        )

    gr.Examples(
        examples=get_example(),
        inputs=[files, prompt, seed, id_scale],
        run_on_click=True,
        fn=upload_example_to_gallery,
        outputs=[uploaded_files, clear_button, files],
    )

demo.launch(server_port=port)