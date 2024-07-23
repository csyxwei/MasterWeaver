import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
from masterweaver.resampler import IDResampler
from masterweaver.utils import is_torch2_available

if is_torch2_available():
    from masterweaver.attention_processor import IDAttnProcessor2_0 as IDAttnProcessor
else:
    from masterweaver.attention_processor import IDAttnProcessor

from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

class MasterWeaverPipeline:

    def __init__(self, sd_pipe_path, vae_model_path, image_encoder_path, encoder_path, device, num_tokens=4, dtype=torch.float32):

        self.device = device
        self.image_encoder_path = image_encoder_path
        self.encoder_path = encoder_path
        self.num_tokens = num_tokens
        self.dtype = dtype

        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=dtype)

        # load SD pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            sd_pipe_path,
            torch_dtype=dtype,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None
        )

        self.pipe = pipe.to(self.device)
        self.set_attn_processor()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(self.device, dtype=self.dtype)
        self.clip_image_processor = CLIPImageProcessor()
        # id encoder
        self.id_encoder = self.init_encoder()

        self.load_pretrain()

    def init_encoder(self):
        id_encoder = IDResampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=1,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4
        ).to(self.device, dtype=self.dtype)
        return id_encoder

    def set_attn_processor(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = IDAttnProcessor()
            else:
                attn_procs[name] = IDAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim).to(self.device, dtype=self.dtype)
            attn_procs[name].mname = name
        unet.set_attn_processor(attn_procs)

    def load_pretrain(self):
        state_dict = torch.load(self.encoder_path, map_location="cpu")
        self.id_encoder.load_state_dict(state_dict['id_encoder'])
        adapter_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        adapter_layers.load_state_dict(state_dict['adapter_modules'])

    def get_id_embedding(self, pil_image, num_samples):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]

        clip_images = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_images = clip_images.to(self.device, dtype=self.dtype)

        image_features = self.image_encoder(clip_images, output_hidden_states=True).hidden_states[-2]
        face_embeddings = self.id_encoder(image_features.detach())

        image_features = self.image_encoder(torch.zeros_like(clip_images), output_hidden_states=True).hidden_states[-2]
        zero_face_embeddings = self.id_encoder(image_features.detach())

        face_embeddings = face_embeddings.reshape(1, -1, face_embeddings.shape[-1]).repeat((num_samples, 1, 1))
        zero_face_embeddings = zero_face_embeddings.reshape(1, -1, zero_face_embeddings.shape[-1]).repeat((num_samples, 1, 1))

        ref_num = clip_images.shape[0]
        if ref_num < 3:
            face_embeddings = torch.cat([face_embeddings] + [zero_face_embeddings[:, :self.num_tokens]]* (3 - ref_num) , 1)
            zero_face_embeddings = torch.cat([zero_face_embeddings] + [zero_face_embeddings[:, :self.num_tokens]] * (3 - ref_num), 1)

        return zero_face_embeddings, face_embeddings

    def generate(
            self,
            pil_image,
            prompt=None,
            negative_prompt='',
            num_samples=4,
            seed=None,
            guidance_scale=7.5,
            num_inference_steps=30,
            id_scale=1.0,
            **kwargs,
    ):

        zero_face_embeddings, face_embeddings = self.get_id_embedding(pil_image, num_samples)
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_samples,
            height=512,
            width=512,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            cross_attention_kwargs={'id_embedding': torch.cat((zero_face_embeddings, face_embeddings), 0), 'id_scale': id_scale},
        ).images
        return images


