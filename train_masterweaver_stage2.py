import random
import os
import argparse
from pathlib import Path
import math
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from masterweaver.resampler import IDResampler as Resampler
from diffusers.optimization import get_scheduler
# if is_torch2_available():
from masterweaver.attention_processor import IDAttnProcessor2_0 as AttnProcessor
from datasets_laion import FilteredLaionFaceDataset
from tqdm import tqdm


def th2image(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)


imagenet_templates_small = [
    "a {}",
    "a photo of a {}",
    "a rendering of a {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of the cool {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of a cool {}",
]

class MasterWeaver(torch.nn.Module):

    def __init__(self, id_encoder, id_encoder_bk, adapter_modules):
        super().__init__()
        self.id_encoder = id_encoder
        self.id_encoder_bk = id_encoder_bk
        self.adapter_modules = adapter_modules

    def forward(self):
        pass


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def unfreeze_params(params):
    for param in params:
        param.requires_grad = True


@torch.no_grad()
def save_progress(model, accelerator, args, step=None):
    unwarped_model = accelerator.unwrap_model(model)
    state_dict = {'id_encoder': unwarped_model.id_encoder.state_dict(),
                  'adapter_modules': unwarped_model.adapter_modules.state_dict()}
    if step is not None:
        torch.save(state_dict, os.path.join(args.output_dir, f"adapter_{str(step).zfill(6)}.pt"))
    else:
        torch.save(state_dict, os.path.join(args.output_dir, "adapter_final.pt"))


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to pretrained adapter model.",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--lambda_edit",
        type=float,
        default=1.0,
        help="Weight of Editing Direction Loss.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./adapter_experiments",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=True,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--vis_steps",
        type=int,
        default=200,
        help=(
            "Visualize the intermediate test results every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)

    num_tokens = 16
    id_encoder = Resampler(
        dim=unet.config.cross_attention_dim,
        depth=1,
        dim_head=64,
        heads=12,
        num_queries=num_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4
    )

    id_encoder_bk = Resampler(
        dim=unet.config.cross_attention_dim,
        depth=1,
        dim_head=64,
        heads=12,
        num_queries=num_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4
    )

    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
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
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_img.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_img.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = AttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
        attn_procs[name].mname = name

    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    masterweaver_model = MasterWeaver(id_encoder, id_encoder_bk, adapter_modules)

    if args.adapter_path is not None:
        state_dict = torch.load(args.adapter_path, map_location='cpu')
        masterweaver_model.id_encoder.load_state_dict(state_dict['id_encoder'])
        masterweaver_model.adapter_modules.load_state_dict(state_dict['adapter_modules'])
        new_state = {}
        for k in state_dict['id_encoder']:
            new_state[k] = state_dict['id_encoder'][k].clone()
        masterweaver_model.id_encoder_bk.load_state_dict(new_state)
        print(args.adapter_path)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    # freeze parameters of models to save more memory
    freeze_params(vae.parameters())
    # freeze_params(unet.parameters())
    freeze_params(text_encoder.parameters())
    freeze_params(image_encoder.parameters())

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer = torch.optim.AdamW(
        [
            {"params": masterweaver_model.id_encoder.parameters(), "lr": args.learning_rate * 0.02},
        ],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # dataloader
    train_dataset = FilteredLaionFaceDataset(
        data_root=args.data_root_path,
        tokenizer=tokenizer,
        size=args.resolution,
        stylegan_aug=True
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    masterweaver_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(masterweaver_model, optimizer, train_dataloader, lr_scheduler)

    vae.eval()
    unet.eval()
    image_encoder.eval()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    edit_prompts = open('./data_scripts/prompt.txt').readlines()
    edit_prompts = [prompt.strip() for prompt in edit_prompts]

    for epoch in range(0, args.num_train_epochs):
        masterweaver_model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(masterweaver_model):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample().detach()
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                clip_images = []
                for clip_image, drop_image_embed in zip(batch["clip_images"], batch["drop_image_embeds"]):
                    if drop_image_embed == 1:
                        clip_images.append(torch.zeros_like(clip_image))
                    else:
                        clip_images.append(clip_image)
                clip_images = torch.stack(clip_images, dim=0)


                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]
                    selected_edit_prompts = random.sample(edit_prompts, batch["images"].shape[0])
                    text_temps = random.sample(imagenet_templates_small, batch["images"].shape[0])
                    genders = random.sample(['man', 'woman', 'male', 'female', 'girl', 'boy'], batch["images"].shape[0])
                    edit_prompts_ = [text_temp.format(edit_prompt.format(gender)) for text_temp, edit_prompt, gender in zip(text_temps, selected_edit_prompts, genders)]
                    edit_input = tokenizer(
                        edit_prompts_,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        return_tensors="pt",
                    )
                    edit_hidden_states = text_encoder(edit_input.input_ids.to(accelerator.device))[0]

                    rec_prompts = [text_temp.format(gender) for text_temp, gender in zip(text_temps, genders)]
                    rec_input = tokenizer(
                        rec_prompts,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        return_tensors="pt",
                    )
                    rec_hidden_states = text_encoder(rec_input.input_ids.to(accelerator.device))[0]

                    noise_edit = torch.randn_like(latents)
                    timesteps_edit = torch.randint(100, 350, (bsz,), device=latents.device)
                    timesteps_edit = timesteps_edit.long()
                    noisy_latents_edit = noise_scheduler.add_noise(latents, noise_edit, timesteps_edit)

                face_images = torch.chunk(clip_images, 3, dim=1)
                face_embeddings = []
                zero_face_embeddings = []
                for face_img in face_images:
                    image_embeds = image_encoder(face_img, output_hidden_states=True).hidden_states[-2]
                    face_embeddings.append(id_encoder(image_embeds))

                    zero_image_embeds = image_encoder(torch.zeros_like(face_img), output_hidden_states=True).hidden_states[-2]
                    zero_face_embeddings.append(id_encoder_bk(zero_image_embeds))

                face_embeddings = torch.cat(face_embeddings, 1)
                zero_face_embeddings = torch.cat(zero_face_embeddings, 1)

                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states,
                    cross_attention_kwargs={
                        "id_embedding": face_embeddings
                }).sample

                feature_edit = []
                feature_rec = []
                _ = unet(noisy_latents_edit, timesteps_edit, rec_hidden_states,
                    cross_attention_kwargs={
                       "id_embedding": face_embeddings,
                       "tensor_list": feature_rec,
                }).sample
                _ = unet(noisy_latents_edit, timesteps_edit, edit_hidden_states,
                    cross_attention_kwargs={
                        "id_embedding": face_embeddings,
                        "tensor_list": feature_edit,
                }).sample


                with torch.no_grad():
                    aug_lambda = torch.rand((noisy_latents.shape[0], 1, 1), device=accelerator.device) * 0.6 + 0.2
                    face_embeddings_aug = face_embeddings * aug_lambda + torch.flip(face_embeddings, dims=(0,)) * (1 - aug_lambda)
                    noise_pred_aug = unet(noisy_latents, timesteps, encoder_hidden_states,
                        cross_attention_kwargs={
                        "id_embedding": face_embeddings_aug
                    }).sample

                    feature_edit_sd = []
                    feature_rec_sd = []
                    _ = unet(noisy_latents_edit, timesteps_edit, rec_hidden_states,
                        cross_attention_kwargs={
                            "id_embedding": zero_face_embeddings,
                            "tensor_list": feature_rec_sd,
                    }).sample
                    _ = unet(noisy_latents_edit, timesteps_edit, edit_hidden_states,
                        cross_attention_kwargs={
                            "id_embedding": zero_face_embeddings,
                            "tensor_list": feature_edit_sd,
                    }).sample

                bbox_mask = F.interpolate(batch["face_bbox"], (64, 64), mode='nearest')
                bbox_mask = torch.sum(bbox_mask, dim=1, keepdim=True)
                bbox_mask = torch.clamp(bbox_mask, 0, 1)
                mask_region = 1 - bbox_mask.repeat(1, 4, 1, 1)
                face_region = 1 - mask_region
                loss_mle = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean") * 0.8
                loss_bg = F.mse_loss(noise_pred_aug.detach().float() * mask_region, noise_pred.float() * mask_region, reduction="mean")

                face_mask = face_region[:, 0:1, :, :]
                loss_edit = loss_mle * 0.0
                if global_step % 2 == 0:
                    for fea_edit, fea_rec, fea_edit_sd, fea_rec_sd in zip(feature_edit, feature_rec, feature_edit_sd, feature_rec_sd):
                        hh = int(math.sqrt(fea_edit_sd.shape[1]))
                        face_mask_ = F.interpolate(face_mask, (hh, hh), mode='bilinear').reshape((-1, hh * hh, 1))
                        delta_face_sd = (fea_edit_sd - fea_rec_sd) * face_mask_
                        delta_face = (fea_edit - fea_rec) * face_mask_
                        loss_edit += torch.sum(1 - torch.cosine_similarity(delta_face, delta_face_sd, dim=-1)) / (torch.sum(face_mask_) + 1e-13) * args.lambda_edit

                loss = loss_mle + loss_bg + loss_edit

                # Backpropagate
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(masterweaver_model.parameters(), 1)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                if global_step % args.save_steps == 0:
                    save_progress(masterweaver_model, accelerator, args, global_step)

                if global_step % args.vis_steps == 0:
                    prompts_ = ['a person smiling',
                                'a bald person swimming',
                                'a person with curly hair',
                                'Manga drawing of a person',
                                'a baby',
                                'a person wearing a spacesuit',
                                'photo of a person in the snow',
                                'a person is playing the guitar'] * 16

                    with torch.no_grad():
                        pipeline = StableDiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            vae=vae,
                            text_encoder=text_encoder,
                            unet=unet,
                            torch_dtype=weight_dtype,
                            feature_extractor=None,
                            safety_checker=None
                        )
                        pipeline.to(accelerator.device)

                        id_embeddings = []
                        zero_id_embeddings = []
                        for face_img in face_images:
                            image_embeds = image_encoder(face_img, output_hidden_states=True).hidden_states[-2]
                            id_embeddings.append(id_encoder(image_embeds))

                            zero_image_embeds = image_encoder(torch.zeros_like(face_img), output_hidden_states=True).hidden_states[-2]
                            zero_id_embeddings.append(id_encoder(zero_image_embeds))

                        id_embeddings = torch.cat(id_embeddings, 1)
                        zero_id_embeddings = torch.cat(zero_id_embeddings, 1)

                        syn_images = pipeline(prompt=prompts_[:latents.shape[0]],
                                              width=args.resolution, height=args.resolution,
                                              num_inference_steps=30,
                                              guidance_scale=5,
                                              cross_attention_kwargs={
                                                  'id_embedding': torch.cat((zero_id_embeddings, id_embeddings), 0),
                                                  'id_scale': 1},
                                              ).images
                        del pipeline
                        torch.cuda.empty_cache()

                    input_images = [th2image(img) for img in batch["images"]]
                    id_images0 = [th2image(img).resize((512, 512)) for img in face_images[0]]
                    id_images1 = [th2image(img).resize((512, 512)) for img in face_images[1]]
                    id_images2 = [th2image(img).resize((512, 512)) for img in face_images[2]]
                    face_bboxs = [th2image(img) for img in batch["face_bbox"].repeat((1, 3, 1, 1))]
                    img_list = []
                    for syn, input_img, id0, id1, id2, bbox in zip(syn_images, input_images, id_images0, id_images1, id_images2, face_bboxs,):
                        img_list.append(np.concatenate((np.array(syn), np.array(input_img), np.array(id0), np.array(id1), np.array(id2), np.array(bbox)), axis=1))

                    img_list = np.concatenate(img_list, axis=0)
                    Image.fromarray(img_list).save(os.path.join(args.output_dir, f"{str(global_step).zfill(5)}.jpg"))

            logs = {"loss_mle": loss_mle.detach().item(), "loss_bg": loss_bg.detach().item(), "loss_edit": loss_edit.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break


if __name__ == "__main__":
    main()