import os
import gc
import lpips
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
import torchvision
from torchvision import transforms
from torchvision.models import vgg16
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

#import wandb
from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats

from pix2pix_turbo import Pix2Pix_Turbo
from my_utils.training_utils import parse_args_paired_training, LidarRefDataset

def gram_matrix(input_tensor):
    """计算输入特征图的Gram矩阵"""
    batch_size, channels, height, width = input_tensor.size()
    features = input_tensor.view(batch_size * channels, height * width)
    gram = torch.mm(features, features.t())
    return gram.div(batch_size * channels * height * width)

def get_vgg_features(image, model, layers):
    features = {}
    x = image
    conv_count = 0
    block_count = 1
    
    for idx, layer in enumerate(model):
        if isinstance(layer, torch.nn.Conv2d):
            conv_count += 1
            layer_name = f'conv{block_count}_{conv_count}'
            
            x = layer(x)
            if layer_name in layers:
                features[layer_name] = x
        elif isinstance(layer, torch.nn.MaxPool2d):
            # 每当遇到MaxPool2d时，意味着当前块结束，进入下一个块
            block_count += 1
            conv_count = 0
        else:
            # 对于ReLU等其他层，只需前向传播
            x = layer(x)
    
    return features

def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    if args.pretrained_model_name_or_path == "stabilityai/sd-turbo":
        net_pix2pix = Pix2Pix_Turbo(pretrained_path=None, lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae)
        net_pix2pix.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_pix2pix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_pix2pix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gan_disc_type == "vagan_clip":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
    else:
        raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")

    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    #net_clip, _ = clip.load("ViT-B/32", device="cuda")
    #net_clip.requires_grad_(False)
    #net_clip.eval()
    vgg = torchvision.models.vgg16(pretrained=True).features.cuda().eval()
    for param in vgg.parameters():
        param.requires_grad_(False)
    style_layers = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']

    net_lpips.requires_grad_(False)
    ssim_metric = StructuralSimilarityIndexMeasure().to("cuda")

    # make the optimizer
    layers_to_opt = []
    for n, _p in net_pix2pix.unet.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt += list(net_pix2pix.unet.conv_in.parameters())
    for n, _p in net_pix2pix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt = layers_to_opt + list(net_pix2pix.vae.decoder.skip_conv_1.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_2.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_3.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_4.parameters())

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles, power=args.lr_power)

    dataset_train = LidarRefDataset(dataset_folder=args.dataset_folder, image_prep=args.train_image_prep, split="train", tokenizer=net_pix2pix.tokenizer)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dataset_val = LidarRefDataset(dataset_folder=args.dataset_folder, image_prep=args.test_image_prep, split="test", tokenizer=net_pix2pix.tokenizer)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # Prepare everything with our `accelerator`.
    net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc
    )
    #net_clip, net_lpips = accelerator.prepare(net_clip, net_lpips)
    net_lpips, vgg = accelerator.prepare(net_lpips, vgg)
    # renorm with image net statistics
    t_clip_renorm = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    t_vgg_renorm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networksr to device and cast to weight_dtype
    net_pix2pix.to(accelerator.device, dtype=weight_dtype)
    net_disc.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    #net_clip.to(accelerator.device, dtype=weight_dtype)
    vgg.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    #if accelerator.is_main_process:
    #    tracker_config = dict(vars(args))
    #    accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # turn off eff. attn for the discriminator
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    # compute the reference stats for FID tracking
    '''
    if accelerator.is_main_process and args.track_val_fid:
        feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)

        def fn_transform(x):
            x_pil = Image.fromarray(x)
            out_pil = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.LANCZOS)(x_pil)
            return np.array(out_pil)

        ref_stats = get_folder_features(os.path.join(args.dataset_folder, "test_B"), model=feat_model, num_workers=0, num=None,
                shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                mode="clean", custom_image_tranform=fn_transform, description="", verbose=True)
    '''

    # start the training loop
    global_step = 0
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_pix2pix, net_disc]
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"]
                ref_src = batch["ref_input"]
                x_tgt = batch["output_pixel_values"]
                B, C, H, W = x_src.shape
                # 雷达覆盖
                ref_mask = torch.any(ref_src > 0.20, dim=1, keepdim=True).float()
                lidar_region = ref_src * ref_mask
                source_region = x_src * (1 - ref_mask)
                x_combined = lidar_region + source_region
                # forward pass
                x_tgt_pred = net_pix2pix(x_combined, prompt_tokens=batch["input_ids"], deterministic=True)
                # Reconstruction loss
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips
                loss_ssim = (1 - ssim_metric(x_tgt_pred.float(), x_tgt.float())) * args.lambda_ssim
                loss = loss_l2 + loss_lpips + loss_ssim
                # Gram Matrix Loss
                with torch.no_grad():
                    x_tgt_pred_renorm = t_vgg_renorm(x_tgt_pred * 0.5 + 0.5)
                    x_tgt_pred_renorm = F.interpolate(x_tgt_pred_renorm, (224, 224), mode="bilinear", align_corners=False)
                    x_tgt_renorm = t_vgg_renorm(x_tgt * 0.5 + 0.5)
                    x_tgt_renorm = F.interpolate(x_tgt_renorm, (224, 224), mode="bilinear", align_corners=False)
                    target_features = get_vgg_features(x_tgt_renorm, vgg, style_layers)
                    pred_features = get_vgg_features(x_tgt_pred_renorm, vgg, style_layers)
                loss_gram = 0
                for layer in style_layers:
                    target_gram = gram_matrix(target_features[layer])
                    pred_gram = gram_matrix(pred_features[layer])
                    loss_gram += F.mse_loss(pred_gram, target_gram) * 1.0
                loss += loss_gram
                # CLIP similarity loss
                '''
                if args.lambda_clipsim > 0:
                    x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred * 0.5 + 0.5)
                    x_tgt_pred_renorm = F.interpolate(x_tgt_pred_renorm, (224, 224), mode="bilinear", align_corners=False)
                    caption_tokens = clip.tokenize(batch["caption"], truncate=True).to(x_tgt_pred.device)
                    clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                    loss_clipsim = (1 - clipsim.mean() / 100)
                    loss += loss_clipsim * args.lambda_clipsim
                '''
                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                """
                Generator loss: fool the discriminator
                """
                x_tgt_pred = net_pix2pix(x_combined, prompt_tokens=batch["input_ids"], deterministic=True)
                lossG = net_disc(x_tgt_pred, for_G=True).mean() * args.lambda_gan
                accelerator.backward(lossG)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                """
                Discriminator loss: fake image vs real image
                """
                # real image
                lossD_real = net_disc(x_tgt.detach(), for_real=True).mean() * args.lambda_gan
                accelerator.backward(lossD_real.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                # fake image
                lossD_fake = net_disc(x_tgt_pred.detach(), for_real=False).mean() * args.lambda_gan
                accelerator.backward(lossD_fake.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                lossD = lossD_real + lossD_fake

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    #logs["lossG"] = lossG.detach().item()
                    #logs["lossD"] = lossD.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    logs["loss_ssim"] = loss_ssim.detach().item()
                    #logs["loss_l1dar"] = loss_lidar.detach().item()
                    logs["loss_gram"] = loss_gram.detach().item()
                    '''
                    if args.lambda_clipsim > 0:
                        logs["loss_clipsim"] = loss_clipsim.detach().item()
                    '''
                    progress_bar.set_postfix(**logs)

                    # viz some images
                    '''
                    if global_step % args.viz_freq == 1:
                        log_dict = {
                            "train/source": [wandb.Image(x_src[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/target": [wandb.Image(x_tgt[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/model_output": [wandb.Image(x_tgt_pred[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                        }
                        for k in log_dict:
                            logs[k] = log_dict[k]
                    '''

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_pix2pix).save_model(outf)

                    # compute validation set FID, L2, LPIPS, CLIP-SIM
                    if global_step % args.eval_freq == 1:
                        l_l2, l_lpips, l_clipsim = [], [], []
                        if args.track_val_fid:
                            os.makedirs(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)
                        for step, batch_val in enumerate(dl_val):
                            if step >= args.num_samples_eval:
                                break
                            x_src = batch_val["conditioning_pixel_values"].cuda()
                            ref_src = batch_val["ref_input"].cuda()
                            x_tgt = batch_val["output_pixel_values"].cuda()
                            B, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            # 雷达覆盖
                            ref_mask = torch.any(ref_src > 0.20, dim=1, keepdim=True).float()
                            lidar_region = ref_src * ref_mask
                            source_region = x_src * (1 - ref_mask)
                            x_combined = lidar_region + source_region
                            with torch.no_grad():
                                # forward pass
                                x_tgt_pred = accelerator.unwrap_model(net_pix2pix)(x_combined, prompt_tokens=batch_val["input_ids"].cuda(), deterministic=True)
                                # compute the reconstruction losses
                                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean")
                                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean()
                                # compute clip similarity loss
                                '''
                                x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred * 0.5 + 0.5)
                                x_tgt_pred_renorm = F.interpolate(x_tgt_pred_renorm, (224, 224), mode="bilinear", align_corners=False)
                                caption_tokens = clip.tokenize(batch_val["caption"], truncate=True).to(x_tgt_pred.device)
                                clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                                clipsim = clipsim.mean()
                                '''

                                l_l2.append(loss_l2.item())
                                l_lpips.append(loss_lpips.item())
                                #l_clipsim.append(clipsim.item())
                            # save output images to file for FID evaluation
                            if args.track_val_fid:
                                img_com = transforms.ToPILImage()(x_combined[0].cpu())
                                img_tgt_pred = transforms.ToPILImage()(x_tgt_pred[0].cpu() * 0.5 + 0.5)
                                combined_width = img_com.width * 2
                                combined_height = img_com.height
                                combined_img = Image.new('RGB', (combined_width, combined_height))
                                # 拼接图像
                                combined_img.paste(img_com, (0, 0))
                                combined_img.paste(img_tgt_pred, (img_com.width, 0))
                                outf = os.path.join(args.output_dir, "eval", f"fid_{global_step}", f"val_{step}.png")
                                combined_img.save(outf)
                        '''
                        if args.track_val_fid:
                            curr_stats = get_folder_features(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), model=feat_model, num_workers=0, num=None,
                                    shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                                    mode="clean", custom_image_tranform=fn_transform, description="", verbose=True)
                            fid_score = fid_from_feats(ref_stats, curr_stats)
                            logs["val/clean_fid"] = fid_score
                        '''
                        logs["val/l2"] = np.mean(l_l2)
                        logs["val/lpips"] = np.mean(l_lpips)
                        #logs["val/clipsim"] = np.mean(l_clipsim)
                        gc.collect()
                        torch.cuda.empty_cache()
                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
