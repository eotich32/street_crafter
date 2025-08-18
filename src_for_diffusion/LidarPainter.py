import os
import requests
import sys
import copy
from tqdm import tqdm
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
p = "src_for_diffusion/"
sys.path.append(p)
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd

class LidarPainter(torch.nn.Module):
    def __init__(self, pretrained_path=None, lora_rank_unet=8, lora_rank_vae=4):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("/src/51sim-ai/img2img-turbo/ckpt", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()

        # 加载VAE并修改解码器
        vae = AutoencoderKL.from_pretrained("/src/51sim-ai/img2img-turbo/ckpt", subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=1).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=1).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=1).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=1).cuda()
        vae.decoder.ignore_skip = False

        # 加载UNet并调整输入通道支持参考图像拼接
        unet = UNet2DConditionModel.from_pretrained("/src/51sim-ai/img2img-turbo/ckpt", subfolder="unet")
        original_conv_in = unet.conv_in; original_conv_out = unet.conv_out
        #print(original_conv_in.in_channels, original_conv_in.out_channels, original_conv_in.kernel_size, original_conv_in.stride, original_conv_in.padding)
        new_conv_in = torch.nn.Conv2d(in_channels=8, out_channels=original_conv_in.out_channels, kernel_size=original_conv_in.kernel_size, stride=original_conv_in.stride, padding=original_conv_in.padding).cuda()# 原4（c_t） + 4（ref_img）
        new_conv_out = torch.nn.Conv2d(in_channels=original_conv_out.in_channels, out_channels=8, kernel_size=original_conv_out.kernel_size, stride=original_conv_out.stride, padding=original_conv_out.padding).cuda()# 原4（c_t） + 4（ref_img）
        # 初始化权重：前4通道复制原权重，后4通道随机初始化 or 复制原有？
        with torch.no_grad():
            new_conv_in.weight[:, :4] = original_conv_in.weight.clone()
            new_conv_in.weight[:, 4:] = original_conv_in.weight.clone()
            #torch.nn.init.xavier_uniform_(new_conv_in.weight[:, 4:])
            new_conv_out.weight[:4, :] = original_conv_out.weight.clone()
            new_conv_out.weight[4:, :] = original_conv_out.weight.clone()
        # 替换原conv_in/out层
        unet.conv_in = new_conv_in; unet.conv_out = new_conv_out

        if pretrained_path == None:
            print("Initializing model with random weights")
            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            target_modules_unet = [
                "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
            ]
            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
                target_modules=target_modules_unet
            )
            unet.add_adapter(unet_lora_config)
            self.lora_rank_unet = lora_rank_unet
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae
            self.target_modules_unet = target_modules_unet
        else:
            sd = torch.load(pretrained_path, map_location="cpu")
            self.target_modules_unet = sd["unet_lora_target_modules"]
            self.target_modules_vae = sd["vae_lora_target_modules"]
            self.lora_rank_unet = sd["rank_unet"]
            self.lora_rank_vae = sd["rank_vae"]
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        # unet.enable_xformers_memory_efficient_attention()
        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([200], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    def forward(self, c_t, ref_img, prompt=None, prompt_tokens=None, deterministic=True, r=1.0, noise_map=None):
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"
        assert ref_img is not None, "参考图像ref_img必须提供"

        if prompt is not None:
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]

        # 编码参考图像
        ref_latent = self.vae.encode(ref_img).latent_dist.sample() * self.vae.config.scaling_factor  # [B, 4, H/8, W/8]
        # 编码条件图像
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor  # [B, 4, H/8, W/8]
        # 拼接隐变量
        combined_input = torch.cat([encoded_control, ref_latent], dim=1)  # [B, 8, H/8, W/8]

        if deterministic:
            model_pred = self.unet(combined_input, self.timesteps, encoder_hidden_states=caption_enc).sample
            x_denoised = self.sched.step(model_pred, self.timesteps, combined_input, return_dict=True).prev_sample
            x_denoised = x_denoised.to(model_pred.dtype)
            denoised_tgt = x_denoised[:, :4, :, :]
            denoised_ref = x_denoised[:, 4:, :, :]
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            output_tgt = (self.vae.decode(denoised_tgt / self.vae.config.scaling_factor).sample).clamp(-1, 1)
            output_ref = (self.vae.decode(denoised_ref / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        else:
            self.unet.set_adapters(["default"], weights=[r])
            set_weights_and_activate_adapters(self.vae, ["vae_skip"], [r])
            unet_input = combined_input * r + noise_map * (1 - r)  # 使用拼接后的输入
            self.unet.conv_in.r = r
            unet_output = self.unet(unet_input, self.timesteps, encoder_hidden_states=caption_enc).sample
            self.unet.conv_in.r = None
            x_denoised = self.sched.step(unet_output, self.timesteps, unet_input, return_dict=True).prev_sample
            x_denoised = x_denoised.to(unet_output.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            self.vae.decoder.gamma = r
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        return output_tgt, output_ref

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        torch.save(sd, outf)

class LidarPintora(torch.nn.Module):
    def __init__(self, pretrained_path=None, lora_rank_unet=8, lora_rank_vae=4):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("/src/51sim-ai/img2img-turbo/ckpt", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()

        # 加载VAE并修改解码器
        vae = AutoencoderKL.from_pretrained("/src/51sim-ai/img2img-turbo/ckpt", subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=1).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=1).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=1).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=1).cuda()
        vae.decoder.ignore_skip = False

        # 加载UNet并调整输入通道支持参考图像拼接
        unet = UNet2DConditionModel.from_pretrained("/src/51sim-ai/img2img-turbo/ckpt", subfolder="unet")
        original_conv_in = unet.conv_in; #original_conv_out = unet.conv_out
        #print(original_conv_in.in_channels, original_conv_in.out_channels, original_conv_in.kernel_size, original_conv_in.stride, original_conv_in.padding)
        new_conv_in = torch.nn.Conv2d(in_channels=8, out_channels=original_conv_in.out_channels, kernel_size=original_conv_in.kernel_size, stride=original_conv_in.stride, padding=original_conv_in.padding).cuda()# 原4（c_t） + 4（ref_img）
        #new_conv_out = torch.nn.Conv2d(in_channels=original_conv_out.in_channels, out_channels=8, kernel_size=original_conv_out.kernel_size, stride=original_conv_out.stride, padding=original_conv_out.padding).cuda()# 原4（c_t） + 4（ref_img）
        # 初始化权重：前4通道复制原权重，后4通道随机初始化 or 复制原有？
        with torch.no_grad():
            new_conv_in.weight[:, :4] = original_conv_in.weight.clone()
            new_conv_in.weight[:, 4:] = original_conv_in.weight.clone()
            #torch.nn.init.xavier_uniform_(new_conv_in.weight[:, 4:])
            #new_conv_out.weight[:4, :] = original_conv_out.weight.clone()
            #new_conv_out.weight[4:, :] = original_conv_out.weight.clone()
        # 替换原conv_in/out层
        unet.conv_in = new_conv_in; #unet.conv_out = new_conv_out

        if pretrained_path == None:
            print("Initializing model with random weights")
            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            target_modules_unet = [
                "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
            ]
            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
                target_modules=target_modules_unet
            )
            unet.add_adapter(unet_lora_config)
            self.lora_rank_unet = lora_rank_unet
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae
            self.target_modules_unet = target_modules_unet
        else:
            sd = torch.load(pretrained_path, map_location="cpu")
            self.target_modules_unet = sd["unet_lora_target_modules"]
            self.target_modules_vae = sd["vae_lora_target_modules"]
            self.lora_rank_unet = sd["rank_unet"]
            self.lora_rank_vae = sd["rank_vae"]
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        # unet.enable_xformers_memory_efficient_attention()
        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([200], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    def forward(self, c_t, ref_img, prompt=None, prompt_tokens=None, deterministic=True, r=1.0, noise_map=None):
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"
        assert ref_img is not None, "参考图像ref_img必须提供"

        if prompt is not None:
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]

        # 编码参考图像
        ref_latent = self.vae.encode(ref_img).latent_dist.sample() * self.vae.config.scaling_factor  # [B, 4, H/8, W/8]
        # 编码条件图像
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor  # [B, 4, H/8, W/8]
        # 拼接隐变量
        combined_input = torch.cat([encoded_control, ref_latent], dim=1)  # [B, 8, H/8, W/8]

        if deterministic:
            model_pred = self.unet(combined_input, self.timesteps, encoder_hidden_states=caption_enc).sample
            x_denoised = self.sched.step(model_pred, self.timesteps, combined_input, return_dict=True).prev_sample #注意这里传入的是单个还是拼接的
            x_denoised = x_denoised.to(model_pred.dtype)
            #denoised_tgt = x_denoised[:, :4, :, :]
            #denoised_ref = x_denoised[:, 4:, :, :]
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            output_img = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
            #output_tgt = (self.vae.decode(denoised_tgt / self.vae.config.scaling_factor).sample).clamp(-1, 1)
            #output_ref = (self.vae.decode(denoised_ref / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        else:
            self.unet.set_adapters(["default"], weights=[r])
            set_weights_and_activate_adapters(self.vae, ["vae_skip"], [r])
            unet_input = combined_input * r + noise_map * (1 - r)  # 使用拼接后的输入
            self.unet.conv_in.r = r
            unet_output = self.unet(unet_input, self.timesteps, encoder_hidden_states=caption_enc).sample
            self.unet.conv_in.r = None
            x_denoised = self.sched.step(unet_output, self.timesteps, unet_input, return_dict=True).prev_sample
            x_denoised = x_denoised.to(unet_output.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            self.vae.decoder.gamma = r
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        #return output_tgt, output_ref
        return output_img

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        torch.save(sd, outf)

class LatentFusionModule(torch.nn.Module):
    def __init__(self, num_channels=4):
        super().__init__()
        # 通道级别的注意力机制
        self.attention = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels * 2, num_channels, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(num_channels, num_channels, kernel_size=1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, control_latent, ref_latent):
        # 通道注意力融合
        combined = torch.cat([control_latent, ref_latent], dim=1)
        attn_map = self.attention(combined)
        attention_fused = control_latent * attn_map + ref_latent * (1 - attn_map)
        return attention_fused

class LidarPintoraConvIn(torch.nn.Module):
    def __init__(self, pretrained_path=None, lora_rank_unet=8, lora_rank_vae=4):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("/src/51sim-ai/img2img-turbo/ckpt", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()

        # 通道注意力融合
        self.fusion_module = LatentFusionModule(num_channels=4)

        # 加载VAE并修改解码器
        vae = AutoencoderKL.from_pretrained("/src/51sim-ai/img2img-turbo/ckpt", subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=1).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=1).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=1).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=1).cuda()
        vae.decoder.ignore_skip = False

        # 加载UNet并调整输入通道支持参考图像拼接
        unet = UNet2DConditionModel.from_pretrained("/src/51sim-ai/img2img-turbo/ckpt", subfolder="unet")
        original_conv_in = unet.conv_in; #original_conv_out = unet.conv_out
        #print(original_conv_in.in_channels, original_conv_in.out_channels, original_conv_in.kernel_size, original_conv_in.stride, original_conv_in.padding)
        new_conv_in = torch.nn.Conv2d(in_channels=8, out_channels=original_conv_in.out_channels, kernel_size=original_conv_in.kernel_size, stride=original_conv_in.stride, padding=original_conv_in.padding).cuda()# 原4（c_t） + 4（ref_img）
        #new_conv_out = torch.nn.Conv2d(in_channels=original_conv_out.in_channels, out_channels=8, kernel_size=original_conv_out.kernel_size, stride=original_conv_out.stride, padding=original_conv_out.padding).cuda()# 原4（c_t） + 4（ref_img）
        # 初始化权重：前4通道复制原权重，后4通道随机初始化 or 复制原有？
        with torch.no_grad():
            new_conv_in.weight[:, :4] = original_conv_in.weight.clone()
            new_conv_in.weight[:, 4:] = original_conv_in.weight.clone()
            #torch.nn.init.xavier_uniform_(new_conv_in.weight[:, 4:])
            #new_conv_out.weight[:4, :] = original_conv_out.weight.clone()
            #new_conv_out.weight[4:, :] = original_conv_out.weight.clone()
        # 替换原conv_in/out层
        unet.conv_in = new_conv_in; #unet.conv_out = new_conv_out

        if pretrained_path == None:
            print("Initializing model with random weights")
            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            target_modules_unet = [
                "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out", "conv_in",
                "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
            ]
            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
                target_modules=target_modules_unet
            )
            unet.add_adapter(unet_lora_config)
            self.lora_rank_unet = lora_rank_unet
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae
            self.target_modules_unet = target_modules_unet
        else:
            sd = torch.load(pretrained_path, map_location="cpu")
            self.target_modules_unet = sd["unet_lora_target_modules"]
            self.target_modules_vae = sd["vae_lora_target_modules"]
            self.lora_rank_unet = sd["rank_unet"]
            self.lora_rank_vae = sd["rank_vae"]
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)
            # 融合部分权重加载
            fusion_state_dict = self.fusion_module.state_dict()
            fusion_state_dict.update(sd["state_dict_fusion"])
            self.fusion_module.load_state_dict(fusion_state_dict)

        # unet.enable_xformers_memory_efficient_attention()
        unet.to("cuda")
        vae.to("cuda")
        self.fusion_module.to("cuda")
        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([200], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.fusion_module.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.fusion_module.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.fusion_module.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)
        for param in self.fusion_module.parameters():
            param.requires_grad = True

    def forward(self, c_t, ref_img, prompt=None, prompt_tokens=None, deterministic=True, r=1.0, noise_map=None):
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"
        assert ref_img is not None, "参考图像ref_img必须提供"

        if prompt is not None:
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]

        # 编码参考图像
        ref_latent = self.vae.encode(ref_img).latent_dist.sample() * self.vae.config.scaling_factor  # [B, 4, H/8, W/8]
        # 编码条件图像
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor  # [B, 4, H/8, W/8]
        # 拼接隐变量
        combined_input = torch.cat([encoded_control, ref_latent], dim=1)  # [B, 8, H/8, W/8]
        # 融合噪声
        fused_latent = self.fusion_module(encoded_control, ref_latent)

        if deterministic:
            model_pred = self.unet(combined_input, self.timesteps, encoder_hidden_states=caption_enc).sample
            x_denoised = self.sched.step(model_pred, self.timesteps, fused_latent, return_dict=True).prev_sample
            x_denoised = x_denoised.to(model_pred.dtype)
            #denoised_tgt = x_denoised[:, :4, :, :]
            #denoised_ref = x_denoised[:, 4:, :, :]
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            output_img = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
            #output_tgt = (self.vae.decode(denoised_tgt / self.vae.config.scaling_factor).sample).clamp(-1, 1)
            #output_ref = (self.vae.decode(denoised_ref / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        else:
            self.unet.set_adapters(["default"], weights=[r])
            set_weights_and_activate_adapters(self.vae, ["vae_skip"], [r])
            unet_input = combined_input * r + noise_map * (1 - r)  # 使用拼接后的输入
            self.unet.conv_in.r = r
            unet_output = self.unet(unet_input, self.timesteps, encoder_hidden_states=caption_enc).sample
            self.unet.conv_in.r = None
            x_denoised = self.sched.step(unet_output, self.timesteps, unet_input, return_dict=True).prev_sample
            x_denoised = x_denoised.to(unet_output.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            self.vae.decoder.gamma = r
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        #return output_tgt, output_ref
        return output_img

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        sd["state_dict_fusion"] = self.fusion_module.state_dict()
        torch.save(sd, outf)


'''
model = LidarPainter()
T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS)
        ])
ori_img = Image.open('/src/51sim-ai/img2img-turbo/data/lidar_mini/train_A/0061_000000_0.png')
ref_img = Image.open('/src/51sim-ai/img2img-turbo/data/lidar_mini/train_B/0061_000000_0.png')
ori_img = T(ori_img); ori_img = F.to_tensor(ori_img)
ref_img = T(ref_img); ref_img = F.to_tensor(ref_img)
ori_batch = ori_img.unsqueeze(0).cuda()
ref_batch = ref_img.unsqueeze(0).cuda()
output_tgt, output_ref = model.forward(ori_batch, ref_batch, 'street')
print(output_tgt.shape, output_ref.shape, output_tgt.min(), output_tgt.max())
'''
'''
model = LidarPintora()
T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS)
        ])
ori_img = Image.open('/src/51sim-ai/img2img-turbo/data/lidar_mini/train_A/0061_000000_0.png')
ref_img = Image.open('/src/51sim-ai/img2img-turbo/data/lidar_mini/train_B/0061_000000_0.png')
ori_img = T(ori_img); ori_img = F.to_tensor(ori_img)
ref_img = T(ref_img); ref_img = F.to_tensor(ref_img)
ori_batch = ori_img.unsqueeze(0).cuda()
ref_batch = ref_img.unsqueeze(0).cuda()
output_img = model.forward(ori_batch, ref_batch, 'street')
print(output_img.shape, output_img.min(), output_img.max())
'''