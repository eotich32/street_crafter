from pipeline_difix import DifixPipeline
from diffusers.utils import load_image
from PIL import Image

pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
pipe.to("cuda")

input_image = load_image("/src/51sim-ai/street_crafter/output/waymo/waymo_049_5cam_full/novel_view/test/-2.0_train/000005_3_shift_-2.00.png")#.crop((0, 166, 1600, 1066)) # 伪影图
input_image = input_image.resize((1024, 576), Image.LANCZOS)
ref_image = load_image('/src/51sim-ai/street_crafter/waymo_data/049/images/000005_3.png')#.crop((0, 166, 1600, 1066)) # 原视角参考图
ref_image = ref_image.resize((1024, 576), Image.LANCZOS)
prompt = "remove degradation"

output_image = pipe(prompt, image=input_image, ref_image=ref_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
output_image.save("example_output.png")