import gradio as gr
import numpy as np
from PIL import Image
import cv2
import torch
torch.cuda.empty_cache()
from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d

#Link: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
model_id_15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pip_1_5 = StableDiffusionPipeline.from_pretrained(model_id_15, torch_dtype=torch.float16)
pip_1_5 = pip_1_5.to("cuda")

#Link: https://huggingface.co/stabilityai/stable-diffusion-2-1
model_id_21 = "stabilityai/stable-diffusion-2-1"
pip_2_1 = StableDiffusionPipeline.from_pretrained(model_id_21, torch_dtype=torch.float16)
pip_2_1 = pip_2_1.to("cuda")

''' Remove to use SDXL
#Link: https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")
'''

prompt_prev = None
sd_options_prev = None
seed_prev = None 
sd_image_prev = None

def generate_feature_map(image):
    """
    Generate a feature map from the input image using the specified method.

    Args:
        image (PIL.Image or np.ndarray): The input image from which to generate the feature map.
        method (str): The method to use for generating the feature map ('mean', 'max', 'heatmap').

    Returns:
        PIL.Image: The generated feature map as a PIL image.
    """
    # Convert image to numpy array for processing
    image_np = np.array(image)

    feature_map = np.mean(image_np, axis=2)
    feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map))
    feature_map = (feature_map * 255).astype(np.uint8)
    feature_map = cv2.applyColorMap(feature_map, cv2.COLORMAP_JET)

    # Normalize the feature map for visualization
    feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map))
    feature_map = (feature_map * 255).astype(np.uint8)  # Scale to 0-255

    noise_level= 0.8
    noise = np.random.normal(0, noise_level * 255, feature_map.shape)
    feature_map = feature_map.astype(np.float32) + noise 
    feature_map = np.clip(feature_map, 0, 255).astype(np.uint8)

    return Image.fromarray(feature_map) 

def infer(prompt, sd_options, seed, b1, b2, s1, s2):
    global prompt_prev
    global sd_options_prev
    global seed_prev
    global sd_image_prev

    # Select the pipeline based on sd_options
    if sd_options == 'SD1.5':
         pip = pip_1_5
    else:
         pip = pip_2_1

    run_baseline = False
    if prompt != prompt_prev or sd_options != sd_options_prev or seed != seed_prev:
        run_baseline = True
        prompt_prev = prompt
        sd_options_prev = sd_options
        seed_prev = seed

    if run_baseline:
        register_free_upblock2d(pip, b1=1.0, b2=1.0, s1=1.0, s2=1.0)
        register_free_crossattn_upblock2d(pip, b1=1.0, b2=1.0, s1=1.0, s2=1.0)
       
        torch.manual_seed(seed)
        print("Generating SD:")
        sd_image = pip(prompt, num_inference_steps=25).images[0]
        sd_image_prev = sd_image
        
        # Refine the image if using SDXL
        if sd_options == 'SDXL':
            sd_image = refiner(prompt, image=sd_image, num_inference_steps=25).images[0]
        sd_image_prev = sd_image
        
    else:
        sd_image = sd_image_prev
    
    register_free_upblock2d(pip, b1=b1, b2=b2, s1=s1, s2=s1)
    register_free_crossattn_upblock2d(pip, b1=b1, b2=b2, s1=s1, s2=s1)

    torch.manual_seed(seed)
    print("Generating FreeU:")
    freeu_image = pip(prompt, num_inference_steps=25).images[0]
    freeu_feature_map = freeu_image
    freeu_feature_map = generate_feature_map(freeu_image)
    feature_map = generate_feature_map(sd_image)

    # First SD, then Feature map (SD without FreeU), then Freeu, then Feature map (SD with FreeU)
    images = [sd_image, feature_map, freeu_image, freeu_feature_map]

    return images

examples = [
    [
        "A drone view of celebration with Christma tree and fireworks, starry sky - background.",
    ],
    [
        "happy dog wearing a yellow turtleneck, studio, portrait, facing camera, studio, dark bg"
    ],
    [
        "Campfire at night in a snowy forest with starry sky in the background."
    ],
    [
        "a fantasy landscape, trending on artstation"
    ],
    [
        "Busy freeway at night."
    ],
    [
        "An astronaut is riding a horse in the space in a photorealistic style."
    ],
    [
        "Turtle swimming in ocean."
    ],
    [
        "A storm trooper vacuuming the beach."
    ],
    [
        "An astronaut feeding ducks on a sunny afternoon, reflection from the water."
    ],
    [
        "Fireworks."
    ],
    [
        "A fat rabbit wearing a purple robe walking through a fantasy landscape."
    ],
    [
        "A koala bear playing piano in the forest."
    ],
    [
        "An astronaut flying in space, 4k, high resolution."
    ],
    [
        "Flying through fantasy landscapes, 4k, high resolution."
    ],
    [
        "A small cabin on top of a snowy mountain in the style of Disney, artstation",
    ],
    [
        "half human half cat, a human cat hybrid",
    ],
    [
        "a drone flying over a snowy forest."
    ],
]
    
css = """
h1 {
  text-align: center;
}

#component-0 {
  max-width: 730px;
  margin: auto;
}
"""

block = gr.Blocks(css='style.css')
            
with block:
    gr.Markdown("# SD vs. FreeU")
    with gr.Group():
        with gr.Row(elem_id="prompt-container").style(mobile_collapse=False, equal_height=True):
            with gr.Column():
                text = gr.Textbox(
                    label="Enter your prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                    )
            btn = gr.Button("Generate image", scale=0)
        with gr.Row():             
            sd_options = gr.Dropdown(["SD1.5","SD2.1","SDXL"], label="SD options", value="SD2.1", visible=True)
    
    with gr.Group():
        with gr.Row():
            with gr.Accordion('FreeU Parameters (feel free to adjust these parameters based on your prompt): ', open=True):
                with gr.Row():
                    b1 = gr.Slider(label='b1: backbone factor of the first stage block of decoder',
                                            minimum=1,
                                            maximum=3.0,
                                            step=0.01,
                                            value=1.1)
                    b2 = gr.Slider(label='b2: backbone factor of the second stage block of decoder',
                                            minimum=1,
                                            maximum=3.0,
                                            step=0.01,
                                            value=1.2)
                with gr.Row():
                    s1 = gr.Slider(label='s1: skip factor of the first stage block of decoder',
                                            minimum=0,
                                            maximum=3.0,
                                            step=0.1,
                                            value=0.2)
                    s2 = gr.Slider(label='s2: skip factor of the second stage block of decoder',
                                            minimum=0,
                                            maximum=3.0,
                                            step=0.1,
                                            value=0.2)    
                
                seed = gr.Slider(label='seed',
                             minimum=0,
                             maximum=500,
                             step=1,
                             value=42)
                    
    with gr.Row():
        with gr.Group():
            # btn = gr.Button("Generate image", scale=0)
            with gr.Row():
                with gr.Column() as c1:
                    image_1 = gr.Image(interactive=False)
                    image_1_label = gr.Markdown("SD")

        with gr.Group():
            # btn = gr.Button("Generate image", scale=0)
            with gr.Row():
                with gr.Column() as c2:
                    image_2 = gr.Image(interactive=False)
                    image_2_label = gr.Markdown("Feature Map (SD without FreeU)")

        with gr.Group():
            with gr.Row():
                with gr.Column() as c3:
                    image_3 = gr.Image(interactive=False)
                    image_3_label = gr.Markdown("FreeU")

        with gr.Group():
            with gr.Row():
                with gr.Column() as c4:
                    image_4 = gr.Image(interactive=False)
                    image_4_label = gr.Markdown("Feature Map (SD with FreeU)")
            
    ex = gr.Examples(examples=examples, fn=infer, inputs=[text, sd_options, seed, b1, b2, s1, s2], outputs=[image_1, image_2, image_3, image_4], cache_examples=False)
    ex.dataset.headers = [""]

    text.submit(infer, inputs=[text, sd_options, seed, b1, b2, s1, s2], outputs=[image_1, image_2, image_3, image_4])
    btn.click(infer, inputs=[text, sd_options, seed, b1, b2, s1, s2], outputs=[image_1, image_2, image_3, image_4])

block.launch(share=True, debug=True)
# block.queue(default_enabled=False).launch(share=True)