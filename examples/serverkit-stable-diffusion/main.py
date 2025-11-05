import numpy as np
import torch
from diffusers import StableDiffusionPipeline
import imaging_server_kit as sk


@sk.algorithm(
    parameters={
        "prompt": sk.String(
            name="Prompt",
            description="Text prompt",
            default="An astronaut riding a horse on the moon.",
        )
    },
)
def stable_diffusion_algo(prompt: str):
    model_id = "sd-legacy/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    generated_image = pipe(prompt).images[0]
    generated_image = np.asarray(generated_image)
    return sk.Image(generated_image)


if __name__ == "__main__":
    sk.serve(stable_diffusion_algo)
