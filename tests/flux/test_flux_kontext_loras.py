import pytest
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.utils import get_precision, is_turing

from .utils import hash_str_to_int


@pytest.mark.skipif(is_turing(), reason="Skip tests due to using Turing GPUs")
def test_flux_kontext_loras():
    precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        f"mit-han-lab/nunchaku-flux.1-kontext-dev/svdq-{precision}_r32-flux.1-kontext-dev.safetensors"
    )
    pipeline = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev", transformer=transformer, torch_dtype=torch.bfloat16
    ).to("cuda")

    transformer.update_lora_params(
        "nunchaku-tech/nunchaku-test-models/relight-kontext-lora-single-caption_comfy.safetensors"
    )
    transformer.set_lora_strength(1)

    image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
    ).convert("RGB")
    prompts = [
        "Make Pikachu hold a sign that says 'Nunchaku is awesome', yarn art style, detailed, vibrant colors",
        "Convert the image to ghibli style",
        "help me convert it to manga style",
        "Convert it to a realistic photo",
    ]

    for prompt in prompts:
        seed = hash_str_to_int(prompt)
        reuslt = pipeline(image=image, prompt=prompt, generator=torch.Generator().manual_seed(seed)).images[0]
        reuslt.save(f"flux.1-kontext-dev-{precision}-{seed}.png")
