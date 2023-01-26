import torch
from diffusers import StableDiffusionPipeline


def download_pretrained_stable_diffusion_model(model_dir: str, sd_model: str, revision: str = None):
    """Downloads pretrained stable diffusion model."""
    pipe = StableDiffusionPipeline.from_pretrained(
        sd_model, use_auth_token=True, revision=revision, torch_dtype=torch.float16
    )
    pipe.save_pretrained(model_dir)


if __name__ == '__main__':
    # parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--sd_model', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--revision', type=str, default=None)

    args = parser.parse_args()

    # download pretrained stable diffusion model
    download_pretrained_stable_diffusion_model(args.model_dir, args.sd_model, args.revision)
