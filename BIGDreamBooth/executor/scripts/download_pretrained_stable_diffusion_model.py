import os

import torch
from diffusers import StableDiffusionPipeline


# if I run this script, it won't find BIGDreamBoothExecutor, what can I do?
# I tried to add the path to the sys.path, but it didn't work
# I also tried to add the path to the PYTHONPATH, but it didn't work

# add to the sys.path
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_dir, ''))


from BIGDreamBooth.executor import BIGDreamBoothExecutor


def download_pretrained_stable_diffusion_model(model_dir: str, sd_model: str, revision: str = None):
    """Downloads pretrained stable diffusion model."""
    if not all(os.path.exists(os.path.join(model_dir, _dir)) for _dir in [
        BIGDreamBoothExecutor.PRE_TRAINDED_MODEL_DIR, BIGDreamBoothExecutor.METAMODEL_DIR,
    ]):
        pipe = StableDiffusionPipeline.from_pretrained(
            sd_model, use_auth_token=True, revision=revision, torch_dtype=torch.float16
        )
        for _dir in [
            BIGDreamBoothExecutor.PRE_TRAINDED_MODEL_DIR, BIGDreamBoothExecutor.METAMODEL_DIR,
        ]:
            pipe.save_pretrained(os.path.join(model_dir, _dir))


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
