import inspect
import io
import os

import torch
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

# add to the sys.path
import sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)


from BIGDreamBooth.executor.dreambooth import PromptDataset


def _generate(save_dir: str, num_images: int, model_path: str, prompt: str, batch_size: int, revision=None):
    accelerator = Accelerator()

    pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, revision=revision)
    pipeline.set_progress_bar_config(disable=True)

    sample_dataset = PromptDataset(prompt, num_images)
    sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=batch_size)

    sample_dataloader = accelerator.prepare(sample_dataloader)
    pipeline.to(accelerator.device)

    cnt = 0
    for example in tqdm(sample_dataloader, desc="Generating images", disable=not accelerator.is_local_main_process):
        images = pipeline(example["prompt"]).images
        for image in images:
            with io.BytesIO() as buffer:
                # save image as jpeg to save space with quality 95
                image.save(buffer, format="JPEG", quality=95)
                # save image to file
                with open(f"{save_dir}/{cnt}.jpg", "wb") as f:
                    f.write(buffer.getvalue())
            cnt += 1


if __name__ == '__main__':
    # parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--num_images', type=int, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--revision', type=str, default=None)

    args = parser.parse_args()

    # generate images
    _generate(args.save_dir, args.num_images, args.model_path, args.prompt, args.batch_size, args.revision)