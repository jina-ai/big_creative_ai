import gc
import os

import torch
from diffusers import StableDiffusionPipeline
from torch.utils.data import Dataset
from tqdm import tqdm


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def _generate(save_dir: str, num_images: int, model_path: str, prompt: str, batch_size: int, revision=None):
    pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, revision=revision)
    pipeline.set_progress_bar_config(disable=True)

    sample_dataset = PromptDataset(prompt, num_images)
    sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=batch_size)

    pipeline.to('cuda' if torch.cuda.is_available() else 'cpu')

    cnt = 0
    for example in tqdm(sample_dataloader, desc="Generating images"):
        images = pipeline(example["prompt"]).images
        for image in images:
            image.save(os.path.join(save_dir, f"{cnt}.jpg"), format="JPEG", quality=95)
            cnt += 1

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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