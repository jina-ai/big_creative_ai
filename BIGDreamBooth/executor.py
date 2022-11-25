import io
import os
import subprocess
import tempfile
from collections import defaultdict
from typing import List, Dict

from accelerate.utils import write_basic_config
import PIL
import torch
from diffusers import StableDiffusionPipeline
from docarray import Document
from jina import DocumentArray
from huggingface_hub import login as hf_login

from auth import get_auth_executor_class, secure_request, SecurityLevel, _get_user_info

Executor = get_auth_executor_class()


PRE_TRAINDED_MODEL_DIR = 'stable-diffusion-v1-4'
METAMODEL_ID = 'meta'
METAMODEL_DIR = 'metamodel'

RARE_IDENTIFIERS = [
    'sks',
    'lnl',
    'brc',
    'mkd',
    'rvt',
    'pyr',
    'sph',
    'cyl',
    'cnc',
    'dkd',
    'dmd',
    'dld',
    'dvu',
]


class BIGDreamBooth(Executor):
    """BigDreamBooth trains Stable Diffusion"""

    def __init__(
            self,
            hf_token: str,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        hf_login(token=hf_token)

        # determine if gpu is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.models_dir = (
            os.path.join(self.workspace, 'models')
            if self.workspace
            else None
        )
        if self.models_dir:
            os.makedirs(self.models_dir, exist_ok=True)
        download_pretrained_stable_diffusion_model(self.models_dir)

        self.used_identifiers: Dict[str, List[str]] = defaultdict(List[str])

        write_basic_config(mixed_precision='no')

    def _get_model_dir(self, user_id: str, identifier: str = None) -> str:
        """Returns the path to the model directory of the user with the given user_id and identifier.

        :param user_id: The user_id of the user; is METAMODEL_ID for the metamodel
        :param identifier: The identifier used to train the model given object
        :return: The path to the model directory
        """
        if user_id == METAMODEL_ID:
            return os.path.join(self.models_dir, METAMODEL_DIR)
        return os.path.join(self.models_dir, user_id, identifier)

    @staticmethod
    def _get_user_id(parameters: Dict = None):
        """Returns the user_id of the model which shall be finetuned.
        Using 'own' in the parameters for 'target_model' will return the user_id of the user who sent
        the request. Using METAMODEL_ID will return the user_id of the metamodel.
        """
        target_model = parameters.get('target_model', 'own')
        if target_model == 'own':
            user_id = _get_user_info(parameters['jwt']['token'])['_id']
        elif target_model == METAMODEL_ID:
            user_id = METAMODEL_ID
        else:
            raise ValueError(f'Unknown target model {target_model}; must be either "own" or "{METAMODEL_ID}"')
        return user_id

    @secure_request(level=SecurityLevel.USER, on='/finetune')
    def finetune(self, docs: DocumentArray, parameters: Dict = None, **kwargs):
        """Finetunes stable diffusion model with DreamBooth for given object and returns the used identifier for that
        object.

        :param docs: The images of the object to finetune the model with. Only 3-5 images are accepted.
        :param parameters: The parameters for the finetuning; must contain the key 'target_model' which can be either
            'own' or METAMODEL_ID, where 'own' will finetune the model of the user who sent the request and METAMODEL_ID
            will finetune the metamodel; must contain the key 'class_name' which is the class of the object
        :return: The identifier used for the finetuning, which can be used to generate images of the object
        """
        if len(docs) not in [3, 4, 5]:
            raise ValueError(f'Expected 3, 4 or 5 documents but got {len(docs)}')
        if 'class_name' not in parameters:
            raise ValueError('No class for the images provided')
        class_name = parameters['class_name']

        user_id = self._get_user_id(parameters)

        identifier = self._get_next_identifier(self.used_identifiers[user_id])
        instance_prompt = f"a {identifier} {class_name}"

        # save finetuned model into user_id/identifier folder if user_id is not metamodel, else save into METAMODEL_DIR
        output_dir = self._get_model_dir(user_id, identifier)
        pretrained_model_dir = output_dir if user_id == METAMODEL_ID \
            else os.path.join(self.models_dir, PRE_TRAINDED_MODEL_DIR)

        # create temporary folder for instance data
        with tempfile.TemporaryDirectory() as instance_data_dir:
            # save documents as pngs
            for doc in docs:
                if not doc.mime_type.startswith('image') and not doc.modality == 'image':
                    raise ValueError(f'Only images are allowed but doc {doc.id} has mime_type {doc.mime_type} '
                                     f'and modality {doc.modality}')

                if doc.uri:
                    doc.load_uri_to_image_tensor(timeout=10)
                elif doc.blob:
                    doc.convert_blob_to_image_tensor()
                doc.save_image_tensor_to_file(file=os.path.join(instance_data_dir, f'{doc.id}.png'), image_format='png')

            # execute dreambooth.py
            _, err = cmd(
                f'''accelerate dreambooth.py 
                    --pretrained_model_name_or_path="{pretrained_model_dir}" 
                    --output_dir="{output_dir}" 
                    --instance_data_dir="{instance_data_dir}" 
                    --instance_prompt="{instance_prompt}" 
                    --resolution=512 
                    --train_batch_size=1 
                    --learning_rate=5e-6 --lr_scheduler="constant" --lr_warmup_steps=0 
                    --max_train_steps=200
                    --gradient_accumulation_steps=1
                '''
            )
            if err:
                error_message = err.decode('utf-8').split('ERROR')[-1]
                raise RuntimeError(f'DreamBooth failed\n{error_message}')

        self.used_identifiers[user_id].append(identifier)
        return identifier

    @secure_request(level=SecurityLevel.USER, on='/generate')
    def generate(self, docs: DocumentArray, parameters: Dict = None, **kwargs):
        """Generates images of the object with the given identifier.

        :param docs: Only one document is expected which contains the prompt for the generation
        :param parameters: The parameters for the generation; must contain the key 'target_model' which can be either
            'own' or METAMODEL_ID, where 'own' will generate images of the model of the user who sent the request and
            METAMODEL_ID will generate images of the metamodel; if using own model, must contain the key 'identifier'
            which is the identifier used to fit original images of object
        :return: The generated image
        """
        if len(docs) != 1:
            raise ValueError(f'Expected 1 document for prompt but got {len(docs)}')
        prompt = docs[0].text.strip()
        if not prompt:
            raise ValueError('No prompt provided')

        user_id = self._get_user_id(parameters)
        if user_id != METAMODEL_ID:
            identifier = parameters.get('identifier', '')
            if not identifier or identifier not in self.used_identifiers[user_id]:
                raise ValueError(f'No identifier provided in parameters or identifier not used for finetuning\n'
                                 f'(identifier: "{identifier}", used identifiers: {self.used_identifiers[user_id]})')
        else:
            identifier = None

        model_path = self._get_model_dir(user_id, identifier)

        pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(self.device)
        pipe.safety_checker = None

        image: PIL.Image.Image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        # save pil image to blob
        with io.BytesIO() as buffer:
            image.save(buffer, format='png')
            doc = Document(blob=buffer.getvalue())
        return doc

    @staticmethod
    def _get_next_identifier(used_identifiers: List[str]) -> str:
        """Returns the next identifier that is not yet used."""
        for identifier in RARE_IDENTIFIERS:
            if identifier not in used_identifiers:
                return identifier
        raise RuntimeError('No identifier left for this user')


def download_pretrained_stable_diffusion_model(model_dir: str, sd_version: str = 'stable-diffusion-v1-4'):
    """Downloads pretrained stable diffusion model."""
    pipe = StableDiffusionPipeline.from_pretrained(f"CompVis/{sd_version}", use_auth_token=True)
    for dir in [PRE_TRAINDED_MODEL_DIR, METAMODEL_DIR]:
        pipe.save_pretrained(os.path.join(model_dir, dir))


def cmd(command, std_output=False, wait=True):
    if isinstance(command, str):
        command = command.split()
    if not std_output:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    else:
        process = subprocess.Popen(command)
    if wait:
        output, error = process.communicate()
        return output, error
