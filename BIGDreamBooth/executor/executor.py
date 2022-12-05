import io
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from typing import List, Dict

from accelerate.utils import write_basic_config
from diffusers import StableDiffusionPipeline
from docarray import Document
from huggingface_hub import login as hf_login
from jina import DocumentArray
import PIL
import torch

from .auth import NOWAuthExecutor as Executor, secure_request, SecurityLevel, _get_user_info


class BIGDreamBoothExecutor(Executor):
    """BIGDreamBoothExecutor trains Stable Diffusion"""

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
        'dwd',
        'dxd',
        'dyd',
        'dzd',
        'scs',
        'qtq',
        'qrq',
        'xjy',
        'xky',
        'xly',
        'xmy',
        'xny',
        'wkw',
        'vkv',
        'mpm',
        'mnm',
    ]

    DEFAULT_LEARNING_RATE = 5e-6
    DEFAULT_MAX_TRAIN_STEPS = 200
    MAX_MAX_TRAIN_STEPS = 1500

    def __init__(
            self,
            hf_token: str,
            device: str = 'cuda',
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        hf_login(token=hf_token)

        # determine if gpu is available
        self.device = device

        self.models_dir = (
            os.path.join(self.workspace, 'models')
            if self.workspace
            else None
        )
        os.makedirs(self.models_dir, exist_ok=True)
        download_pretrained_stable_diffusion_model(self.models_dir)

        self.user_to_identifiers_and_categories: Dict[str, Dict[str, str]] = defaultdict(lambda: defaultdict(str))
        self.user_to_identifiers_and_categories_path = os.path.join(
            self.models_dir, 'user_to_identifiers_and_categories.json'
        )
        if os.path.exists(self.user_to_identifiers_and_categories_path):
            with open(self.user_to_identifiers_and_categories_path, 'r') as fp:
                tmp_dict = json.load(fp)
                # update the dict
                for user_id, identifiers_and_categories in tmp_dict.items():
                    self.user_to_identifiers_and_categories[user_id].update(identifiers_and_categories)

        write_basic_config(mixed_precision='no')

    def _get_model_dir(self, user_id: str, identifier: str = None) -> str:
        """Returns the path to the model directory of the user with the given user_id and identifier.

        :param user_id: The user_id of the user; is METAMODEL_ID for the metamodel
        :param identifier: The identifier used to train the model given object
        :return: The path to the model directory
        """
        if user_id == self.METAMODEL_ID:
            return os.path.join(self.models_dir, self.METAMODEL_DIR)
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
        elif target_model == BIGDreamBoothExecutor.METAMODEL_ID:
            user_id = BIGDreamBoothExecutor.METAMODEL_ID
        else:
            raise ValueError(f'Unknown target model {target_model}; must be either "own" or '
                             f'"{BIGDreamBoothExecutor.METAMODEL_ID}"')
        return user_id

    @secure_request(SecurityLevel.ADMIN, on='/update_rare_identifiers')
    def update_rare_identifiers(self, parameters: Dict = None, **kwargs):
        """Updates the list of rare identifiers."""
        self.logger.info(f'Updating rare identifiers to {parameters["rare_identifiers"]}')
        self.RARE_IDENTIFIERS = parameters['rare_identifiers']

    @secure_request(SecurityLevel.USER, on='/list_identifiers_n_categories')
    def list_identifiers_n_categories(self, parameters, **kwargs):
        """Returns the identifiers & their categories of the models which were trained for the user and the metamodel."""
        user_id = _get_user_info(parameters['jwt']['token'])['_id']
        return DocumentArray(
            Document(
                tags={
                    'own': self.user_to_identifiers_and_categories.get(user_id, {}),
                    self.METAMODEL_ID: self.user_to_identifiers_and_categories.get(self.METAMODEL_ID)
                }
            )
        )

    @secure_request(SecurityLevel.USER, on='/experimental/finetune')
    def experimental_finetune(self, docs: DocumentArray, parameters: Dict = None, **kwargs):
        if 'category' not in parameters:
            raise ValueError('No category for the images provided in parameters')

        category = parameters['category']
        user_id = self._get_user_id(parameters)
        identifier = self._get_next_identifier(list(self.user_to_identifiers_and_categories[user_id].keys()))
        self.logger.info(f'Finetuning model for {user_id} model with identifier {identifier} and category {category}')

        learning_rate = parameters.get('learning_rate', self.DEFAULT_LEARNING_RATE)
        max_train_steps = int(parameters.get('max_train_steps', self.DEFAULT_MAX_TRAIN_STEPS))
        num_category_images = int(parameters.get('num_category_images', 200))
        if max_train_steps < 0 or max_train_steps > self.MAX_MAX_TRAIN_STEPS:
            raise ValueError(
                f'Expected max_train_steps to be in [0, {self.MAX_MAX_TRAIN_STEPS}] but got {max_train_steps}')
        self.logger.info(f'Using learning rate {learning_rate}, max training steps {max_train_steps} and '
                         f'{num_category_images} images for category {category}')

        instance_prompt = f"a {identifier} {category}"

        # save finetuned model into user_id/identifier folder if user_id is not metamodel, else save into METAMODEL_DIR
        output_dir = f"{str(self._get_model_dir(user_id, identifier))}-experimental"
        pretrained_model_dir = output_dir if user_id == self.METAMODEL_ID \
            else os.path.join(self.models_dir, self.PRE_TRAINDED_MODEL_DIR)

        # create temporary folder for instance data
        with tempfile.TemporaryDirectory() as tmp_dir:
            # create sub folder for instance data
            instance_data_dir = os.path.join(tmp_dir, 'instance_data')
            os.makedirs(instance_data_dir, exist_ok=True)
            # create sub folder for class data
            class_data_dir = os.path.join(tmp_dir, 'class_data')
            os.makedirs(class_data_dir, exist_ok=True)
            # save documents as pngs
            for doc in docs:
                if not doc.mime_type.startswith('image') and not doc.modality == 'image':
                    raise ValueError(f'Only images are allowed but doc {doc.id} has mime_type {doc.mime_type} '
                                     f'and modality {doc.modality}')

                if doc.blob:
                    doc.convert_blob_to_image_tensor()
                elif doc.uri:
                    doc.load_uri_to_image_tensor(timeout=10)
                doc.save_image_tensor_to_file(file=os.path.join(instance_data_dir, f'{doc.id}.png'), image_format='png')

            torch.cuda.empty_cache()
            # execute dreambooth.py
            cur_dir = os.path.abspath(os.path.join(__file__, '..'))
            # note this the output and error are switched for accelerate launch dreambooth.py
            output, err = cmd(
                [
                    'accelerate', 'launch', f"{cur_dir}/dreambooth.py",
                    "--pretrained_model_name_or_path", f"{pretrained_model_dir}",
                    "--output_dir", f"{output_dir}",
                    "--instance_data_dir", f"{instance_data_dir}", "--instance_prompt", f"{instance_prompt}",
                    "--class_data_dir", f"{class_data_dir}", "--class_prompt", f"a {category}",
                    '--with_prior_preservation',
                    "--resolution", "512",
                    "--learning_rate", f"{learning_rate}", "--lr_scheduler", "constant", "--lr_warmup_steps", "0",
                    "--max_train_steps", f"{max_train_steps}", "--train_batch_size", "1",
                    "--gradient_accumulation_steps", "2", "--gradient_checkpointing", "--use_8bit_adam",
                ]
            )
            for cmd_ret, cmd_ret_str in zip([output, err], ['output', 'error']):
                if cmd_ret:
                    error_message = err.decode('utf-8')  # .split('ERROR')[-1]
                    # error_message = err.decode('utf-8').split('ERROR')[-1]
                    error_message_print = f"----------\n{cmd_ret_str} message from dreambooth.py [Might not actually fail:"
                    for line in error_message.splitlines():
                        error_message_print += '\n' + line
                    error_message_print += '\n----------'
                    print(error_message_print, file=sys.stderr)

            # if output:
            #     output_message = output.decode('utf-8')
            # raise RuntimeError(f'DreamBooth failed: {error_message}')

        self.user_to_identifiers_and_categories[user_id][identifier] = category
        with open(self.user_to_identifiers_and_categories_path, 'w') as f:
            json.dump(self.user_to_identifiers_and_categories, f)

        return DocumentArray(Document(text=identifier))

    @secure_request(level=SecurityLevel.USER, on='/finetune')
    def finetune(self, docs: DocumentArray, parameters: Dict = None, **kwargs):
        """Finetunes stable diffusion model with DreamBooth for given object and returns the used identifier for that
        object.

        :param docs: The images of the object to finetune the model with.
        :param parameters: The parameters for the finetuning; must contain the key 'target_model' which can be either
            'own' or METAMODEL_ID, where 'own' will finetune the model of the user who sent the request and METAMODEL_ID
            will finetune the metamodel; must contain the key 'category' which is the category of the object/style
        :return: The identifier used for the finetuning, which can be used to generate images of the object
        """
        if 'category' not in parameters:
            raise ValueError('No category for the images provided in parameters')

        category = parameters['category']
        user_id = self._get_user_id(parameters)
        identifier = self._get_next_identifier(list(self.user_to_identifiers_and_categories[user_id].keys()))
        self.logger.info(f'Finetuning model for {user_id} model with identifier {identifier} and category {category}')

        learning_rate = parameters.get('learning_rate', self.DEFAULT_LEARNING_RATE)
        max_train_steps = int(parameters.get('max_train_steps', self.DEFAULT_MAX_TRAIN_STEPS))
        if max_train_steps < 0 or max_train_steps > self.MAX_MAX_TRAIN_STEPS:
            raise ValueError(f'Expected max_train_steps to be in [0, {self.MAX_MAX_TRAIN_STEPS}] but got {max_train_steps}')
        self.logger.info(f'Using learning rate {learning_rate} and max training steps {max_train_steps}')

        instance_prompt = f"a {identifier} {category}"

        # save finetuned model into user_id/identifier folder if user_id is not metamodel, else save into METAMODEL_DIR
        output_dir = self._get_model_dir(user_id, identifier)
        pretrained_model_dir = output_dir if user_id == self.METAMODEL_ID \
            else os.path.join(self.models_dir, self.PRE_TRAINDED_MODEL_DIR)

        # create temporary folder for instance data
        with tempfile.TemporaryDirectory() as instance_data_dir:
            # save documents as pngs
            for doc in docs:
                if not doc.mime_type.startswith('image') and not doc.modality == 'image':
                    raise ValueError(f'Only images are allowed but doc {doc.id} has mime_type {doc.mime_type} '
                                     f'and modality {doc.modality}')

                if doc.blob:
                    doc.convert_blob_to_image_tensor()
                elif doc.uri:
                    doc.load_uri_to_image_tensor(timeout=10)
                doc.save_image_tensor_to_file(file=os.path.join(instance_data_dir, f'{doc.id}.png'), image_format='png')

            torch.cuda.empty_cache()
            # execute dreambooth.py
            cur_dir = os.path.abspath(os.path.join(__file__, '..'))
            # note this the output and error are switched for accelerate launch dreambooth.py
            output, err = cmd(
                [
                    'accelerate', 'launch', f"{cur_dir}/dreambooth.py",
                    "--pretrained_model_name_or_path", f"{pretrained_model_dir}",
                    "--output_dir", f"{output_dir}",
                    "--instance_data_dir", f"{instance_data_dir}",
                    "--instance_prompt", f"{instance_prompt}",
                    "--resolution", "512",
                    "--learning_rate", f"{learning_rate}", "--lr_scheduler", "constant", "--lr_warmup_steps", "0",
                    "--max_train_steps", f"{max_train_steps}", "--train_batch_size", "1",
                    "--gradient_accumulation_steps", "1"
                 ]
            )
            for cmd_ret, cmd_ret_str in zip([output, err], ['output', 'error']):
                if cmd_ret:
                    error_message = err.decode('utf-8')#.split('ERROR')[-1]
                    # error_message = err.decode('utf-8').split('ERROR')[-1]
                    error_message_print = f"----------\n{cmd_ret_str} message from dreambooth.py [Might not actually fail:"
                    for line in error_message.splitlines():
                        error_message_print += '\n' + line
                    error_message_print += '\n----------'
                    print(error_message_print, file=sys.stderr)

            # if output:
            #     output_message = output.decode('utf-8')
                # raise RuntimeError(f'DreamBooth failed: {error_message}')

        self.user_to_identifiers_and_categories[user_id][identifier] = category
        with open(self.user_to_identifiers_and_categories_path, 'w') as f:
            json.dump(self.user_to_identifiers_and_categories, f)

        return DocumentArray(Document(text=identifier))

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
        if user_id != self.METAMODEL_ID:
            identifier = parameters.get('identifier', '')
            if not identifier or identifier not in list(self.user_to_identifiers_and_categories[user_id].keys()):
                raise ValueError(f'No identifier provided in parameters or identifier not used for finetuning\n'
                                 f'(identifier: "{identifier}", '
                                 f'used identifiers: {list(self.user_to_identifiers_and_categories[user_id].keys())})')
        else:
            identifier = None

        model_path = self._get_model_dir(user_id, identifier)
        if 'experimental' in parameters.keys():
            model_path = str(model_path) + '-experimental'

        doc = self._generate(model_path=model_path, prompt=prompt)
        torch.cuda.empty_cache()

        return DocumentArray(doc)

    def _generate(self, model_path: str, prompt: str) -> Document:
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
        for identifier in BIGDreamBoothExecutor.RARE_IDENTIFIERS:
            if identifier not in used_identifiers:
                return identifier
        raise RuntimeError('No identifier left for this user. Please, inform the administrator.')


def download_pretrained_stable_diffusion_model(model_dir: str, sd_version: str = 'stable-diffusion-v1-4'):
    """Downloads pretrained stable diffusion model."""
    if not all(os.path.exists(os.path.join(model_dir, _dir)) for _dir in [
        BIGDreamBoothExecutor.PRE_TRAINDED_MODEL_DIR, BIGDreamBoothExecutor.METAMODEL_DIR,
        f"{BIGDreamBoothExecutor.METAMODEL_DIR}-experimental",
    ]):
        pipe = StableDiffusionPipeline.from_pretrained(f"CompVis/{sd_version}", use_auth_token=True)
        for _dir in [
            BIGDreamBoothExecutor.PRE_TRAINDED_MODEL_DIR, BIGDreamBoothExecutor.METAMODEL_DIR,
            f"{BIGDreamBoothExecutor.METAMODEL_DIR}-experimental",
        ]:
            pipe.save_pretrained(os.path.join(model_dir, _dir))


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
