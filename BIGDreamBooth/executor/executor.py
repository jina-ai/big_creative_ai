import glob
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from typing import List, Dict, Tuple

from PIL import Image
from accelerate import Accelerator, notebook_launcher
from accelerate.utils import write_basic_config
from diffusers import StableDiffusionPipeline
from docarray import Document
from huggingface_hub import login as hf_login
from jina import DocumentArray
import torch
from tqdm import tqdm

from .auth import NOWAuthExecutor as Executor, secure_request, SecurityLevel, _get_user_info
from .dreambooth import PromptDataset, parse_args as parse_args_db, main_given_list as main_db


class BIGDreamBoothExecutor(Executor):
    """BIGDreamBoothExecutor trains Stable Diffusion"""

    PRE_TRAINED_MODEL_ID = 'pretrained'
    PRE_TRAINDED_MODEL_DIR = 'stable-diffusion-v1-4'
    METAMODEL_ID = 'meta'
    METAMODEL_DIR = 'metamodel'
    PRIVATE_METAMODEL_ID = 'private_meta'
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
        'mrm',
        'msm',
        'mtm',
        'yry',
        'zsz',
        'qrq',
        'qsq',
        'qtr',
        'pzp',
        'qzq',
        'qyq',
        'byb',
        'klk',
        'kjk',
        'kmk',
        'kpk',
        'kqk',
        'kzk',
        'kxk',
        'kck',
        'kdk',
        'cic',
        'cjc',
        'ckc',
        'clc',
        'cmc',
        'cpc',
        'cqc',
        'crc',
        'csc',
        'ctc',
    ]

    DEFAULT_LEARNING_RATE = 5e-6
    DEFAULT_MAX_TRAIN_STEPS = 200
    MAX_MAX_TRAIN_STEPS = 1500

    def __init__(
            self,
            hf_token: str,
            is_colab: bool = False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        hf_login(token=hf_token)

        self.is_colab = is_colab

        self.models_dir = os.path.join(self.workspace, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        self.category_images_dir = os.path.join(self.workspace, 'category_images')
        os.makedirs(self.category_images_dir, exist_ok=True)
        self.metamodel_instance_images_dir = lambda _user_id: \
            os.path.join(self.workspace, 'metamodel_instance_images', _user_id)
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
        elif user_id == self.PRE_TRAINED_MODEL_ID:
            return os.path.join(self.models_dir, self.PRE_TRAINDED_MODEL_DIR)
        elif user_id.endswith(self.PRIVATE_METAMODEL_ID):
            return os.path.join(self.models_dir, *user_id.split('-'))
        return os.path.join(self.models_dir, user_id, identifier)

    @staticmethod
    def _get_user_id(parameters: Dict = None):
        """Returns the user_id of the model which shall be finetuned.
        Using 'private' in the parameters for 'target_model' will return the user_id of the user who sent
        the request. Using METAMODEL_ID or PRE_TRAINED_MODEL_ID will return the user_id of the metamodel or pretrained.
        """
        target_model = parameters.get('target_model', 'private')
        if target_model == 'private':
            user_id = _get_user_info(parameters['jwt']['token'])['_id']
        elif target_model == BIGDreamBoothExecutor.PRIVATE_METAMODEL_ID:
            user_id = _get_user_info(parameters['jwt']['token'])['_id'] + '-' \
                      + BIGDreamBoothExecutor.PRIVATE_METAMODEL_ID
        elif target_model == BIGDreamBoothExecutor.METAMODEL_ID:
            user_id = BIGDreamBoothExecutor.METAMODEL_ID
        elif target_model == BIGDreamBoothExecutor.PRE_TRAINED_MODEL_ID:
            user_id = BIGDreamBoothExecutor.PRE_TRAINED_MODEL_ID
        else:
            raise ValueError(f'Unknown target model {target_model}; must be either "private" or '
                             f'"{BIGDreamBoothExecutor.METAMODEL_ID}" or "{BIGDreamBoothExecutor.PRE_TRAINED_MODEL_ID}'
                             f' or "{BIGDreamBoothExecutor.PRIVATE_METAMODEL_ID}"')
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
                    'private': self.user_to_identifiers_and_categories.get(user_id, {}),
                    self.PRIVATE_METAMODEL_ID: self.user_to_identifiers_and_categories.get(
                        user_id + '-' + self.PRIVATE_METAMODEL_ID, {}
                    ),
                    self.METAMODEL_ID: self.user_to_identifiers_and_categories.get(self.METAMODEL_ID)
                }
            )
        )

    @secure_request(SecurityLevel.ADMIN, on='/admin/reset_model')
    def reset_model(self, parameters: Dict, **kwargs):
        """Resets the model for given model directory."""
        user_id, identifier, model_path = self._get_user_id_identifier_model_path(parameters, generate_id=False)
        self.logger.info(f'Deleting model {model_path}')
        if user_id == self.METAMODEL_ID or user_id.endswith('-' + self.PRIVATE_METAMODEL_ID):
            self.user_to_identifiers_and_categories[user_id] = {}
            pipe = StableDiffusionPipeline.from_pretrained(os.path.join(self.models_dir, self.PRE_TRAINDED_MODEL_DIR))
            pipe.save_pretrained(model_path)
            # delete all subdirectories of self.metamodel_instance_images_dir
            for sub_dir in os.listdir(self.metamodel_instance_images_dir(user_id)):
                shutil.rmtree(os.path.join(self.metamodel_instance_images_dir(user_id), sub_dir))
        else:
            del self.user_to_identifiers_and_categories[user_id][identifier]
            shutil.rmtree(model_path)
        with open(self.user_to_identifiers_and_categories_path, 'w') as f:
            json.dump(self.user_to_identifiers_and_categories, f)
        return None

    def _get_user_id_identifier_model_path(self, parameters: Dict, generate_id: bool) -> Tuple[str, str, str]:
        user_id = self._get_user_id(parameters)
        if generate_id:
            identifier = self._get_next_identifier(list(self.user_to_identifiers_and_categories[user_id].keys()))
        elif user_id in [self.METAMODEL_ID, self.PRE_TRAINED_MODEL_ID] or user_id.endswith(self.PRIVATE_METAMODEL_ID):
            identifier = None
        else:
            identifier = parameters.get('identifier', '')
            if not identifier or identifier not in list(self.user_to_identifiers_and_categories[user_id].keys()):
                raise ValueError(f'No identifier provided in parameters or identifier not used for finetuning\n'
                                 f'(identifier: "{identifier}", '
                                 f'used identifiers: {list(self.user_to_identifiers_and_categories[user_id].keys())})')
        model_path = self._get_model_dir(user_id, identifier)
        return user_id, identifier, model_path

    def _get_cat_inst_to_num_images_metamodel(
            self, num_category_images: int, cur_cat: str, cat_to_prev_ids: Dict[str, List[str]]
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        num_other_cats = len([_cat for _cat in cat_to_prev_ids.keys() if _cat != cur_cat])
        # half of all images are for the current category, the rest are split evenly among the other categories
        if num_other_cats > 0:
            num_category_images_cur_cat = num_category_images // 2
            num_category_images_other_cat = num_category_images // 2 // num_other_cats
        else:
            num_category_images_cur_cat = num_category_images
            num_category_images_other_cat = 0

        if cur_cat not in cat_to_prev_ids.keys():
            category2num_images = {cur_cat: num_category_images_cur_cat}
            instance2num_images = {}
        else:
            category2num_images = {cur_cat: num_category_images_cur_cat // 2}
            instance2num_images = {
                _prev_id: num_category_images_cur_cat // 2 // len(cat_to_prev_ids[cur_cat])
                for _prev_id in cat_to_prev_ids[cur_cat]}

        for _cat, _prev_ids in cat_to_prev_ids.items():
            if _cat == cur_cat:
                continue
            category2num_images[_cat] = num_category_images_other_cat // 2
            for _prev_id in _prev_ids:
                instance2num_images[_prev_id] = num_category_images_other_cat // 2 // len(_prev_ids)

        self.logger.info(f'_get_cat_inst_to_num_images_metamodel(num_category_images={num_category_images}, '
                         f'cur_cat={cur_cat}, cat_to_prev_ids={cat_to_prev_ids}):\n'
                         f'category2num_images = {category2num_images}\n'
                         f'instance2num_images = {instance2num_images}')
        return category2num_images, instance2num_images

    def _get_prior_preservation_loss_data_dirs_prompts(
            self, user_id: str, identifier: str, instance_images: DocumentArray, category: str, max_train_steps: int,
            tmp_dir
    ) -> Tuple[List[str], List[str]]:
        # if metamodel, then save instance images to disk
        if user_id == self.METAMODEL_ID or user_id.endswith(self.PRIVATE_METAMODEL_ID):
            _metamodel_instance_data_dir = os.path.join(self.metamodel_instance_images_dir(user_id), identifier)
            os.makedirs(_metamodel_instance_data_dir, exist_ok=True)
            for i, doc in enumerate(instance_images):
                if doc.blob:
                    doc.convert_blob_to_image_tensor()
                elif doc.uri:
                    doc.convert_uri_to_image_tensor(timeout=10)
                doc.save_image_tensor_to_file(
                    file=os.path.join(_metamodel_instance_data_dir, f'{i}.jpeg'),
                    image_format='jpeg'
                )

        # get quantities needed
        if user_id == self.METAMODEL_ID or user_id.endswith(self.PRIVATE_METAMODEL_ID):
            cat2prev_ids = defaultdict(list)
            instance2category = self.user_to_identifiers_and_categories[user_id]
            for _id, _cat in instance2category.items():
                cat2prev_ids[_cat].append(_id)
            category2num_images, instance2num_images = self._get_cat_inst_to_num_images_metamodel(
                num_category_images=max_train_steps, cur_cat=category, cat_to_prev_ids=cat2prev_ids
            )
        else:
            category2num_images = {category: max_train_steps}
            instance2num_images = {}
            instance2category = {}

        # generate category images if needed
        for _category, _num_images in category2num_images.items():
            _category_data_dir = os.path.join(self.category_images_dir, _category)
            os.makedirs(_category_data_dir, exist_ok=True)
            _num_existing_images = len(glob.glob(os.path.join(_category_data_dir, '*.jpeg')))
            if _num_existing_images < _num_images:
                self.logger.info(f'Generating {_num_images - _num_existing_images} images for category {_category}')
                _category_images = self._generate(
                    num_images=_num_images,
                    prompt=_category,
                    model_path=os.path.join(self.models_dir, self.PRE_TRAINDED_MODEL_DIR),
                    batch_size=4 if self.is_colab else 8
                )
                torch.cuda.empty_cache()
                for i, doc in enumerate(_category_images):
                    doc.convert_blob_to_image_tensor()
                    doc.save_image_tensor_to_file(
                        file=os.path.join(_category_data_dir, f'{_num_existing_images + i}.jpeg'), image_format='jpeg'
                    )

        # copy images to tmp dir
        data_dirs = []
        prompts = []
        for _category, _num_images in category2num_images.items():
            _category_data_dir = os.path.join(self.category_images_dir, _category)
            _tmp_category_data_dir = os.path.join(tmp_dir, _category)
            os.makedirs(_tmp_category_data_dir, exist_ok=True)
            for _file in glob.glob(os.path.join(_category_data_dir, '*.jpeg'))[:_num_images]:
                shutil.copy(_file, _tmp_category_data_dir)
            data_dirs.append(_tmp_category_data_dir)
            prompts.append(f"a {_category}")

        for _instance, _num_images in instance2num_images.items():
            _instance_data_dir = os.path.join(self.metamodel_instance_images_dir(user_id), _instance)
            _tmp_instance_data_dir = os.path.join(tmp_dir, _instance)
            os.makedirs(_tmp_instance_data_dir, exist_ok=True)
            while len(glob.glob(os.path.join(_tmp_instance_data_dir, '*.jpeg'))) < _num_images:
                difference = _num_images - len(glob.glob(os.path.join(_tmp_instance_data_dir, '*.jpeg')))
                for _file in glob.glob(os.path.join(_instance_data_dir, '*.jpeg'))[:difference]:
                    _file_name = os.path.basename(_file)
                    _file_name = _file_name.split('.')[0] + str(time.time()).replace('.', '-') + '.jpeg'
                    shutil.copy(_file, os.path.join(_tmp_instance_data_dir, _file_name))
            data_dirs.append(_tmp_instance_data_dir)
            prompts.append(f"a {_instance} {instance2category[_instance]}")

        return data_dirs, prompts

    @secure_request(SecurityLevel.USER, on='/finetune')
    def finetune(self, docs: DocumentArray, parameters: Dict = None, **kwargs):
        """Finetunes stable diffusion model with DreamBooth for given object and returns the used identifier for that
        object.
        :param docs: The images of the object to finetune the model with.
        :param parameters: The parameters for the finetuning; must contain the key 'target_model' which can be either
            'private' or METAMODEL_ID, where 'private' will finetune the model of the user who sent the request and
            METAMODEL_ID will finetune the metamodel; must contain the key 'category' which is the category of the
            object/style
        :return: The identifier used for the finetuning, which can be used to generate images of the object
        """
        category = parameters['category']

        learning_rate = parameters.get('learning_rate', self.DEFAULT_LEARNING_RATE)
        max_train_steps = int(parameters.get('max_train_steps', self.DEFAULT_MAX_TRAIN_STEPS))
        if max_train_steps < 0 or max_train_steps > self.MAX_MAX_TRAIN_STEPS:
            raise ValueError(
                f'Expected max_train_steps to be in [0, {self.MAX_MAX_TRAIN_STEPS}] but got {max_train_steps}')
        self.logger.info(f'Using learning rate {learning_rate}, max training steps {max_train_steps} '
                         f'for category {category}')

        user_id, identifier, output_dir = self._get_user_id_identifier_model_path(parameters, generate_id=True)
        assert user_id != self.PRE_TRAINED_MODEL_ID, f"User id {user_id} is not allowed"
        if user_id == self.METAMODEL_ID:
            pretrained_model_dir = output_dir
        elif user_id.endswith(self.PRIVATE_METAMODEL_ID):
            pretrained_model_dir = output_dir if os.path.exists(output_dir) else os.path.join(
                self.models_dir, self.PRE_TRAINDED_MODEL_DIR)
        else:
            pretrained_model_dir = os.path.join(self.models_dir, self.PRE_TRAINDED_MODEL_DIR)

        self.logger.info(f'Finetuning model in {output_dir} model with identifier {identifier} and category {category}')

        instance_prompt = f"a {identifier} {category}"

        # create temporary folder for instance data
        with tempfile.TemporaryDirectory() as tmp_dir:
            # create sub folder for instance data
            instance_data_dir = os.path.join(tmp_dir, 'instance_data')
            os.makedirs(instance_data_dir, exist_ok=True)
            # save documents as pngs
            for doc in docs:
                if doc.blob:
                    doc.convert_blob_to_image_tensor()
                elif doc.uri:
                    doc.load_uri_to_image_tensor(timeout=10)
                doc.blob = ndarray_to_jpeg_bytes(doc.tensor)
                doc.save_blob_to_file(file=os.path.join(instance_data_dir, f'{doc.id}.jpeg'))
            # class data
            pp_data_dirs, pp_prompts = self._get_prior_preservation_loss_data_dirs_prompts(
                user_id=user_id, identifier=identifier, category=category, max_train_steps=max_train_steps,
                tmp_dir=tmp_dir, instance_images=docs,
            )

            # execute dreambooth.py
            cur_dir = os.path.abspath(os.path.join(__file__, '..'))
            if not os.path.exists(os.path.join(cur_dir, 'dreambooth.py')):
                raise FileNotFoundError(f'Could not find dreambooth.py in {cur_dir}')
            # note this the output and error are switched for accelerate launch dreambooth.py
            cmd_args = [
                "--pretrained_model_name_or_path", f"{pretrained_model_dir}",
                "--output_dir", f"{output_dir}",
                "--instance_data_dir", f"{instance_data_dir}", "--instance_prompt", f"{instance_prompt}",
                "--class_data_dir", ','.join(pp_data_dirs), "--class_prompt", ','.join(pp_prompts),
                '--with_prior_preservation',
                "--resolution", "512",
                "--learning_rate", f"{learning_rate}", "--lr_scheduler", "constant", "--lr_warmup_steps", "0",
                "--max_train_steps", f"{max_train_steps}", "--num_class_images", f"{max_train_steps}",
                "--train_batch_size", "1" if self.is_colab else "2",
                # "--gradient_accumulation_steps", "2", "--gradient_checkpointing", "--use_8bit_adam",
            ] + parameters.get('dreambooth_args', [])
            self.logger.info(f'Executing {" ".join(cmd_args)}')
            print(f'Executing {" ".join(cmd_args)}')
            if self.is_colab:
                # args_parsed = parse_args_db(cmd_args)
                notebook_launcher(main_db, cmd_args)
            else:
                cmd_args = ['accelerate', 'launch', f"{cur_dir}/dreambooth.py",] + cmd_args
                output, err = cmd(cmd_args)
                for cmd_ret in [output, err]:
                    if cmd_ret:
                        error_message = cmd_ret.decode('utf-8')
                        if 'error' in error_message.lower():
                            error_message_print = f"----------\nOutput:"
                            for line in error_message.splitlines():
                                error_message_print += '\n' + line
                            error_message_print += '\n----------'
                            print(error_message_print, file=sys.stderr)
                            raise RuntimeError(
                                f'Error while executing dreambooth.py:'
                                f'{" ".join(cmd_args)}\n{error_message_print}'
                                # f"{err.decode('utf-8').split('ERROR')[-1]}"
                            )

        self.user_to_identifiers_and_categories[user_id][identifier] = category
        with open(self.user_to_identifiers_and_categories_path, 'w') as f:
            json.dump(self.user_to_identifiers_and_categories, f)

        return DocumentArray(Document(text=identifier))

    @secure_request(level=SecurityLevel.USER, on='/generate')
    def generate(self, docs: DocumentArray, parameters: Dict = None, **kwargs):
        """Generates images of the object with the given identifier.

        :param docs: Prompts for the generation of the images.
        :param parameters: The parameters for the generation; must contain the key 'target_model' which can be either
            'private' or METAMODEL_ID, where 'private' will generate images of the model of the user who sent the request and
            METAMODEL_ID will generate images of the metamodel; if using private model, must contain the key 'identifier'
            which is the identifier used to fit original images of object
        :return: The generated image
        """
        num_images = int(parameters.get('num_images', 1))
        user_id, identifier, model_path = self._get_user_id_identifier_model_path(parameters, generate_id=False)

        for doc in docs:
            prompt = doc.text.strip()
            doc.chunks = self._generate(
                num_images=num_images,
                model_path=model_path,
                prompt=prompt,
                batch_size=4 if self.is_colab else 8
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _generate(num_images: int, model_path: str, prompt: str, batch_size: int) -> DocumentArray:
        accelerator = Accelerator()

        torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
        pipeline.safetyer_checker = None
        pipeline.set_progress_bar_config(disable=True)

        sample_dataset = PromptDataset(prompt, num_images)
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=batch_size)

        sample_dataloader = accelerator.prepare(sample_dataloader)
        pipeline.to(accelerator.device)

        docs = DocumentArray()
        for example in tqdm(
                sample_dataloader, desc="Generating images", disable=not accelerator.is_local_main_process
        ):
            images = pipeline(example["prompt"]).images
            for image in images:
                with io.BytesIO() as buffer:
                    # save image as jpeg to save space with quality 95
                    image.save(buffer, format="JPEG", quality=95)
                    docs.append(Document(blob=buffer.getvalue()))

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return docs

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
    ]):
        pipe = StableDiffusionPipeline.from_pretrained(f"CompVis/{sd_version}", use_auth_token=True)
        for _dir in [
            BIGDreamBoothExecutor.PRE_TRAINDED_MODEL_DIR, BIGDreamBoothExecutor.METAMODEL_DIR,
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


def ndarray_to_jpeg_bytes(arr) -> bytes:
    pil_img = Image.fromarray(arr)
    pil_img.thumbnail((512, 512))
    pil_img = pil_img.convert('RGB')
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format="JPEG", quality=95)
    return img_byte_arr.getvalue()
