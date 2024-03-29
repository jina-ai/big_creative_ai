import time
from pathlib import Path

from docarray import DocumentArray
from jina import Client, Document
import hubble

object_style_identifier = 'sks'
prompt = f'a {object_style_identifier}'
# 'private' for using private model, 'meta' for using metamodel, 'pretrained' for using pretrained model
target_model = 'private'

num_images = 10


client = Client(host='grpc://87.191.159.105:51111')

# update prompt with category of used identifiers
identifier_n_categories = client.post(
    on='/list_identifiers_n_categories',
    parameters={
        'jwt': {
            'token': hubble.get_token(),
        },
    }
)
identifier_n_categories = identifier_n_categories[0].tags[target_model]
for _identifier, _category in identifier_n_categories.items():
    prompt = prompt.replace(_identifier, f"{_identifier} {_category}")

# generate images
folder_images_prefix = 'generated_images'
if target_model == 'private':
    folder_images_prefix += f'/{object_style_identifier}'
elif target_model == 'meta':
    folder_images_prefix += f'/metamodel'
elif target_model == 'pretrained':
    folder_images_prefix += f'/pretrained'
elif target_model == 'private_meta':
    folder_images_prefix += f'/private_metamodel'
else:
    raise ValueError(f"Unknown target_model '{target_model}'")
folder_images = Path(f"{folder_images_prefix}/{prompt.replace(' ', '-').replace(',', '')}")
folder_images = Path(f"{str(folder_images)[:200]}-{time.time()}")

image_docs: DocumentArray = client.post(
    on='/generate',
    inputs=Document(text=prompt),
    parameters={
        'jwt': {
            'token': hubble.get_token(),
        },
        'identifier': object_style_identifier,
        'target_model': target_model,
        'num_images': num_images,
    }
)
folder_images.mkdir(exist_ok=True, parents=True)
for i, image_doc in enumerate(image_docs[0].chunks):
    image_doc.save_blob_to_file(f"{str(folder_images)}/generation-{i}.png")

print(f"Generations were successful and were saved to {folder_images}")
