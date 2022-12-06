import time
from pathlib import Path

from jina import Client, Document
import hubble

object_style_identifier = 'sks'
prompt = f'a {object_style_identifier}'


# use the first host if accessing from outside Berlin office, else use the second one
host = 'grpc://87.191.159.105:51111'
# host = 'grpc://192.168.178.31:51111'

num_images = 10

target_model = 'pretrained'  # 'own' for using own model, 'meta' for using metamodel, 'pretrained' for using pretrained model


client = Client(host=host)

if target_model != 'pretrained':
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
if target_model == 'own':
    folder_images_prefix += f'/{object_style_identifier}'
elif target_model == 'meta':
    folder_images_prefix += f'/metamodel'
elif target_model == 'pretrained':
    folder_images_prefix += f'/pretrained'
else:
    raise ValueError(f"Unknown target_model '{target_model}'")
folder_images = Path(f"{folder_images_prefix}/{prompt.replace(' ', '-').replace(',', '')}")
folder_images = Path(f"{str(folder_images)[:200]}-{time.time()}")

image_docs = client.post(
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
for i, image_doc in enumerate(image_docs):
    image_doc.save_blob_to_file(f"{str(folder_images)}/generation-{i}.png")

print(f"Generations were successful and were saved to {folder_images}")

