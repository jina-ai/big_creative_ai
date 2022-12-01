import time
from pathlib import Path

from jina import Client, Document
import hubble

object_style_identifier = 'sks'
prompt = f'a {object_style_identifier}'


# use the first host if accessing from outside Berlin office, else use the second one
# host = 'grpc://87.191.159.105:51111'
host = 'grpc://192.168.178.31:51111'

target_model = 'own'  # 'own' for using own model, 'meta' for using metamodel


client = Client(host=host)

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

image_docs = client.post(
    on='/generate',
    inputs=Document(text=prompt),
    parameters={
        'jwt': {
            'token': hubble.get_token(),
        },
        'identifier': object_style_identifier,
        'target_model': target_model,
    }
)

print(f"Generation was successful. ")

path = Path(f"generated_images")
path.mkdir(exist_ok=True)
image_docs[0].save_blob_to_file(f"{str(path)}/{prompt.replace(' ', '-').replace(',', '')}-{time.time()}.png")
