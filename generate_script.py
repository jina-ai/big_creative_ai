import time
from pathlib import Path

from jina import Client, Document
import hubble

prompt = 'a picasso painting of a sks [class_name]'
identifier = 'sks'

client = Client(host='')

image_docs = client.post(
    on='/generate',
    inputs=Document(text=prompt),
    parameters={
        'jwt': {
            'token': hubble.get_token(),
        },
        'identifier': identifier,
        'target_model': 'own',  # 'own' for using own model, 'meta' for using metamodel
    }
)

print(f"Generation was successful. ")

path = Path(f"output")
path.mkdir(exist_ok=True)
image_docs[0].save_blob_to_file(f"{str(path)}/{prompt.replace(' ', '-').replace(',', '')}-{time.time()}.png")
