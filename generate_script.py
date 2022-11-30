import time
from pathlib import Path

from jina import Client, Document
import hubble

prompt = 'a picasso painting of a sks [class_name]'
identifier = 'sks'

# use the first host if accessing from outside Berlin office, else use the second one
host = 'grpc://87.191.159.105:51111'
host = 'grpc://192.168.178.31:51111'


client = Client(host=host)

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
