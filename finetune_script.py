from jina import Client, DocumentArray
import hubble

docs = DocumentArray.from_files('path_to_files/*.png')
class_name = 'dog'

# use the first host if accessing from outside Berlin office, else use the second one
host = 'grpc://87.191.159.105:51111'
host = 'grpc://192.168.178.31:51111'


for doc in docs:
    doc.load_uri_to_blob()
    doc.uri = None

client = Client(host=host)

identifier_doc = client.post(
    on='/finetune',
    inputs=docs,
    parameters={
        'jwt': {
            'token': hubble.get_token(),
        },
        'class_name': class_name,
        'target_model': 'own',  # 'own' for training from pretrained model, 'meta' for training metamodel
    },
)

print(f"Finetunging was successful. The identifier for the object is {identifier_doc[0].text}")
