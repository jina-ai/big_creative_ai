from jina import Client, DocumentArray
import hubble

# specify the path to the images
path_to_images = '/some/path/to/images'
# specify the category of the images
category = 'dog|painting|ping pong table|...'

# use the first host if accessing from outside Berlin office, else use the second one
# host = 'grpc://87.191.159.105:51111'
host = 'grpc://192.168.178.31:51111'

# 'own' for training from pretrained model, 'meta' for training metamodel
target_model = 'own'


docs = DocumentArray.from_files(f'{path_to_images}/**')
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
        'category': category,
        'target_model': target_model
    },
)

print(f"Finetuning was successful. The identifier for the object is '{identifier_doc[0].text}'")
