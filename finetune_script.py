from jina import Client, DocumentArray
import hubble

# specify the path to the images
path_to_instance_images = '/path/to/instance/images'
# specify the category of the images, this could be e.g. 'painting', 'dog', 'bottle', etc.
category = 'painting'
# 'private' for training private model from pretrained model, 'meta' for training metamodel
target_model = 'private'

# some custom parameters for the training
max_train_steps = 300
learning_rate = 1e-6


docs = DocumentArray.from_files(f'{path_to_instance_images}/**')
for doc in docs:
    doc.load_uri_to_blob()
    doc.uri = None

client = Client(host='grpc://87.191.159.105:51111')

identifier_doc = client.post(
    on='/finetune',
    inputs=docs,
    parameters={
        'jwt': {
            'token': hubble.get_token(),
        },
        'category': category,
        'target_model': target_model,
        'learning_rate': learning_rate,
        'max_train_steps': max_train_steps,
    },
)

print(f"Finetuning was successful. The identifier for the object is '{identifier_doc[0].text}'")
