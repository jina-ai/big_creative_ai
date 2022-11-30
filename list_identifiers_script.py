from jina import Client, Document
import hubble

# use the first host if accessing from outside Berlin office, else use the second one
host = 'grpc://87.191.159.105:51111'
host = 'grpc://192.168.178.31:51111'


client = Client(host=host)

identifier_n_classes = client.post(
    on='/list_identifiers_n_classes',
    parameters={
        'jwt': {
            'token': hubble.get_token(),
        },
    }
)

print(f"Used identifiers & classes: {identifier_n_classes[0].tags}")
