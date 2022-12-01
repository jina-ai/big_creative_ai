from jina import Client
import hubble

# use the first host if accessing from outside Berlin office, else use the second one
host = 'grpc://87.191.159.105:51111'
host = 'grpc://192.168.178.31:51111'


client = Client(host=host)

identifier_n_categories = client.post(
    on='/list_identifiers_n_categories',
    parameters={
        'jwt': {
            'token': hubble.get_token(),
        },
    }
)

print(f"Used identifiers & their categories: {identifier_n_categories[0].tags}")
