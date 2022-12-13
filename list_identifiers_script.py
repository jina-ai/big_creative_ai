from jina import Client
import hubble


client = Client(host='grpc://87.191.159.105:51111')

identifier_n_categories = client.post(
    on='/list_identifiers_n_categories',
    parameters={
        'jwt': {
            'token': hubble.get_token(),
        },
    }
)

print(f"Used identifiers & their categories: {identifier_n_categories[0].tags}")
