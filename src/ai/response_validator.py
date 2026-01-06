
from jsonschema import validate

SCHEMA = {'type': 'object', 'properties': {'status': {'type': 'string'}}, 'required': ['status']}

def validate_response(data: dict) -> bool:
    validate(instance=data, schema=SCHEMA)
    return True
