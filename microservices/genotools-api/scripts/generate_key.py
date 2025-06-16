import secrets
import string

def generate_api_key(length=50):
    allowed_chars = string.ascii_letters + string.digits + '-'
    return ''.join(secrets.choice(allowed_chars) for _ in range(length))

api_key = generate_api_key()
print(api_key)