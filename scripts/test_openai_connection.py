
import os
from src.ai.openai_client import OpenAIClient

def main():
    client = OpenAIClient()
    if not os.getenv('OPENAI_API_KEY'):
        print('skip: missing key')
        return
    print(client.analyze('ping'))

if __name__ == '__main__':
    main()
