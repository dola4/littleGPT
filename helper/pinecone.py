import Pinecone
from api import PINECONE_API_KEY, ENVIRONMENT

Pinecone.init(api_key=PINECONE_API_KEY,
              environment=ENVIRONMENT)

def store_code_in_pinecone(code):
   pinecone_namespace = "generated-code"
   Pinecone.upsert(items={pinecone_namespace: code})

#IStockez le code généré dans Pinecone
store_code_in_pinecone(generated_code)