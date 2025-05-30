## Requirements 
``pip install re transformers uuid torch numpy json langchain_openai langchain_core`` 
## Description of functions used :
- **Chunking** : First files are read using `.read()` and the documents are given unique IDs using `uuid.uuid4` . All the files within the directory are saved with their `base` and `sku`. 
- 
