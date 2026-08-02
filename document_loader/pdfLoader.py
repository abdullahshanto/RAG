import os
from langchain_community.document_loaders import PyPDFLoader

file_path = os.path.join(os.path.dirname(__file__), 'CN.pdf')
data = PyPDFLoader(file_path)

docs = data.load()
#print(docs)
print(len(docs)) ## each page becomes a loader