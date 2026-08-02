import os
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import TokenTextSplitter

file_path = os.path.join(os.path.dirname(__file__), 'ch11.pdf')
data = PyPDFLoader(file_path)
docs = data.load()

splitter = TokenTextSplitter(
  chunk_size = 100,
  chunk_overlap=10
)
chunk=splitter.split_documents(docs)
print(len(chunk))
print(chunk[5].page_content)