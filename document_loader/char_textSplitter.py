import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
  separator="",
  chunk_size=5,
  chunk_overlap=1
)

file_path = os.path.join(os.path.dirname(__file__), 'notes.txt')
data = TextLoader(file_path)
docs = data.load()

chunks = splitter.split_documents(docs)

for i in chunks:
    print(i.page_content)