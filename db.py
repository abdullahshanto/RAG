
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

file_path = os.path.join(os.path.dirname(__file__), './document_loader/CN.pdf')
data = PyPDFLoader(file_path)
docs = data.load()


splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000,
  chunk_overlap=200
)

chunks = splitter.split_documents(docs)

embeddings= MistralAIEmbeddings()
vectorStore = Chroma.from_documents(
  documents = chunks,
  embedding = embeddings,
  persist_directory = "chroma-db"
)