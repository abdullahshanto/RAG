from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

from langchain_core.documents import Document

load_dotenv()

docs = [
  Document(page_content="pythin is widly used in AI", metadata={"source" : "AI_book"}),
  Document(page_content="pandas is used for data analysis in python", metadata={"source":"Shantos_book"}),
  Document(page_content="neural networks are used in deep laerning", metadata={"source":"hello_book"})
]

embeddings = MistralAIEmbeddings()

vectorStore = Chroma.from_documents(
  documents = docs,
  embedding = embeddings,
  persist_directory = "chroma-db"
)

# to retrieve data from chroma db we need retrievers

result = vectorStore.similarity_search("what is used for data analysis",k=1) #k= how many documents i wanna get

for i in result:
  print(i.page_content)
