import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()
file_path = os.path.join(os.path.dirname(__file__), 'ch11.pdf')
data = PyPDFLoader(file_path)
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
  chunk_size = 100,
  chunk_overlap=10
)
chunk=splitter.split_documents(docs)


template = ChatPromptTemplate.from_messages(
  [
    ("system","note down the important topic serially"),
    ("human","{chunk}")
  ]
)
model = ChatMistralAI(model = "mistral-small-2603")
prompt = template.format_messages(chunk="\n\n".join(d.page_content for d in chunk))
result = model.invoke(prompt)
print(result)
# print(len(chunk))
# print(chunk[5].page_content)