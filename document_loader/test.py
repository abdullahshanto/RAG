import os

from langchain_community.document_loaders import TextLoader

file_path = os.path.join(os.path.dirname(__file__), 'notes.txt')
data = TextLoader(file_path)
#print(data)  it returns a object
docs=data.load() # it returns metadata and page content
print(docs[0].metadata)


print(len(docs[0].page_content))

print(len(docs))
