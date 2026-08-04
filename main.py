# 1. Imports
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# 2. Load API keys from .env
load_dotenv()

# 3. Create the embedding model (converts text -> vectors)
embedding_model = MistralAIEmbeddings()

# 4. Load the existing vector store from the chroma-db folder
vectorstore = Chroma(
    persist_directory= "chroma-db",
    embedding_function=embedding_model
)

# 5. Create the retriever that searches the vector store
#    - mmr: balanced, diverse results
#    - k=4: return 4 most relevant chunks
#    - fetch_k=10: pull 10 candidates first for diversity
#    - lambda_mult=0.5: balance between relevance and diversity
retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k":10,
        "lambda_mult" :0.5
    }
)

# 6. Create the LLM that will generate answers
llm = ChatMistralAI(model = "mistral-small-2506")

# 7. Create the prompt template (system rules + human question)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("Rag system created ")

print("press 0 to exit ")

# 8. Chat loop
while True:
    query = input("You : ")
    if query == "0":
        break 

    # 8a. Retrieve the most relevant chunks for the query
    docs = retriever.invoke(query)

    # 8b. Combine the retrieved chunks into a single context string
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    # 8c. Fill the prompt template with the context and question
    final_prompt = prompt.invoke({
        "context" :context,
        "question": query
    })
    
    # 8d. Ask the LLM to generate an answer
    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")
    