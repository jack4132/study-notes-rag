from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

print("Total chunks in DB:", vectorstore._collection.count())

results = vectorstore.similarity_search("is the pdf on Day+3+ more about kt concepts", k=3)
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print("Source:", doc.metadata)
    print(doc.page_content[:200])
