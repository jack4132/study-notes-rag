from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vs = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
print("Total chunks:", vs._collection.count())
