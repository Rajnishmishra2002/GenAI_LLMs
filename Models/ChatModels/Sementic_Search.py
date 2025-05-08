from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

# Step 1: Load documents and create embeddings
documents = [
    "Python is a great programming language for data science.",
    "Laptops with high RAM and GPU are good for machine learning.",
    "JavaScript is popular for web development.",
    "MacBooks are preferred by many software engineers."
]

# Convert text to vector embeddings
embedding_model = OpenAIEmbeddings()
doc_vectors = embedding_model.embed_documents(documents)

# Step 2: Store embeddings in FAISS vector store
vector_db = FAISS.from_embeddings(doc_vectors, documents)

# Step 3: User query
query = "Which laptop is best for AI projects?"
query_vector = embedding_model.embed_query(query)

# Step 4: Search for the most relevant document
search_result = vector_db.similarity_search(query_vector, k=1)
print("Top result:", search_result[0])
