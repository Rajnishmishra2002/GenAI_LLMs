from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = [
    "India is a vast and diverse country known for its rich history, culture, and stunning landscapes. It boasts a unique blend of traditions, languages, and religions, making it a vibrant tapestry of unity in diversity. From the Himalayas to the Indian Ocean, India offers a wide range of geographical features and natural beauty. ",
    "The Republic of India",
    "2. What is the national language of India? Hindi.",
    "3. What are some of the key features of Indian culture? India's culture is characterized by its rich history, diverse traditions, vibrant festivals, and unique cuisine. It is also known for its various religious practices, including Hinduism, Islam, Christianity, Sikhism, and Buddhism. "

]


query = "tell me about India"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)


scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print(score)
