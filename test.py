from langchain_ollama import OllamaEmbeddings

# Initialize Qwen3 embedding model via Ollama
embeddings = OllamaEmbeddings(
    model="qwen3-embedding"
)

# Generate an embedding for a text query
vector = embeddings.embed_query("BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models Junnan Li Dongxu Li Silvio Savarese Steven Hoi Salesforce Research https://github.com/salesforce/LAVIS/tree/main/projects/blip2 Abstract The cost of vision-and-language pre-training has become increasingly prohibitive due to end-toend training of large-scale models. This paper proposes BLIP-2, a generic and efﬁcient pretraining strategy that bootstraps vision-language pre-training from off-the-")
print(vector)  # Print the first 5 dimensions of the vector

