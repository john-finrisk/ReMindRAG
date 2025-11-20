from .base import EmbeddingBase
import vertexai
from vertexai.language_models import TextEmbeddingModel
import numpy as np
from typing import List

class VertexEmbedding(EmbeddingBase):
    def __init__(self, project_id: str, location: str, model_name: str = "text-embedding-preview-0409"):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        
        # Initialize Vertex AI SDK
        vertexai.init(project=project_id, location=location)
        self.model = TextEmbeddingModel.from_pretrained(model_name)

    def sentence_embedding(self, sentence: str) -> np.array:
        if not sentence:
            return np.array([])
        
        embeddings = self.model.get_embeddings([sentence])
        return np.array(embeddings[0].values)

    def sentence_list_embedding(self, sentences: List[str]) -> np.array:
        if not sentences:
            return np.array([])
        
        # Vertex AI typically has a batch limit (often 250). We use 100 to be safe.
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batch_embeddings = self.model.get_embeddings(batch)
            all_embeddings.extend([e.values for e in batch_embeddings])
            
        return np.array(all_embeddings)
    
    def get_hidden_state_size(self) -> int:
        # Get a dummy embedding to check size
        example_embedding = self.sentence_embedding("example")
        return len(example_embedding)
