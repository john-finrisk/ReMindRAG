import sys
sys.path.append('../')

from ReMindRag.llms import OpenaiAgent
from ReMindRag.llms import GeminiAgent
from ReMindRag.embeddings import VertexEmbedding #, HgEmbedding
from ReMindRag.chunking import NaiveChunker
from ReMindRag import ReMindRag
from ReMindRag.webui import launch_webui

import json
from datetime import datetime
from transformers import AutoTokenizer

# Step 1: Get Basic Information
with open('../api_key.json', 'r', encoding='utf-8') as file:
    api_data = json.load(file)

base_url = api_data[0]["base_url"]
api_key = api_data[0]["api_key"]
model_cache_dir = "../model_cache"

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = f"logs/log_{timestamp}.log"


# Step 2: Load Base Components
# chunk_agent = OpenaiAgent(base_url, api_key, "gpt-4o-mini")
# generate_agent = OpenaiAgent(base_url, api_key, "gpt-4o-mini")
gemini_agent = GeminiAgent(
    project_id="finrisk-sandbox", 
    location="us-central1", 
    model_name="gemini-2.5-flash"
)
# embedding = HgEmbedding("nomic-ai/nomic-embed-text-v2-moe", model_cache_dir)
embedding = VertexEmbedding(
    project_id="finrisk-sandbox", 
    location="us-central1",        
    model_name="text-embedding-preview-0409"
)
chunker = NaiveChunker("nomic-ai/nomic-embed-text-v2-moe", model_cache_dir, max_token_length=750)
tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v2-moe", cache_dir = model_cache_dir)



# Step 3: Create ReMindRag Instance
rag_instance = ReMindRag(
    logger_level = 10,
    log_path= log_path,
    # chunk_agent = chunk_agent, 
    # kg_agent= generate_agent,
    # generate_agent = generate_agent, 
    chunk_agent = gemini_agent,    
    kg_agent= gemini_agent,        
    generate_agent = gemini_agent,
    embedding = embedding,
    chunker = chunker,
    tokenizer=tokenizer,
    # database_description = "DnD Player Handbook---Paladin"
    database_description = "Francis Bacon's The Advancement of Learning"
    )

# Step 4: Load Content
rag_instance.load_file("./example/ebook.md",language="en")

# Step 5: Ask A Question
query = "What does a level 20 paladin gain?"
response, _, _ = rag_instance.generate_response(query, force_do_rag=True)
print(f"\n\nQuery: {query}")
print(f"\n\nResponse: {response}")


# Step 6: Launch WebUI Backend
launch_webui(rag_instance)