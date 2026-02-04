#!/usr/bin/env python3

# Essential.
import pandas as pd
import os
from dotenv import load_dotenv

## RAG libraries.
# Vectors
import chromadb
from chromadb.api.client import Client as ChromaClient
from chromadb import EmbeddingFunction, Documents, Embeddings

# Langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables.base import RunnableBinding

## -- SETTING PARAMETERS --
# OPENAI API KEY
load_dotenv()
OPENAI_KEY = os.getenv('OPENAI_API_KEY')

# PROJECT NAME
LEGAL_PROJ = "consultoria-juridica"
# PROJECT LOCATION
DIR = "./data/" # it has to be in .env

# Adapter for chroma.
class LangChainEmbeddingAdapter(EmbeddingFunction):
    def __init__(self, openai_ef):
        self.openai_ef = openai_ef
    def __call__(self, input:Documents) -> Embeddings:
        return self.openai_ef.embed_documents(input)
    def name(self) -> str:
        return "openai"
    
## --- END OF SETTING PARAMETERS ---

## --Cleaning--
def cleaned_dataset_from_legal_info(raw_data:pd.DataFrame, cols_to_clean:list=["Tipo", "Tema - subtema"]) -> pd.DataFrame:
    # Raw
    raw_data = pd.read_excel(raw_data)
    data = raw_data.copy()
    data[cols_to_clean] = data[cols_to_clean].fillna("")
    data.set_index("#")
    
    return data

# Initialize client in ChromaDB
def init_client_chroma(
        input_path:str, 
        llm_key:str=OPENAI_KEY, 
        model:str="text-embedding-ada-002"
    ) -> tuple[ChromaClient, LangChainEmbeddingAdapter]:
    
    # Init client
    client = chromadb.PersistentClient(path=input_path)
    
    langchain_embeddings = OpenAIEmbeddings(
        model=model,
        api_key=llm_key
    )
    # Adapter
    adapter_from_langchain_to_chroma = LangChainEmbeddingAdapter(langchain_embeddings)
    
    return client, adapter_from_langchain_to_chroma, langchain_embeddings

# Chunks for long text strings.
def create_chunks(
        data:pd.DataFrame,
        chunk_size:int=800,
        overlap:int=100
    ):
        
    # Embedding lists
    documents = []
    metadata = []
    ids = []
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    # Creating chunks
    for idx, row in data.iterrows():
        
        # Hardcoded columns
        full_text = f"TEMA : {row['Tema - subtema']} \n SINTESIS : {row['sintesis']} \n RESUELVE : {row['resuelve']}"
        
        chunks = text_splitter.split_text(full_text)
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadata.append(
                {
                    "row_index" : idx,
                    "tema" : str(row['Tema - subtema']),
                    "sentencia" : str(row['resuelve']),
                    "resumen_decision" : str(row['sintesis'])
                }
            )
            ids.append(f"doc_{idx}_chunk_{i}")
            
    return ids, documents, metadata

# Chroma collection. 
def assemble_collection(
        ids:list, # ids, documents, metadata  come from chunking 
        documents:list,
        metadata:list,
        collection_name:str, # you input it
        client:ChromaClient, # comes from an init function 
        adapter:LangChainEmbeddingAdapter, # comes from an init function 
        batch_size:int=100
    ) -> None:

    # # Create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=adapter
    )
    
    for i in range(0, len(documents), batch_size):
        # Indexing con batches.
        batch_ids = ids[i : i + batch_size]
        batch_docs = documents[i : i + batch_size]
        batch_metadatas = metadata[i : i + batch_size]
        
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metadatas
        )
        print(f"Uploaded batch {i // batch_size + 1} of {len(documents) // batch_size + 1}")
        
def get_response(user_query:str, chain:RunnableBinding):
    
    # Invoking response by bind among LLM > prompt > retrieval from vector db.
    response = chain.invoke({"input": user_query})
    formatted = {
        "output": response["answer"],
        "metadata": [doc.metadata for doc in response["context"]]
    }
    return formatted
        

if __name__ == "__main__":
    
    ### --------------------------===============----- RUN MAIN -------===========-----------------------------------
    
    ## Data from Legal Info in .xlsx format.
    DATAPATH = os.path.join(DIR, "sentencias_pasadas.xlsx")
    try:
        os.path.exists(DATAPATH)
    except FileNotFoundError as fe:
        print("No se encuentra la informacion sobre las consultas legales.")
        
    data = cleaned_dataset_from_legal_info(raw_data=DATAPATH) # cleaned.
    
    ## Client init
    chroma_client, adapter_func, langchain_embeddings = init_client_chroma(input_path=DIR)
    ## Chunking
    ids, documents, metadata = create_chunks(data=data)
    ## Vector DB creation
    db = assemble_collection(
        ids=ids,
        documents=documents,
        metadata=metadata,
        collection_name=LEGAL_PROJ,
        client=chroma_client,
        adapter=adapter_func
    )    