#!/usr/bin/env python3

### FIX STRUCTURE IN THIS SCRIPT!!!!
import os
from dotenv import load_dotenv
from rag import init_client_chroma, get_response, DIR, LEGAL_PROJ

# Retrieval.
from langchain_chroma import Chroma
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI

#### GLOBAL == SO FAR.
# OPENAI API KEY
load_dotenv()
OPENAI_KEY = os.getenv('OPENAI_API_KEY')

## Client init
chroma_client, _, openai_embedder = init_client_chroma(input_path=DIR, llm_key=OPENAI_KEY)

## Open vector database
db = Chroma(client=chroma_client, collection_name=LEGAL_PROJ, embedding_function=openai_embedder)

# Setting metadata.
metadata_info = [
    AttributeInfo(
        name="tema",
        description="\
            La sintesis sobre el tema de la(s) tutela(s) y proceso(s) legales llevados a cabo.",
        type="string"
    )
]

document_content_description = "Información sobre posibles demandas y sus resultados, relacionadas mayoritariamente con redes sociales."

if __name__ == "__main__":
    
    # LLM instance (OpenAI)
    openai_llm_instance = ChatOpenAI(
        api_key=OPENAI_KEY,
        model="gpt-4o",
        temperature=0
    )

    ## RESPONSE CONFIG
    retriever = SelfQueryRetriever.from_llm(
        llm=openai_llm_instance,
        vectorstore=db,
        document_contents=document_content_description,
        metadata_field_info=metadata_info,
        verbose=True
    )
    
    prompt_for_system = (  ### FIX PROMPT!!!
        "You are a specialized legal assistant. Use the following pieces of retrieved "
        "context to answer the user's question accurately. "
        "If you don't know the answer, say that you don't know. "
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_for_system),
        ("human", "{input}"),
    ])
    chain_to_answer_query = create_stuff_documents_chain(openai_llm_instance, prompt)
    
    # Retrieval
    rag_chain = create_retrieval_chain(retriever, chain_to_answer_query) # RunnableBinding
    
    # QUESTION (HARDCODED)
    question = "Is there any case found of bullying in the school." ### FASTAPI!!!
    
    chat_answer = get_response(
        user_query=question,
        chain=rag_chain
    )
    print(chat_answer["output"])