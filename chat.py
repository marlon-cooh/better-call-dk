#!/usr/bin/env python3

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

#### GLOBAL 
# OPENAI API KEY
load_dotenv()
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
PROMPT_INSTRUCTION = "./prompt_instruction.txt"

## Client init
chroma_client, _, openai_embedder = init_client_chroma(input_path=DIR, llm_key=OPENAI_KEY)

## Open vector database
db = Chroma(client=chroma_client, collection_name=LEGAL_PROJ, embedding_function=openai_embedder)

# Setting metadata.
metadata_info = [
    
    AttributeInfo(
        name="row_index",
        description="El numero especifico de la fuente del documento de Excel, leido como un DataFrame en Pandas.",
        type="integer"
    ),
    
    AttributeInfo(
        name="tema",
        description="\
            La sintesis sobre el tema de la(s) tutela(s) y proceso(s) legales llevados a cabo.",
        type="string"
    ),

    AttributeInfo(
        name="sentencia",
        description="\
            Es la decision tomada por la corte constitucional sobre el caso legal (demanda, tutela, auto), y imparte si falla o no",
        type="string"
    ),
    
    AttributeInfo(
        name="resumen_decision",
        description="\
            Se trata del resumen del fallo dado por la corte o juzgado acerca del caso.",
        type="string"
    ),
    
]

document_content_description = "Historial de demandas y sentencias legales relacionadas con redes sociales, acoso escolar y derechos educativos."

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
    
    ## --- IMPORT PROMPT ---
    with open(PROMPT_INSTRUCTION, "r", encoding="utf-8") as prompt:
        prompt_for_system = prompt.read()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_for_system),
        ("human", "{input}"),
    ])
    chain_to_answer_query = create_stuff_documents_chain(openai_llm_instance, prompt)
    
    # Retrieval
    rag_chain = create_retrieval_chain(retriever, chain_to_answer_query) # RunnableBinding
    
    # QUESTION (HARDCODED)
    question = "¿Cuál fue la sentencia del caso que habla de acoso escolar?"
    
    chat_answer = get_response(
        user_query=question,
        chain=rag_chain
    )
    print(chat_answer["output"])