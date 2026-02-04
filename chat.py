#!/usr/bin/env python3

import os
import pandas as pd
from dotenv import load_dotenv
from embeddings import init_client_chroma, get_response, DIR, LEGAL_PROJ

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
        temperature=0.1
    )

    ## RESPONSE CONFIG
    retriever = SelfQueryRetriever.from_llm(
        llm=openai_llm_instance,
        vectorstore=db,
        document_contents=document_content_description,
        metadata_field_info=metadata_info,
        verbose=True
    )
    
    ## --- IMPORT PROMPT CONFIG ---
    with open(PROMPT_INSTRUCTION, "r", encoding="utf-8") as prompt:
        prompt_for_system = prompt.read()
    ## --- BINDING PROMPT AND METADATA-RICH QUERY.
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_for_system),
        ("human", "{input}"),
    ])
    chain_to_answer_query = create_stuff_documents_chain(openai_llm_instance, prompt)
    
    # Retrieval
    rag_chain = create_retrieval_chain(retriever, chain_to_answer_query) # RunnableBinding
    
    # TEST QUESTIONS.
    questions = [
        "¿Cuál fue la sentencia del caso que habla de acoso escolar?",
        "¿Cuáles son las sentencias de 3 demandas?",
        "¿De qué se trataron las 3 demandas anteriores?",
        "¿diga el detalle de la demanda relacionada con acoso escolar?",
        "¿existen casos que hablan sobre el PIAR, indique de que trataron los casos y cuáles fueron sus sentencias?"
    ]
    
    test_responses = {"Pregunta de prueba" : [], "Respuesta" : []}
    for question in questions:
        test_responses["Pregunta de prueba"].append(question)
        response = get_response(
            user_query=question,
            chain=rag_chain
        )
        test_responses["Respuesta"].append(response["output"])
    
    test_responses_dt = pd.DataFrame.from_dict(test_responses)
    test_responses_dt.to_csv("prueba_2.csv")
    
    print(test_responses_dt)