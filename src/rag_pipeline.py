import streamlit as st
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from operator import itemgetter 
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser  


load_dotenv()


def get_retriever(embeddings, docs):
    """Cria e retorna o retriever FAISS."""
    try:
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    except Exception as e:
        st.error(f"Erro ao criar vector store: {e}")
        st.stop()


def get_contextualize_prompt():
    """Prompt para reescrever a pergunta com base no histórico."""
    return ChatPromptTemplate.from_messages([
        ("system",
         "Dada a conversa e a última pergunta, formule uma pergunta independente "
         "que possa ser entendida mesmo sem o histórico. Não responda, só reformule."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])


def get_rag_prompt():
    """Prompt principal do RAG."""
    return ChatPromptTemplate.from_messages([
        ("system", """
        Você é um assistente de IA especialista do iFood.
        Use SOMENTE o conteúdo dos CONTEXTOS para responder.

        Regras:
        1. Seja amigável.
        2. Ao sugerir pratos, liste Nome do Prato, Restaurante e Preço.
        3. Não invente respostas.
        4. Se não achar nada, diga: 
           "Não consegui encontrar uma opção exata para o seu pedido, mas posso procurar por outra coisa?"
        
        CONTEXTOS:
        {context}
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])



@st.cache_resource(show_spinner="Processando e vetorizando o cardápio...")
def setup_rag_pipeline():
    """Carrega dados, embedders, LLM, FAISS e constrói o pipeline RAG conversacional."""

    json_path = Path("data/restaurantes.json")
    
    if not json_path.exists():
        st.error("Arquivo data/restaurantes.json não encontrado!")
        st.stop()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)


    docs = []
    for restaurante in data:

        restaurante_nome = restaurante.get("nome_restaurante", "N/A")
        restaurante_tipo = restaurante.get("tipo_cozinha", "N/A")
        restaurante_rating = restaurante.get("rating", "N/A")

        for item in restaurante.get("menu", []):
            
       
            content = (
                f"Nome do Prato: {item.get('nome', '')}\n"
                f"Descrição: {item.get('descricao', '')}\n"
                f"Tags: {', '.join(item.get('tags', []))}"
            )

   
            metadata = {
                "nome_restaurante": restaurante_nome,
                "tipo_cozinha": restaurante_tipo,
                "rating": restaurante_rating,
                "preco": item.get("preco", 0.0),
                "id_item": item.get("id_item", "")
            }
            

            docs.append(Document(page_content=content, metadata=metadata))




    llm_model_name = os.getenv("LLM_MODEL", "phi3:mini")

    embed_model_name = os.getenv("EMBEDDING_MODEL", "nomic-embed-text") 

    try:
        embeddings = OllamaEmbeddings(model=embed_model_name)
        llm = Ollama(model=llm_model_name, temperature=0)

        st.sidebar.info(f"Usando LLM: {llm_model_name}")
        st.sidebar.info(f"Usando Embedding: {embed_model_name}")

    except Exception as e:
        st.error(f"Erro ao conectar ao Ollama: {e}")
        st.stop()


    retriever = get_retriever(embeddings, docs)


    contextualize_prompt = get_contextualize_prompt()

    history_aware_retriever = (
        contextualize_prompt
        | llm
        | StrOutputParser() 
        | retriever
    )

    rag_prompt = get_rag_prompt()

    question_answer_chain = (
        rag_prompt
        | llm
        | StrOutputParser() 
    )


    conversational_rag_chain = (
        {

            "input": itemgetter("input"),

            "chat_history": itemgetter("chat_history"),

            "context": history_aware_retriever,
        }
        | question_answer_chain 
    )

    return conversational_rag_chain