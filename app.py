import streamlit as st
from src.rag_pipeline import setup_rag_pipeline
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(
    page_title="AI FOOD",
    page_icon="🧠",
    layout="centered"
)

st.title("🤖  AI FOOD ")
st.divider()

try:
    rag_chain = setup_rag_pipeline()
except Exception as e:
    st.error(f"Ocorreu um erro no setup: {e}")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá! Sou seu chefe de cozinha. O que vamos pedir hoje?")
    ]

for msg in st.session_state.chat_history:
    if isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)
    else:
        with st.chat_message("user"):
            st.markdown(msg.content)

if prompt := st.chat_input("Digite seu pedido ou pergunta..."):
    
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Preparando uma recomendação deliciosa... 🍽️"):

            try:
                answer = rag_chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history[:-1]  
                })

                st.markdown(answer)


                st.session_state.chat_history.append(AIMessage(content=answer))

            except Exception as e:
                err = f"Erro ao gerar resposta: {e}"
                st.error(err)
                st.session_state.chat_history.append(AIMessage(content=err))
