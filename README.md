# Agente de Recomendações iFood (Demo RAG)

Este projeto é uma demonstração técnica de um Agente de IA Conversacional construído para simular uma experiência de descoberta de pratos no iFood.

O objetivo é ir além de filtros e buscas por palavras-chave, permitindo que o usuário "converse" com o catálogo para encontrar pratos usando linguagem natural, contexto e memória.

Este projeto foi desenvolvido como um estudo de caso focando nas habilidades de desenvolvimento de agentes, LLMs, LLMOps e LangChain.

## Demo em Ação
<img src="https://raw.githubusercontent.com/tomielopedro/ifood-ai-recommendation/main/img/demo.png" alt="Demonstração do Agente em Ação" width="600"/>

## Funcionalidades Principais

*   **Interface de Chat Conversacional:** Construído com Streamlit para interação em tempo real.
*   **Memória de Conversa:** O agente entende o contexto. Você pode fazer perguntas de acompanhamento como "E qual deles é o mais barato?" e ele saberá que "deles" se refere aos pratos da pergunta anterior.
*   **Busca Semântica (RAG):** Utiliza um pipeline de Retrieval-Augmented Generation (RAG) para encontrar os pratos mais relevantes no "banco de dados" (.json) com base no significado da pergunta, não apenas em palavras-chave.
*   **LLM Local & Privado:** Roda 100% local usando Ollama, garantindo que nenhum dado (como o cardápio ou as perguntas) saia da máquina.
*   **Configurável:** Os modelos de LLM e de embedding podem ser trocados instantaneamente através de um arquivo `.env`, sem tocar no código-fonte.

## Stack Tecnológica

| Componente | Tecnologia | Descrição |
| :--- | :--- | :--- |
| **Frontend** | Streamlit | Interface de usuário para o chat. |
| **Orquestração de IA** | LangChain (LCEL) | Framework para construção do pipeline RAG. |
| **Modelos de Linguagem (LLM)** | Ollama (ex: `phi3:mini`) | LLM local para geração de respostas. |
| **Modelos de Embedding** | Ollama (ex: `nomic-embed-text`) | Modelo local para busca semântica. |
| **Banco de Dados Vetorial** | FAISS-CPU | Para busca de similaridade em memória. |
| **Core** | Python 3 | Linguagem de programação principal. |

# Arquitetura: O Pipeline de RAG Conversacional

Esta é a parte mais importante do projeto. O agente não é apenas um LLM respondendo perguntas. Ele segue um pipeline RAG sofisticado orquestrado com LangChain (LCEL) para garantir que as respostas sejam baseadas apenas nos dados dos restaurantes.

<img src="https://raw.githubusercontent.com/tomielopedro/ifood-ai-recommendation/main/img/diagrama.png" alt="Diagrama de arquitetura" width="600"/>


O fluxo de uma pergunta do usuário é o seguinte:

1.  **Entrada do Usuário:** O usuário envia uma nova pergunta (ex: "E algum sem glúten?").

2.  **Passo de "Reescrita" (History-Aware):**
    *   O `app.py` envia a nova pergunta e o histórico do chat (`chat_history`) para o `history_aware_retriever`.
    *   Este retriever usa um LLM (via `get_contextualize_prompt`) para reescrever a nova pergunta em uma consulta completa e independente.
    *   Exemplo: "E algum sem glúten?" se torna "Qual prato vegetariano sem glúten você tem?".

3.  **Passo de "Recuperação" (Retrieval):**
    *   A nova pergunta reescrita (uma string) é enviada ao banco de dados vetorial FAISS.
    *   O FAISS realiza uma busca de similaridade semântica e retorna os `k` documentos (pratos) mais relevantes do nosso `restaurantes.json`.

4.  **Passo de "Geração" (Generation):**
    *   O pipeline final (`conversational_rag_chain`) coleta todas as informações:
        *   A Pergunta Original do usuário (`input`).
        *   O Histórico do Chat (`chat_history`) (para manter o tom da conversa).
        *   O Contexto Recuperado (`context`) (os pratos encontrados pelo FAISS).
    *   Essas três peças são injetadas no prompt principal (`get_rag_prompt`).
    *   Este prompt instrui o LLM a gerar uma resposta amigável usando apenas o contexto fornecido.

5.  **Resposta Final:** O `StrOutputParser` extrai o texto da resposta do LLM, que é então exibido no Streamlit.

## Instalação e Execução Local

Você precisará ter o Ollama instalado e rodando em sua máquina.

1.  **Clone o Repositório:**
    ```bash
    git clone https://github.com/tomielopedro/ifood-ai-recommendation.git
    cd ifood-ai-recommendation
    ```

2.  **Crie e Ative um Ambiente Virtual:** (Recomendado usar Python 3.11 ou 3.12, pois o 3.13 pode ter problemas de compatibilidade com bibliotecas de IA).
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Instale as Dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Baixe os Modelos do Ollama:** (O pipeline precisa de um LLM de chat e um modelo de embedding).
    ```bash
    ollama pull phi3:mini
    ollama pull nomic-embed-text
    ```

5.  **Configure seu Ambiente:** Crie um arquivo `.env` na raiz do projeto (use o `.env.example` como base, se houver).

6.  **Execute o Aplicativo Streamlit:**
    ```bash
    streamlit run app.py
    ```

## Estrutura do Projeto

A estrutura segue as boas práticas de engenharia de software, separando a lógica da aplicação (`src`) da interface (`app.py`) e dos dados (`data`).

```text
/agente-ifood-langchain/
|
|-- .gitignore           # Ignora arquivos desnecessários (como .venv, .env)
|-- README.md            # (Este arquivo)
|-- requirements.txt     # Dependências do Python
|-- app.py               # Ponto de entrada - Apenas a interface (Streamlit)
|-- .env                 # Arquivo de configuração (local)
|
|-- data/
|   |-- restaurantes.json  # Nosso "banco de dados" de pratos e cardápios
|
|-- src/
|   |-- __init__.py      # Torna 'src' um pacote Python
|   |-- rag_pipeline.py  # O "cérebro" do projeto: toda a lógica de RAG e LangChain
```
