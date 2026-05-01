import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model

load_dotenv()
for env in ("PG_VECTOR_COLLECTION_NAME", "DATABASE_URL", "GOOGLE_MODEL", "GOOGLE_EMBEDDING_MODEL", "GOOGLE_API_KEY"):
    if not os.getenv(env):
        raise RuntimeError(f"Variável de ambiente {env} não encontrada")

GOOGLE_MODEL = os.getenv("GOOGLE_MODEL")

def search_prompt(question=None):
    if not question:
        raise RuntimeError("Faça uma pergunta")
    
    embedding_model = GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL"))
    db = PGVector(
        embeddings=embedding_model,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )

    resultsSearch = db.similarity_search_with_score(question, k=10)

    if not resultsSearch:
        raise RuntimeError("Não foram encontrados resultados compatíveis no PGVector.")

    contexto = "\n\n".join([f"Documento {i+1} (Potuação: {score:.4f}):\n{doc.page_content}" for i, (doc, score) in enumerate(resultsSearch)])

    model = init_chat_model(model=GOOGLE_MODEL, model_provider="google_genai")    
    question_template = PromptTemplate(
      input_variables=["pergunta"],
      partial_variables={"contexto": contexto},
      template=PROMPT_TEMPLATE
    )

    chain = question_template | model

    return chain

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""