# Desafio MBA Engenharia de Software com IA - Full Cycle

RAG (Retrieval-Augmented Generation) com LangChain, Google Gemini e PostgreSQL + pgvector.

## Como executar

### 1. Ambiente virtual e dependências

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2. Variáveis de ambiente

Copie o arquivo de exemplo e preencha as chaves:

```bash
cp .env.example .env
```

Edite o `.env` com seus valores:

```env
GOOGLE_API_KEY=sua_chave_aqui
GOOGLE_MODEL='gemini-2.5-flash-lite'
GOOGLE_EMBEDDING_MODEL='models/gemini-embedding-001'
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=nome_da_colecao
PDF_PATH=document.pdf
```

> Obtenha sua `GOOGLE_API_KEY` em [Google AI Studio](https://aistudio.google.com/app/apikey).

### 3. Subir o banco de dados

```bash
docker compose up -d
```

O banco PostgreSQL com a extensão pgvector ficará disponível na porta `5432`.

### 4. Ingestar os dados

```bash
python src/ingest.py
```

Este script carrega o PDF definido em `PDF_PATH`, divide em chunks e armazena os embeddings no banco.

### 5. Conversar com o documento

```bash
python src/chat.py
```

Digite sua pergunta sobre o documento e receba respostas baseadas no conteúdo ingerido.
