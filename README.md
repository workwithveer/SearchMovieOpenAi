## Search Movie with OpenAI & LangChain

This project is a small RAG-style helper that lets you search and explore movies from an IMDB-style dataset using modern AI tooling.  
It loads movie metadata from `IMDB.csv`, turns each row into a text document, and prepares it for use with LangChain and OpenAI models (for example, for semantic search or question answering about movies).

### Tech Stack

- **Python**: Core language for data processing and orchestration.
- **Pandas**: Used to load and transform the `IMDB.csv` file into a clean `DataFrame` (renaming columns, filtering to movies, handling missing descriptions, etc.).
- **LangChain & LangChain Community**:
  - **`DataFrameLoader`** from `langchain_community.document_loaders.dataframe` converts the Pandas `DataFrame` into a list of LangChain `Document` objects.
  - This structure is ideal for plugging into vector stores, retrievers, and LLM chains.
- **OpenAI / LangChain-OpenAI**: Intended LLM backend for semantic search, question answering, and other AI-powered interactions over the movie corpus.
- **DuckDB** (optional, from dependencies): Can be used for fast local SQL-style analytics over the dataset if you extend the project.
- **uv**: Modern Python package and environment manager used to install dependencies and run the project.

### What the Project Does (Use Cases)

Right now, the core script (`main.py`) focuses on preparing the data and documents. Once extended, you can use it for:

- **Semantic movie search**:  
  Ask natural language questions such as:
  - “Find sci‑fi movies with strong female leads.”
  - “Show me comedy movies released after 2010 set in high school.”
- **Recommendation-style queries**:  
  - “Recommend similar movies to Inception based on description and genre.”
  - “What are some dark thriller movies with psychological elements?”
- **Exploratory analysis over descriptions**:  
  - Summarize genres, detect themes, or cluster movies by description using LLMs.
- **RAG pipelines over IMDB-like data**:  
  Use the generated `docs` list as the document source for a LangChain RAG pipeline (vector store + retriever + LLM).

You can treat this project as a starting point to plug in:
- Vector stores (Chroma, FAISS, etc.).
- Chat-style interfaces (CLI, web UI, or chatbots).
- Advanced filters combining metadata (genres, year, etc.) with semantic similarity.

### Project Structure (Key Parts)

- `IMDB.csv`: Source dataset containing movie metadata (title, description, genres, etc.).
- `main.py`:
  - Loads `IMDB.csv` with Pandas.
  - Cleans and normalizes columns (e.g., `movie_title`, `movie_description`, `genres`, `source`).
  - Builds a `page_content` column combining title, genre, and description.
  - Uses `DataFrameLoader` to turn rows into LangChain `Document` objects.
  - Prints or returns the resulting documents (ready for retrieval / RAG).
- `pyproject.toml`: Defines the project metadata and dependencies managed via `uv`.

### Prerequisites

- **Python** `>= 3.13` (as specified in `pyproject.toml`).
- **uv** installed globally:

```bash
pip install uv
```

- (Optional but recommended) An **OpenAI API key** in a `.env` file (for when you add LLM calls):

```bash
OPENAI_API_KEY=your_api_key_here
```

### How to Set Up and Run the Project

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd SearchMovieOpenAi
```

2. **Create and sync the environment with uv**

This will create a virtual environment and install all dependencies from `pyproject.toml`:

```bash
uv sync
```

3. **(Optional) Activate the virtual environment**

If you prefer to run Python directly (instead of `uv run`), activate the venv:

```bash
.\.venv\Scripts\activate
```

4. **Run the main script**

Using `uv` (recommended):

```bash
uv run python main.py
```

Or, if your virtual environment is already activated:

```bash
python main.py
```

This will:
- Load and preprocess the movie dataset from `IMDB.csv`.
- Build a list of LangChain `Document` objects.
- Print the resulting documents (note: on Windows you may need to configure your terminal to handle Unicode output properly).

### Extending the Project

Here are some common next steps you can implement:

- **Add a vector store**:  
  - Embed `docs` using OpenAI embeddings and store them in a vector database (Chroma, FAISS, etc.).
  - Build a retriever to perform similarity search over movie descriptions.
- **Create a chat or CLI interface**:  
  - Implement a simple interface where you type a question about movies and get relevant titles and descriptions back.
- **Filter by metadata**:  
  - Combine semantic search with filters like genre, year, or rating for more precise results.
- **Expose as an API or web app**:  
  - Wrap your retrieval and LLM logic in a FastAPI/Flask backend and/or a simple frontend for interactive use.

### Troubleshooting

- **Import errors (e.g., pandas, langchain, langchain_community)**:
  - Ensure you ran `uv sync` in the project root.
  - Make sure your IDE is using the project’s virtual environment (`.venv`).
- **Unicode errors on Windows when printing docs**:
  - Use a UTF‑8 capable terminal, or avoid printing very large Unicode-heavy outputs.
- **OpenAI authentication issues**:
  - Confirm `OPENAI_API_KEY` is set in your environment or `.env` file.

With this foundation, you can quickly build rich, AI-powered movie search and exploration experiences on top of the IMDB-style dataset.

