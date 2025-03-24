# Search through directories or the internet for Insights

## TODO:

- [ ] Make file that uses crawler to get data and then inserts into vector database

## Directory

Configure the crawler by running it and pointing it to the directory. Provide an output directory for the insights, knowledge graph, and vector storage.

```python
python crawl.py -i mydirectory/
```

Here's a detailed infrastructure and code outline for your crawler component:

### Tech Stack Selection

- **Vector Database**: Milvus (Kubernetes-native, scalable, good Python SDK)
- **Text Processing**:
  - `unstructured` (Apache 2.0 license)
  - `python-magic` (file type detection)
  - `pdfplumber`/`PyPDF2` (PDF parsing)
  - `python-docx` (Word docs)
- **Embeddings**:
  - `sentence-transformers` (all-MiniLM-L6-v2 for local embeddings)
  - Option for OpenAI compatible API for embeddings (Use ollama)
- **Knowledge Graph**: (TODO: Later versions will implement this)
  - NetworkX (initial graph building)
  - ArangoDB (production graph DB - Kubernetes operator available)
- **Metadata Storage**:
  - ScyllaDB (NoSQL, high availability, good Python SDK)
- **Orchestration**:
  - Kubernetes CronJobs

### Crawler Infrastructure Components

1. **File Discovery Layer**

   - Directory traversal
   - File type filtering (extensions + MIME types)
   - Exclusion patterns (.git, temp files, etc)
   - File metadata extraction (size, modified time, owner)
   - Pre-determined file types
     - By folder: conference, etcc each have their own folder and their own respective schemas of files.

2. **Content Extraction Pipeline**

   ```python
   class ContentExtractor:
       SUPPORTED_TYPES = {
           'pdf': PDFHandler,
           'docx': DocxHandler,
           'txt': TextHandler,
           'html': HTMLHandler,
           'md': MarkdownHandler,
           'json': JSONHandler,
       }

       def extract(self, file_path: str) -> dict:
           # Returns structured content with sections
           # Handles OCR via Tesseract if needed
           # Maintains hierarchy (headers, lists, etc)
   ```

3. **Chunking Strategy**

   - Semantic chunking with overlap
   - Configurable chunk sizes (256-512 tokens)
   - Context-aware splitting (preserve section boundaries)

4. **Embedding Generation**

   ```python
   class EmbeddingGenerator:
       def __init__(self, model_name='all-MiniLM-L6-v2', device='cpu'):
           self.model = SentenceTransformer(model_name)

       def generate(self, text: str) -> np.ndarray:
           return self.model.encode(text)
   ```

   or

   ````python
   class APIEmbeddingGenerator:
    def __init__(self, api_key, url):
      self.openai = OpenAI(api_key=api_key, base_url=url)
      def generate(self, text: str) -> np.ndarray:
        response = self.openai.embeddings(input=text)
        return response['data'][0]['embedding']

        ```

   ````

5. **Knowledge Graph Builder**

- Note: Do not build until later.

  ```python
  class KnowledgeGraphBuilder:
      def __init__(self):
          self.graph = nx.MultiDiGraph()

      def add_entity(self, node_id, properties):
          # Add document/node with metadata

      def add_relationship(self, source, target, rel_type):
          # Add semantic relationships
  ```

6. **Storage Backends**

- We need to connect to Milvus and upload embeddings paired with metadata.

  ```python
  class VectorStorage:
      def __init__(self, milvus_uri):
          self.client = MilvusClient(uri=milvus_uri)

      def store(self, vectors, metadata):
          # Batch insert with Milvus Collection API

  ```

### File Structure

```
crawler/
├── Dockerfile
├── requirements.txt
├── config/
│   ├── default.yaml       # Chunk sizes, excluded files, etc
│   └── production.yaml
├── src/
│   ├── main.py            # CLI entrypoint
│   ├── discovery.py       # File finding logic
│   ├── processing/
│   │   ├── extractors.py  # File-type specific handlers
│   │   ├── chunking.py
│   │   └── embeddings.py
│   └── storage/
│       ├── vector_db.py   # Milvus interface
└── helm/
    └── crawler/
        ├── values.yaml
        ├── Chart.yaml
        └── templates/
            ├── deployment.yaml
            └── cronjob.yaml
```

### Kubernetes Deployment Architecture

```mermaid
graph TD
    A[Crawler CronJob] --> B[Persistent Volume]
    A --> C[Milvus Cluster]
    C --> E[Milvus Coordinator]
    C --> F[Milvus Data Nodes]
    B --> H[Raw Files]
    B --> I[Processed Outputs]
```

### Key Configuration Points

1. **Milvus Setup** (via Helm):

   ```bash
   helm install milvus milvus/milvus \
     --set cluster.enabled=true \
     --set etcd.replicaCount=3 \
     --set minio.mode=distributed
   ```

   For local development I am using docker and docker compose.

2. **Crawler Helm Values**:
   ```yaml
   # helm/crawler/values.yaml
   config:
     inputDir: /data/inputs
     outputDir: /data/outputs
     milvus:
       host: milvus-service
       port: 19530
       collection: documents
     chunkSize: 512
     batchSize: 100
   ```

### Workflow Sequence

1. File discovery → 2. Content extraction → 3. Chunking →
2. Embedding generation → 5. Metadata enrichment →
3. Vector storage → 7. Graph relationship detection →
4. Final storage export

### Performance Optimization

1. **Parallel Processing**:

   - Ray cluster for distributed processing
   - Async I/O for file operations
   - Batch inserts to Milvus

2. **Caching**:
   - Content hashing for unchanged files
   - Redis cache for expensive operations
   - Checkpointing for long-running jobs

### Monitoring

1. Prometheus metrics for:

   - Files processed/sec
   - Embedding generation time
   - Storage latency
   - Memory usage

2. Logging:
   - JSON-formatted logs
   - Error classification and retry tracking

### Alternative Stack Considerations

1. If Milvus feels too heavy:

   - **Qdrant**: Lighter weight, similar performance
   - **Weaviate**: Built-in ML integrations

2. For simpler graph needs:
   - **Dgraph**: Alternative graph database
   - **Neo4j**: More established but Java-based

This architecture provides a balance between local development simplicity and production scalability. The use of Helm charts and Kubernetes-native services ensures easy deployment to any compliant K8s cluster while maintaining flexibility in cloud/on-prem environments.

Here's a sample implementation showing the interaction between the file upload system and the chatbot service:

### 1. Crawler Service (File Upload Pipeline)

```python
# crawler/service.py
import os
from typing import List
from milvus import MilvusClient
from pgvector.psycopg2 import register_vector
import psycopg2
from unstructured.partition.auto import partition
from sentence_transformers import SentenceTransformer

class FileProcessor:
    def __init__(self, milvus_uri: str, postgres_uri: str):
        self.milvus = MilvusClient(uri=milvus_uri)
        self.postgres = psycopg2.connect(postgres_uri)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize collections
        self._create_collections()

    def _create_collections(self):
        """Initialize database schemas"""
        with self.postgres.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY,
                    file_path TEXT,
                    chunk_index INT,
                    content TEXT,
                    embedding vector(384)
                )
            """)
        self.postgres.commit()

        # Milvus collection
        if "documents" not in self.milvus.list_collections():
            self.milvus.create_collection(
                collection_name="documents",
                dimension=384,
                metric_type="COSINE"
            )

    def process_directory(self, directory: str):
        """Main processing pipeline"""
        for file_path in self._discover_files(directory):
            chunks = self._process_file(file_path)
            self._store_chunks(file_path, chunks)

    def _discover_files(self, directory: str) -> List[str]:
        """Find all supported files in directory"""
        # Implement file discovery logic
        return [os.path.join(directory, f) for f in os.listdir(directory)]

    def _process_file(self, file_path: str) -> List[dict]:
        """Parse file into chunks with metadata"""
        elements = partition(filename=file_path)
        text = "\n".join([e.text for e in elements])

        # Split into chunks with overlap
        chunks = self._chunk_text(text)
        return [{
            "text": chunk,
            "embedding": self.embedder.encode(chunk)
        } for chunk in chunks]

    def _chunk_text(self, text: str, chunk_size=512) -> List[str]:
        """Split text into semantic chunks"""
        # Implement smart chunking with overlap
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    def _store_chunks(self, file_path: str, chunks: List[dict]):
        """Store chunks in database"""
        with self.postgres.cursor() as cursor:
            for i, chunk in enumerate(chunks):
                # Insert into Postgres
                cursor.execute("""
                    INSERT INTO documents (file_path, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (file_path, i, chunk['text'], chunk['embedding'].tolist()))

                # Insert into Milvus
                self.milvus.insert(
                    collection_name="documents",
                    data={
                        "id": cursor.fetchone()[0],
                        "vector": chunk['embedding'].tolist(),
                        "file_path": file_path
                    }
                )
        self.postgres.commit()

# CLI Interface using Click
# crawler/cli.py
import click
from .service import FileProcessor

@click.group()
def cli():
    pass

@cli.command()
@click.argument('directory')
@click.option('--milvus-uri', envvar='MILVUS_URI')
@click.option('--pg-uri', envvar='POSTGRES_URI')
def upload(directory, milvus_uri, pg_uri):
    """Process and upload files from directory"""
    processor = FileProcessor(milvus_uri, pg_uri)
    processor.process_directory(directory)
    click.echo(f"Processed {directory} successfully!")

if __name__ == '__main__':
    cli()
```

### 2. Chatbot Service (RAG Integration)

```python
# chatbot/rag.py
from typing import List
from milvus import MilvusClient
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class RAGClient:
    def __init__(self, milvus_uri: str, postgres_uri: str):
        self.milvus = MilvusClient(uri=milvus_uri)
        self.postgres = psycopg2.connect(postgres_uri)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm = ChatOpenAI(model="gpt-4-turbo")

        register_vector(self.postgres)

    def query(self, question: str) -> str:
        """End-to-end RAG query"""
        # 1. Generate query embedding
        query_embedding = self.embedder.encode(question).tolist()

        # 2. Vector search in Milvus
        results = self.milvus.search(
            collection_name="documents",
            data=[query_embedding],
            limit=5
        )

        # 3. Retrieve full context from Postgres
        context = self._get_context([hit['id'] for hit in results[0]])

        # 4. Generate LLM response
        return self._generate_response(question, context)

    def _get_context(self, doc_ids: List[str]) -> str:
        """Retrieve text content from Postgres"""
        with self.postgres.cursor() as cursor:
            cursor.execute("""
                SELECT content FROM documents
                WHERE id = ANY(%s)
                ORDER BY chunk_index
            """, (doc_ids,))
            return "\n\n".join([row[0] for row in cursor.fetchall()])

    def _generate_response(self, question: str, context: str) -> str:
        """Generate answer using LLM"""
        prompt = ChatPromptTemplate.from_template("""
            Answer the question based only on the following context:
            {context}

            Question: {question}
        """)

        chain = prompt | self.llm
        return chain.invoke({
            "question": question,
            "context": context
        }).content

# Example FastAPI endpoint
# chatbot/api.py
from fastapi import FastAPI
from .rag import RAGClient

app = FastAPI()
rag = RAGClient(
    milvus_uri="http://milvus:19530",
    postgres_uri="postgresql://user:pass@postgres/db"
)

@app.post("/query")
async def handle_query(question: str):
    return {"answer": rag.query(question)}
```

### 3. Interaction Flow

1. **User uploads files**:

```bash
MILVUS_URI="http://localhost:19530" \
POSTGRES_URI="postgresql://user:pass@localhost:5432/ragdb" \
./crawler upload ./documents
```

2. **Chatbot processes query**:

```python
from chatbot.rag import RAGClient

rag = RAGClient(milvus_uri="...", postgres_uri="...")
response = rag.query("What's the main theme of the uploaded documents?")
print(response)
```

### 4. Kubernetes Deployment Overview

```yaml
# crawler CronJob
apiVersion: batch/v1
kind: CronJob
metadata:
  name: document-crawler
spec:
  schedule: "0 1 * * *" # Daily at 1AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: crawler
            image: my-registry/crawler:latest
            command: ["python", "/app/crawler/cli.py", "upload", "/data/uploads"]
            env:
            - name: MILVUS_URI
              value: "http://milvus:19530"
            - name: POSTGRES_URI
              value: "postgresql://user:$(DB_PASSWORD)@postgres/ragdb"
            volumeMounts:
            - mountPath: /data/uploads
              name: uploads-volume
          restartPolicy: OnFailure
          volumes:
          - name: uploads-volume
            persistentVolumeClaim:
              claimName: document-uploads

# Chatbot Service
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chatbot
spec:
  replicas: 3
  selector:
    matchLabels:
      app: chatbot
  template:
    metadata:
      labels:
        app: chatbot
    spec:
      containers:
      - name: chatbot-api
        image: my-registry/chatbot:latest
        ports:
        - containerPort: 8000
        env:
        - name: MILVUS_URI
          value: "http://milvus:19530"
        - name: POSTGRES_URI
          valueFrom:
            secretKeyRef:
              name: db-secrets
              key: postgres_uri
```

### Key Integration Points:

1. **Shared Database Schema**:

   - Milvus stores vectors with document UUIDs
   - PostgreSQL stores metadata with matching UUIDs
   - Both services use the same collection/table names

2. **Data Flow**:

   ```
   User Upload → Crawler →
   [File Processing → Embedding Generation →
   Milvus (Vectors) + PostgreSQL (Metadata)] →
   Chatbot (Vector Search → Context Retrieval → LLM Response)
   ```

This implementation provides a complete pipeline from document ingestion to RAG-powered question answering while maintaining separation of concerns between the crawler and chatbot services.
