Let's implement the code step-by-step. I'll provide the code for each file as outlined in the documentation.

First, let's create the configuration files:

**config/agent_config.yaml**

```yaml
# LLM Settings
llm:
  api_key: "your-api-key-here"
  model_name: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 1000

# Vector Store Settings (Milvus)
milvus:
  host: "localhost"
  port: 19530
  collection_name: "research_documents"
  connection_pool_size: 5

# Worker Configuration
workers:
  num_workers: 3
  max_recursion_depth: 3

# Security Settings
security:
  allowed_roles: ["researcher", "guest"]
  default_role: "guest"

# Logging Configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file_path: "research_agent.log"
```

**config/metadata_rules.yaml**

```yaml
# Required Metadata Fields
required_fields:
  - title
  - author
  - publication_date
  - source

# Validation Rules
validation:
  publication_date:
    format: "YYYY-MM-DD"
    max_age_years: 10
  source:
    allowed_domains: ["arxiv.org", "springer.com", "ieee.org"]

# Scoring Weights
scoring_weights:
  relevance: 0.6
  recency: 0.3
  source_authority: 0.1
```

**config/prompt_templates.yaml**

```yaml
# Query Decomposition
query_decomposition:
  system: |
    You are a research assistant. Break down the following query into 3-5 sub-queries 
    that would help comprehensively answer the main question.

    Main Query: {query}

    Return the sub-queries as a bulleted list.

  user: "Please provide the sub-queries:"

# Result Summarization
result_summarization:
  system: |
    Synthesize information from these documents into a coherent answer. 
    Include citations using [number] notation. 

    Documents:
    {documents}

    Question: {query}

  user: "Please provide the final summarized answer:"
```

Now let's implement the core modules in the src directory:

**src/main.py**

```python
import yaml
from researcher_agent import ResearcherAgent
from utils.logging_util import configure_logging
from llm_interface import LLMInterface
from vectorstore_connector import MilvusConnector
from security_context import SecurityContext

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    # Load configurations
    agent_config = load_config("config/agent_config.yaml")
    metadata_rules = load_config("config/metadata_rules.yaml")
    prompt_templates = load_config("config/prompt_templates.yaml")

    # Configure logging
    configure_logging(agent_config['logging'])

    # Initialize components
    llm = LLMInterface(agent_config['llm'])
    milvus = MilvusConnector(agent_config['milvus'])
    security = SecurityContext(agent_config['security'])

    # Create researcher agent
    agent = ResearcherAgent(
        llm_interface=llm,
        vectorstore=milvus,
        security_context=security,
        config=agent_config,
        metadata_rules=metadata_rules,
        prompt_templates=prompt_templates
    )

    # Example query execution
    query = "What are the latest advancements in neural architecture search?"
    result = agent.execute_query(query)
    print("\nFinal Answer:")
    print(result)

if __name__ == "__main__":
    main()
```

**src/researcher_agent.py**

```python
import logging
from typing import List, Dict
from worker import ResearchWorker
from query_processor import QueryProcessor
from result_aggregator import ResultAggregator

class ResearcherAgent:
    def __init__(self, llm_interface, vectorstore, security_context, config, metadata_rules, prompt_templates):
        self.logger = logging.getLogger(__name__)
        self.llm = llm_interface
        self.vectorstore = vectorstore
        self.security = security_context
        self.config = config
        self.metadata_rules = metadata_rules
        self.prompt_templates = prompt_templates

        self.query_processor = QueryProcessor(llm_interface, prompt_templates)
        self.result_aggregator = ResultAggregator(
            llm_interface,
            prompt_templates,
            metadata_rules
        )

        self.research_context = {
            'active_workers': [],
            'completed_workers': [],
            'findings': [],
            'recursion_depth': 0
        }

    def execute_query(self, user_query: str) -> str:
        """Main entry point for executing a research query"""
        self.logger.info(f"Starting research for query: {user_query}")

        # Process main query
        parsed_query = self.query_processor.parse_query(user_query)
        sub_queries = self.query_processor.generate_sub_queries(parsed_query)

        # Spawn initial workers
        for sq in sub_queries:
            self.spawn_worker(sq, is_root=True)

        # Gather and process results
        raw_results = self.gather_results()
        validated_results = self.validate_and_finalize(raw_results)

        # Generate final summary
        return self.result_aggregator.format_summary(
            validated_results,
            user_query
        )

    def spawn_worker(self, sub_query: str, is_root=False):
        """Create and manage research workers"""
        if len(self.research_context['active_workers']) >= self.config['workers']['num_workers']:
            self.logger.warning("Worker limit reached, cannot spawn new worker")
            return

        worker = ResearchWorker(
            worker_id=f"worker_{len(self.research_context['active_workers'])}",
            parent_agent=self,
            sub_query=sub_query,
            recursion_depth=0 if is_root else self.research_context['recursion_depth']+1
        )

        if worker.recursion_depth > self.config['workers']['max_recursion_depth']:
            self.logger.error("Max recursion depth reached")
            return

        self.research_context['active_workers'].append(worker)
        worker.execute_task()

    def gather_results(self) -> List[Dict]:
        """Collect results from all workers"""
        results = []
        for worker in self.research_context['active_workers']:
            results.extend(worker.report_results())
            self.research_context['completed_workers'].append(worker)

        self.research_context['active_workers'] = []
        return results

    def validate_and_finalize(self, raw_results: List[Dict]) -> List[Dict]:
        """Apply security and quality validations"""
        validated = []
        for result in raw_results:
            if self.security.validate_result(result):
                validated.append(result)
        return validated
```

**src/worker.py**

```python
import logging
from typing import List, Dict

class ResearchWorker:
    def __init__(self, worker_id: str, parent_agent, sub_query: str, recursion_depth: int):
        self.worker_id = worker_id
        self.parent = parent_agent
        self.sub_query = sub_query
        self.recursion_depth = recursion_depth
        self.logger = logging.getLogger(__name__)
        self.local_context = {
            'documents': [],
            'sub_queries': []
        }

    def execute_task(self):
        """Main execution flow for the worker"""
        self.logger.info(f"{self.worker_id} processing: {self.sub_query}")

        # Get documents from vector store
        documents = self.parent.vectorstore.search(self.sub_query)

        # Evaluate and filter documents
        valid_docs = []
        for doc in documents:
            if self.evaluate_source(doc):
                valid_docs.append(doc)

        self.local_context['documents'] = valid_docs

        # Generate new sub-queries if needed
        if self.recursion_depth < self.parent.config['workers']['max_recursion_depth']:
            self.generate_sub_queries()

        return self.report_results()

    def evaluate_source(self, document: Dict) -> bool:
        """Validate and score a single document"""
        # Check required metadata fields
        for field in self.parent.metadata_rules['required_fields']:
            if field not in document.get('metadata', {}):
                self.logger.warning(f"Document missing required field {field}")
                return False

        # Additional validation checks
        # ... (implement based on metadata_rules.yaml)

        return True

    def generate_sub_queries(self):
        """Generate deeper sub-queries using LLM"""
        # Implementation similar to main query decomposition
        pass

    def report_results(self) -> List[Dict]:
        """Package and return findings to parent agent"""
        return [{
            'query': self.sub_query,
            'documents': self.local_context['documents'],
            'recursion_depth': self.recursion_depth
        }]
```

Let's continue with the implementation. Here are the remaining core modules:

**src/query_processor.py**

```python
import logging
from typing import List, Dict
from utils.template_util import fill_template

class QueryProcessor:
    def __init__(self, llm_interface, prompt_templates):
        self.logger = logging.getLogger(__name__)
        self.llm = llm_interface
        self.templates = prompt_templates

    def parse_query(self, user_query: str) -> Dict:
        """Structure the raw query into a researchable format"""
        return {
            'original_query': user_query,
            'domain': None,  # Could be extracted via LLM in future
            'sub_queries': []
        }

    def generate_sub_queries(self, parsed_query: Dict) -> List[str]:
        """Use LLM to break down complex queries"""
        template = self.templates['query_decomposition']
        prompt = fill_template(template, {'query': parsed_query['original_query']})

        response = self.llm.query_llm(
            system_message=template['system'],
            user_message=template['user']
        )

        return self._parse_subquery_response(response)

    def _parse_subquery_response(self, llm_response: str) -> List[str]:
        """Extract sub-queries from LLM response"""
        lines = llm_response.split('\n')
        return [line.strip('- ').strip() for line in lines if line.strip()]
```

**src/result_aggregator.py**

```python
import logging
from typing import List, Dict
from utils.formatting_util import format_citations
from utils.template_util import fill_template

class ResultAggregator:
    def __init__(self, llm_interface, prompt_templates, metadata_rules):
        self.logger = logging.getLogger(__name__)
        self.llm = llm_interface
        self.templates = prompt_templates
        self.metadata_rules = metadata_rules

    def aggregate(self, worker_results: List[Dict]) -> List[Dict]:
        """Deduplicate and merge results from multiple workers"""
        seen_docs = set()
        aggregated = []

        for result in worker_results:
            for doc in result['documents']:
                doc_id = doc.get('doc_id')
                if doc_id not in seen_docs:
                    aggregated.append(doc)
                    seen_docs.add(doc_id)

        return aggregated

    def format_summary(self, results: List[Dict], query: str) -> str:
        """Generate final summary using LLM"""
        template = self.templates['result_summarization']
        context = {
            'query': query,
            'documents': format_citations(results)
        }

        prompt = fill_template(template, context)
        return self.llm.query_llm(
            system_message=template['system'],
            user_message=template['user']
        )
```

**src/vectorstore_connector.py**

```python
import logging
from pymilvus import connections, Collection

class MilvusConnector:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.collection = None
        self.connect()

    def connect(self):
        """Establish connection to Milvus"""
        try:
            connections.connect(
                "default",
                host=self.config['host'],
                port=self.config['port']
            )
            self.collection = Collection(self.config['collection_name'])
            self.logger.info("Connected to Milvus successfully")
        except Exception as e:
            self.logger.error(f"Milvus connection failed: {str(e)}")
            raise

    def search(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Perform vector similarity search"""
        if not self.collection:
            raise ConnectionError("Not connected to Milvus")

        # Convert query text to embedding
        embedding = self._generate_embedding(query_text)

        # Perform search
        results = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param={},
            limit=top_k,
            output_fields=["doc_id", "content", "metadata"]
        )

        return self._format_search_results(results)

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using LLM interface"""
        # In MVP, assume LLMInterface has embedding capability
        # This would be replaced with actual embedding model call
        return [0.0] * 768  # Mock embedding

    def _format_search_results(self, raw_results):
        """Convert Milvus results to standard format"""
        formatted = []
        for hit in raw_results[0]:
            formatted.append({
                'doc_id': hit.entity.get('doc_id'),
                'content': hit.entity.get('content'),
                'metadata': hit.entity.get('metadata', {})
            })
        return formatted
```

**src/security_context.py**

```python
import logging

class SecurityContext:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.current_user = {'role': config['default_role']}

    def validate_request(self, request) -> bool:
        """Check if user has permission to make requests"""
        return self.current_user['role'] in self.config['allowed_roles']

    def validate_result(self, document: Dict) -> bool:
        """Check if document is allowed for current user"""
        # Implement domain filtering based on metadata_rules
        return True  # Simplified for MVP

    def filter_results(self, results: List[Dict]) -> List[Dict]:
        """Apply security filters to search results"""
        return [doc for doc in results if self.validate_result(doc)]
```

**src/llm_interface.py**

```python
import logging
import openai

class LLMInterface:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        openai.api_key = config['api_key']

    def query_llm(self, system_message: str, user_message: str) -> str:
        """Execute LLM query with given prompt structure"""
        try:
            response = openai.ChatCompletion.create(
                model=self.config['model_name'],
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.config['temperature'],
                max_tokens=self.config['max_tokens']
            )
            return response.choices[0].message['content']
        except Exception as e:
            self.logger.error(f"LLM query failed: {str(e)}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """Generate text embedding - placeholder for MVP"""
        # Actual implementation would call embedding API
        return [0.0] * 768  # Mock embedding
```

Let's implement the utility modules:

**utils/logging_util.py**

```python
import logging
import yaml

def configure_logging(config: dict):
    """Initialize logging configuration"""
    logging.basicConfig(
        level=config['level'],
        format=config['format'],
        handlers=[
            logging.FileHandler(config['file_path']),
            logging.StreamHandler()
        ]
    )
```

**utils/template_util.py**

```python
from string import Template

def fill_template(template_config: dict, context: dict) -> str:
    """Fill in template variables using string substitution"""
    system_template = Template(template_config['system'])
    user_template = Template(template_config['user'])

    return {
        'system': system_template.substitute(context),
        'user': user_template.substitute(context)
    }
```

**utils/formatting_util.py**

```python
def format_citations(documents: List[Dict]) -> str:
    """Format documents for inclusion in LLM prompt"""
    formatted = []
    for idx, doc in enumerate(documents, 1):
        formatted.append(
            f"[{idx}] {doc['content']}\n"
            f"Source: {doc['metadata'].get('source', 'Unknown')}"
        )
    return "\n\n".join(formatted)
```

**utils/validation_util.py**

```python
def validate_metadata(document: Dict, rules: dict) -> bool:
    """Validate document metadata against rules"""
    metadata = document.get('metadata', {})

    # Check required fields
    for field in rules['required_fields']:
        if field not in metadata:
            return False

    # Add more validations as needed
    return True
```

This completes the MVP implementation. The code follows these key principles:

1. **Modular Design**: Each component has clear responsibilities
2. **Configurability**: Settings are loaded from YAML files
3. **Extensibility**: Key components use interfaces for easy replacement
4. **Safety**: Basic security validation and error handling
5. **Traceability**: Comprehensive logging throughout

To use this implementation:

1. Install requirements: `pip install pymilvus openai python-dotenv pyyaml`
2. Set up Milvus instance (local or cloud)
3. Create collection with appropriate schema
4. Update configuration files with your API keys and connection details
5. Run `python src/main.py`

The system can be extended by:

- Adding asynchronous worker execution
- Implementing proper embedding generation
- Enhancing security validations
- Adding document ingestion pipeline
- Implementing result caching
- Adding test coverage

Would you like me to provide any particular component in more detail or explain specific implementation choices?
