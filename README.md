# Internal Perplexity

A robust and fast chat interface that integrates a variety of information sources, designed to be accurate, comprehensive, and user-friendly.

## Overview

Internal Perplexity is a chat interface inspired by Perplexity AI. It connects to databases, APIs, the internet, and personal directories, using a crawler to create a graph database for quick and efficient responses. Features include:

- Hyper-fast response times via caching and parallelization.
- A user-friendly chat interface with visualizations, tables, and markdown support.
- A crawler agent that works in the background learning and searchable making connections.
- Citations with source links and summaries.
- Logs and metadata for generated and executed code.

---

## Features

### Core Functionalities

1. **Initial Input Classification**:

   - A fine-tuned model classifies queries to determine the appropriate APIs and direction.
   - Example: Input "What is the weather?" triggers `api-weather` with a description of its purpose.
   - Special tokens capture the API and query intent, improving perceived response time.

2. **Python Packages for Data Extraction**:

   - **Fire Crawling**: Efficient data scraping.
   - **Textract**: Text extraction from documents.
   - **Unstructured**: Parsing and organizing raw data.

3. **Agents**:

   - **Query Generator**: Parses history and input to generate database-specific queries.
   - **Response Generator**: Consolidates context and query history to generate comprehensive responses.
   - **Layer Planner**: Plans new queries if needed, based on incomplete information.
   - **Crawler**: Creates a graph database with dynamic relationship mapping.
   - **Coder**: Generates and executes Python code for mathematical and visualization tasks.

4. **Citation Method**:

   - Generates and ranks sources for each response.
   - Users can re-rank sources, updating ML model parameters.

5. **Configuration Options**:

   - Database connections: Files, relational databases (PostgreSQL), graph databases (Neo4j), and APIs.
   - Caching, model-specific agents, and custom user instructions.

6. **Interface**:
   - Save chat history and responses.
   - Visualization support with D3.js and Plotly.
   - Multiple model support and easy database connection additions.

---

## Roadmap for MVP Development

### Phase 1: Project Setup and Architecture Design (Weeks 1-2)

- **Milestone**: Establish foundational systems and architecture.
- **Tasks**:
  - Define project requirements.
  - Choose tech stack (e.g., Python, Neo4j, React).
  - Design system architecture and documentation.

### Phase 2: Core Components Development (Weeks 3-6)

- **Milestone**: Build basic chat interface and data pipeline.
- **Tasks**:
  - Frontend and backend API for chat.
  - Set up databases (Neo4j, PostgreSQL, Pinecone).
  - Implement a basic query classification system.

### Phase 3: Agent System Implementation (Weeks 7-10)

- **Milestone**: Develop agents for core functionalities.
- **Tasks**:
  - Implement query, response, and layer planner agents.
  - Integrate the crawler with data sources.
  - Develop code execution with isolated environments (e.g., Docker).

### Phase 4: Integration and Advanced Features (Weeks 11-14)

- **Milestone**: Fully functional MVP.
- **Tasks**:
  - Integrate agents and develop caching for performance.
  - Add visualization and citation systems.
  - Implement user feedback for source ranking.

### Phase 5: Testing and Refinement (Weeks 15-16)

- **Milestone**: Validate and optimize the system.
- **Tasks**:
  - Perform system testing and security audits.
  - Refine performance and UX based on feedback.

### Phase 6: Deployment and Launch Preparation (Weeks 17-18)

- **Milestone**: Prepare MVP for production.
- **Tasks**:
  - Set up a production environment (Kubernetes).
  - Implement monitoring (Prometheus) and logging (ELK Stack).
  - Create user documentation and marketing materials.

---

## Challenges and Mitigations

1. **Data Privacy and Security**:

   - Implement end-to-end encryption and strict data handling policies.

2. **Performance at Scale**:

   - Use caching, model load-balancing, and distributed processing (Spark/Dask).

3. **Accuracy of AI Responses**:

   - Fine-tune models and integrate robust validation.
   - Implement a feedback loop for model improvement.
   - Track model function calling failures and improve them.

4. **Integration Complexity**:

   - Employ comprehensive testing.

5. **User Adoption**:
   - Highlight unique features, gather feedback, and provide excellent onboarding.

---

## MVP

### Goal:

- Aspects of RAG that are hard to do well and fast:
  - Integrate and synthesize information from multiple diverse sources (databases, APIs, internet, personal directories)
  - Provide accurate, contextual responses with proper citations and sources
  - Deliver fast responses through intelligent caching and parallelization
  - Create and maintain dynamic knowledge connections through background crawling and graph database relationships
  - Allow for multi-step reasoning for deriving complicated answers from multiple sources
- This project is for:
  - Teams needing quick access to distributed information sources
  - Users who need both factual answers and computational/analytical capabilities
    - This is especially important when multiple different search apis need to be accessed
- Unique aspects:
  - Dynamic knowledge graph that improves over time through background learning
    - The crawler agent will operate in the background, learning and making connections.
  - Adding in multiple search apis and being able to synthesize information from them
  - EASY DEPLOYMENT of a highly complex system
  - Integration multi-step reasoning for deriving complicated answers from multiple sources

### Initial query

- A model classifies queries to determine the appropriate APIs and direction.

### Data Processing

- Point to directory of documents
- Crawler creates index and graph database

### RAG

- Query generator: Parses history and input to generate database-specific queries.
- Response generator: Generates responses based on context.
  - Provides citations as sources and summaries for each section.

## Beta

### Initial query

- A small fine-tuned model classifies queries to determine the appropriate APIs and direction.
- Hints are provided to UI to reduce perceived response time.

### Data Processing

- Crawler can index over multiple directories, many file types, perperly understanding each file including images
- Crawler can search wiki

### RAG

- Layer planner: Determines if new queries are needed based on context.
- Crawler agent: Learns and makes connections in the background.
- Crawler can index on search apis
- Improved response generator: Generates responses based on context.

---
