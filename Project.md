# internal-perplexity

A great chat interface over a variety of information sources. Designed to be fast, accurate, and comprehensive.

## Idea:

### Internal perplexity

This is a chat interface that is a mock of perplexity. It will have access to databases, apis, internet, and personal directories. The crawler will create it's own graph database of the information it has access to. It will use this database to answer questions. The visual interface will be a chat, and will allow visualizations, tables, markdown, and other visualizations. I will try to make this hyper fast using a combination of caching, and parallelization. It should provide sources and when clicked they pull up the actual document or a box with a summary of that document and second link to the actual thing. Any code that is generated and executed will be logged and stored in a chat metadata so the user can use it.

- Initial input classification
  - Use a small model that is finetuned on the set of APIs we have access to. It'll essentially just identify which apis to use and the general direction of the call.
    - For example: Given - "what is the weather", 1B param model includes "api-weather: ..., api-google". It'll output "call weather: looking for weather data"
    - The output will have special sequence tokens designed to capture which api, and description of purpose of call.
    - The purpose of call will be used to tell the users what we are looking for as we search thus decreasing percieved time till first token.
    - The small model will require fine-tuning if oob performance isn't good enough even with many shot example learning.
- python packages for data extraction

  - fire crawling, textract, unstructured

- Agents

  - Citation method:
    - Indexes: scriptures, notes, and other gospel resources
    - citation format: "[scriptures: 3]" which links to actual source
    - Under the hood format: [index name: index number]
    - When the click it'll pull it up
    - If it's a link, it'll open it in a new tab
    - Provide all sources for each section ranked, and if they drag and rerank it'll update the ranking in the ML model backend that does the ranking (learnable parameters)
  - Agent list:
    - Query generator
      - Looks at history of queries, history of answers, and current query.
      - Generates query to each agent
        - Format: xml: `<query><index>scriptures</index><input>mosiah's speech about service</input><input>other query</input></query><response>response</response>`
        - Format option: json: `{"queries": {"the-index-name": ["query 1", "query 2"],...}, "response":  "response"}`
      - The queries are then parsed and sent ot the respective agents. If a response is provided then it is added to the context of the full response generation.
      - Prompt:
        - `Here are the available agents that can query the respective databases: [list of agents and the descriptions of their databases]. Take the formatted chat history and determine which if any agents we need to ask for more information. Answer in the following structured format {...}`
        - Need to restructure prompt so that we have a good workflow for the generator.
    - Response generator
      - Looks at history of queries, history of answers, and current query.
      - Generates response taking in all the context
    - Layer planner
      - similar to the response generator, but it takes all the context and determines if new queries need to be made.
      - If not, then it'll call the response generator, otherwise it'll call the query generator
      - If the user provides a layer number then it'll be told that it must iterate again and generate new queries `for i in range(layer):`
        - can have a max layer number and min layer number
    - Crawler
      - Crawler creates the graph
      - Predetermined relationships to look for
      - If it wants to add a new relationship then it'll put that on the list as potential. If it reccurs often then all the previous relationships mentioned will be added.
        - If it sees "X - related to Y: [opt1, opt2, ...optN]" then it'll add that to the list of relationships.
        - These are stored in a database and updated as the crawler suggests them
      - Starts with nodes for each entry in the database
      - Randomly selects groups of nodes and tries to determine relationships
    - Coder
      - If a question requires code then it'll call the coder
        - types of questions: Mathematics (counting), visualizations

- Configuration and Tool Descriptions
  - Document stores
    - Directory of files (.txt, .pdf, .docx, .md, .html)
    - Database (relational like postgres, graph like neo4j)
    - API (for searching internet or crawling a wiki)
    - Dynamically create new document stores that are completely separate (maybe local only) that a user can point to
      - like their own codebase.
  - Agents
    - Crawler
      - Point it to which stores
      - Enable own graph creation
      - Enable own vector store creation
      - Configure questions that it should look for (relationships, authors, titles, etc)
    - Classifier
      - Enable test time fine tuning
      - Point to stores
        - Enable many shot learning prompts generation
    - Reasoning planner
      - Default disabled
      - Takes initial responses from all queries
        - determines if questions need to be asked of the user, and if so, what questions
        - determines if more searches are needed
        - determines if the question has been answered
        - summarizes responses
    - Query generator
      - Takes input from classifier and reasoning planner
      - Generates queries to agents
    - Tools
      - Coder: generate simple python code to answer questions
      - Calculator: answer calculation questions
      - Database Queryer: generate and execute queries to databases
  - Interface
    - Enable saving chats or chat history
      - If not enabled, then it will be single query at a time
    - Enable multiple models for different agents
      - Can use different model for reasoning, query generator, coder and such
      - Can use APIs for models if not self hosted
      - Can use load balancer
    - Enable adding new database connections
    - Default instructions for user
      - Things like "present data in a table"
    - Enable visual renderings

## Roadmap for Internal Perplexity MVP Development

### Phase 1: Project Setup and Architecture Design (Weeks 1-2)

**Milestone: Project Foundation Established**

1. Define project scope and requirements
2. Set up version control system (e.g., Git repository)
3. Choose tech stack and frameworks
4. Design system architecture
5. Create detailed project documentation

**Infrastructure:**

- Version Control: GitHub
- Project Management: Jira or Trello
- Communication: Slack
- Cloud Platform: AWS or Google Cloud

**Team:**

- Project Manager
- Software Architect
- Full-stack Developer

### Phase 2: Core Components Development (Weeks 3-6)

**Milestone: Basic Chat Interface and Data Processing Pipeline**

1. Develop chat interface frontend
2. Implement backend API for chat functionality
3. Set up database systems (graph database and vector store)
4. Develop initial crawler for data extraction
5. Implement basic query classification system

**Infrastructure:**

- Frontend: React.js
- Backend: Python (FastAPI or Flask)
- Database: Neo4j (graph), PostgreSQL (relational), Pinecone (vector store)
- ML Framework: PyTorch or TensorFlow

**Team:**

- Frontend Developer
- Backend Developer
- ML Engineer
- Data Engineer

### Phase 3: Agent System Implementation (Weeks 7-10)

**Milestone: Functional Agent System**

1. Develop Query Generator agent
2. Implement Response Generator agent
3. Create Layer Planner agent
4. Integrate Crawler agent with data sources
5. Develop Coder agent for code generation and execution

**Infrastructure:**

- NLP Models: Hugging Face Transformers
- Code Execution: Docker containers for isolated environments

**Team:**

- ML Engineers (2-3)
- Backend Developers (1-2)

### Phase 4: Integration and Advanced Features (Weeks 11-14)

**Milestone: MVP with Core Functionality**

1. Integrate all agents into a cohesive system
2. Implement citation and source linking functionality
3. Develop visualization capabilities
4. Create user feedback and ranking system
5. Implement caching and parallelization for performance optimization

**Infrastructure:**

- Visualization Libraries: D3.js, Plotly
- Caching: Redis
- Parallel Processing: Apache Spark or Dask

**Team:**

- Full-stack Developers (2)
- ML Engineers (1-2)
- DevOps Engineer

### Phase 5: Testing and Refinement (Weeks 15-16)

**Milestone: Validated and Optimized MVP**

1. Conduct comprehensive system testing
2. Perform security audits and penetration testing
3. Optimize performance based on test results
4. Gather and incorporate initial user feedback
5. Refine UI/UX based on usability testing

**Infrastructure:**

- Testing Frameworks: Jest (frontend), Pytest (backend)
- Security Tools: OWASP ZAP, SonarQube

**Team:**

- QA Engineers (2)
- Security Specialist
- UX Designer

### Phase 6: Deployment and Launch Preparation (Weeks 17-18)

**Milestone: MVP Ready for Launch**

1. Set up production environment
2. Implement monitoring and logging systems
3. Create user documentation and help resources
4. Develop launch strategy and marketing materials
5. Conduct final end-to-end testing

**Infrastructure:**

- Deployment: Kubernetes
- Monitoring: Prometheus, Grafana
- Logging: ELK Stack (Elasticsearch, Logstash, Kibana)

**Team:**

- DevOps Engineers (2)
- Technical Writer
- Marketing Specialist

## Challenges and Mitigations

1. **Data Privacy and Security**

   - Challenge: Ensuring user data and queries are secure and compliant with regulations.
   - Mitigation: Implement end-to-end encryption, regular security audits, and strict data handling policies.

2. **Performance at Scale**

   - Challenge: Maintaining low latency as the user base and data volume grow.
   - Mitigation: Implement efficient caching strategies, optimize database queries, and use distributed computing for heavy processing tasks.

3. **Accuracy of AI Responses**

   - Challenge: Ensuring the AI generates accurate and relevant responses.
   - Mitigation: Implement robust testing and validation processes, continuous model fine-tuning, and user feedback mechanisms.

4. **Integration Complexity**

   - Challenge: Seamlessly integrating multiple data sources and AI models.
   - Mitigation: Design a modular architecture, use standardized APIs, and implement comprehensive integration testing.

5. **User Adoption**
   - Challenge: Encouraging users to switch from existing tools to Internal Perplexity.
   - Mitigation: Focus on unique value propositions, provide excellent onboarding experience, and gather and act on user feedback regularly.

## Post-Launch Maintenance and Iteration

1. Monitor system performance and user feedback
2. Continuously update and improve AI models
3. Add new data sources and expand knowledge base
4. Implement new features based on user requests
5. Regularly update security measures and conduct audits

By following this roadmap, the development team can create a robust MVP for Internal Perplexity, addressing key challenges and setting the stage for future growth and improvements.

Sources
