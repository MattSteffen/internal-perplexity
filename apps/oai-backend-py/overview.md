## TODO:
(Next steps)

- Auth:
  - **Need to integrate at work to debug and test, can't mock easily**
    - Require auth (keycloak)
    - Tools that require auth (milvus search) have it passed to them correctly
        - This means that /chat/completions needs to have the credentials passed as well
- collections/
    - user credentials are used in creating the client and listing the collections
    - *Note: A user should be able to list collections and see which ones are there but not read / query them*
- Settings:
    - Milvus search (not for /collections) has the proper admin credentials passed to it on server start up (different from user credentials)
    - Milvus host:port uri from some sort of setup config
    - Ollama base url from setup config
- [x] Docker
    - Add the `pip install -e ../crawler` package
- collections/{}/add
    - Implement, should be able to add, using the crawler package, new pdfs to collections
        - collections they can't query, but can add to
        - enforce security correctly
- Deploy with docker and make sure it works correctly
    - Tests to make sure everything works:
        - Add integration tests that do call the LLM
        - Add demo tests that showcase the usage like creating a new collection, uploading a document, chatting with that document, enforcing security.
- Integrate with OI
    - See if it can pass auth through
    - Confirm streaming works
    - *Later*: Integrate their agent api to add tool calls
    - *Later*: Confirm reasoning streams also work as they should.
- Build Front end to upload docs.


## **Core Requirements**

1. **Language & Framework**
   - Fully implemented in **Python 3.14**
   - Use **FastAPI** for REST and SSE support
   - Strict typing enforced with **Pydantic (v2)** models

2. **Authentication**
   - Integrate with **Keycloak** or **GitLab OAuth** for auth and user management.
   - JWT-based securing of endpoints.
   - Access control and permission management for data upload and collection access.

3. **Endpoints**
   - `POST /v1/chat/completions` (OpenAI-compatible)
     - Streaming response using Server-Sent Events or WebSockets
     - Supports function/tool calling in OpenAI schema
       - Internal tools (like RAG)
       - External tool calls (LLM return message is a tool call message)
   - `POST /v1/embeddings`
   - `POST /v1/tools`
     - Direct tool calling endpoint
     - Accepts OpenAI-style tool calling format with `name`, `arguments`, and optional `metadata`
     - Metadata can contain user-specific data required for tool initialization
   - `GET /v1/collections`
     - View metadata/descriptions of collections
   - `POST /v1/collections/new`
     - Create a new Milvus collection
   - `POST /v1/collections/{collection_id}/add`
     - Add documents to a collection

4. **Internal Modules**
   - **Authentication module:** Handles Keycloak/GitLab integration and user identity.
   - **Chat engine module:** Routes chat completions to local models or external LLMs.
   - **Tool engine:** Handles both native tools and dynamic tool calls returned by the LLM.
   - **Data module:** Manages file uploads, document ingestion, and metadata. Document upload (file or pre-processed markdown) always goes through the crawler's `Crawler.crawl_document`; the crawler skips convert and extract when markdown/metadata are already set on the document.
   - **Streaming controller:** Manages server-sent events for chat streaming.
*As much as possible, I want to avoid creating my own types and objects, but using packages like openai or litellm. As they should be able to handle things like streaming for me. All LLM calls will also be sent through them, so I am almost just a proxy*

5. **Milvus Integration**
   - outline for implementation of endpoints, no RAG yet.

6. **Type Safety**
   - All endpoints and internal logic use **Pydantic BaseModels**.
   - Consistent validation and explicit schema definitions.

7. **Makefile**
   - Include targets:
     - `lint`: run `ruff` or `flake8` + `black` + `mypy`
     - `test`: run `pytest`
     - `run`: start the FastAPI server
     - `build`: optional Docker build step

8. **Output Format**
   - Provide:
     1. **High-level architecture overview**
     2. **Directory structure proposal**
     3. **Module/interface outlines with key class/method sketches**
     4. **Sensible design rationales for the chosen structures**
     5. **Dependency graph (textual)**
     6. **Example FastAPI endpoint skeletons (typed)**
     7. **Makefile content**
     8. **Next steps for scaling, testing, and deploying the server**
