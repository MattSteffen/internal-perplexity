# TODO

- **TODAY**:

  - [x] Refactor converter to use pymupdf4llm for all examples
  - [ ] Update folder structure for mono repo with proper cicd, make files, testing, linting, deployment scripts, etc.
  - [ ] Update docs and openapi specs
  - [ ] update examples and make sure they run from scratch
  - [ ] Create server
    - [ ] openai server
    - [ ] crawler server
  - [ ] create frontend for crawler server
    - [ ] make sure docker containers work and test with backend docker containers too
  - [ ] Update RAG pipeline for BD docs too.

- [ ] Load real data into milvus

  - [ ] Test tool calling or structured output for gpt-oss
  - [ ] Get real data and test queries via script

- [ ] Crawler server
  - [ ] Auth
  - [ ] load milvus collections and metadata schema
  - [ ] create new collection
  - [ ] add docs to collection
- [ ] Milvus
  - [ ] Collections descriptions should have the metadata schema, this way when describing to LLM, it'll know what to look for and how to filter.
  - [ ] Create a package that can be imported and used by xmchat, radchat, etc.
  - [ ] Can I deploy the package to internal registry?
    - [ ] I should simplify the package to maybe something that can be reimplemented instead of installed. Maybe don't want them to have to have a package, but it is simple enough to interface that they can do it themselves.
