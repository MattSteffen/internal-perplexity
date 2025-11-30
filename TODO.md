# TODO

- **Tasks**:
  - [ ] Document Pipelines:
    - [ ] Should have an implementation of crawler for each pipeline, so you can call pipeline.crawl(document) to process the document.
    - [ ] Default pipeline
  - [ ] Make sure collection description is set correctly by backend and crawler (including the front end sending the right data)
    - [ ] Front end shows pipeline and security groups, make sure that is set properly in crawler and backend.
  - [ ] Can't upload docs right now?
  - [ ] Good logs
- [ ] Update RAG pipeline for BD docs too.
- [ ] Load real data into milvus

- **Future**:
- [ ] Design local-minified version
  - [ ] single binary (ish) that runs, indexes the directories you tell it, then you can chat with it.
  - [ ] files and indexes stored .rrccp