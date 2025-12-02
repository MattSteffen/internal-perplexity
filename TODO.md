# TODO

- **Tasks**:
  - [ ] Document Pipelines:
    - [ ] Should have an implementation of crawler for each pipeline, so you can call pipeline.crawl(document) to process the document.
    - [ ] Default pipeline
    - [ ] The pipeline name displayed and saved should be different from the collection name (or can be). It should be the name of the pipeline and how the data is processed.
  - [x] Implement search functionality.
    - [x] Use search to get number of documents in the collection.
  - [ ] Make sure collection information is properly rendered in the front end.
    - [ ] Number of partitions is not displayed.
    - [x] Number of documents is not displayed.
    - [ ] Number of chunks is not displayed.
    - [ ] Permissions are not displayed.
  - [ ] Can't upload docs right now?
  - [ ] Good logs
- [ ] Update RAG pipeline for BD docs too.
- [ ] Load real data into milvus

- **Future**:
- [ ] Design local-minified version
  - [ ] single binary (ish) that runs, indexes the directories you tell it, then you can chat with it.
  - [ ] files and indexes stored .rrccp