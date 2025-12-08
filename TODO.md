# TODO

**Tasks**:
- [ ] Remove pipeline_name logic and just use the collection_name
- [ ] Uploading:
  - [ ] Make the metadata extraction more robust
  - [ ] Make sure the search and document counting works, right now it shows one document.
    - [ ] Or recreate is set to false and it works.
- [ ] Good logs
- [ ] Update RAG pipeline for BD docs too.
- [ ] Load real data into milvus

- **Future**:
- [ ] Design local-minified version
  - [ ] single binary (ish) that runs, indexes the directories you tell it, then you can chat with it.
  - [ ] files and indexes stored .rrccp