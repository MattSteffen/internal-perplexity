- Implement security features for milvus
  - Add them to config files
  - Set up 2 repositories with different types of data
  - Test them based on user details in python script (not necessarily in ui)
  - upload different directory too
- Using basic_ollama as function in OI.

# MVP

## General

- [ ] Data sources

  - [ ] What local data do I download
    - [ ] General conference talks
    - [ ] Scriptures in chapters
  - [ ] What search apis do I use
    - [ ] brave search
  - [ ] What crawling do I do?
    - [ ] Levels deep of the link graph
    - [ ] How many links to follow
  - [ ] How to manage citations

- [x] Make repo public
- If using openwebui, include instructions for it's deployment and special config
  - Include pipelines

## Frontend

## Backend

- [x] Decide framework
  - Python then after MVP -> Go
- [ ] Create good async enabled load balancer

## Future

- [ ] After MVP
  - [ ] Refactor backend into Go
  - [ ] Create the small model fine tuning and test time inference, then run locally
