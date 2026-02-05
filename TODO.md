# TODO

**Tasks**:

- Search tool:
  - Make new version that has all the types in a single file for open-webui.
- Radchat:
  - Use the milvus search tool with correct collection name and user token.
    - Should get the token from the user auth header, not body.
    - Should be able to choose collection name from the prompt, not be from the body.
  - Good system prompt.
  - Can answer database metadata questions and semantic questions.
  - Be able to use curl
  - Be able to use in chat with open-webui
  - Make new version that has all the types in a single file for open-webui.

Authorization:

- Create collection with permissions
- Upload document with permissions
- Search with permissions
- Sign into UI
- Interact with IDM

**Other**:

- Test the creation of a new collection via curl.
  - Modify the api so that a it doesn't take a template, but a whole config. The UI can use the template dropdown to select a template and fill in the details.
- Make good templates
  - in ui, let the users select a template from a dropdown.
- Fix the types in search results make sure things are rendered correctly.
- ensure that the upload and search from the ui use rbac.
- The source has the filename as the base name, this should be used to help determine the title if not provided.
  - in the database, the source should be the basename.
- remove `recreate` from the database config.
- Uploading:
  - Make the metadata extraction more robust in crawler
  - test with new collection, upload, search, chat, permissions
- Authentication:
  - Reintegrate oauth again
  - make proper syncing scrupt for milvus users and groups.
  - Milvus collections can have 3 settings for authentication:
    - public: anyone logged in can access
    - private: only the creator user can access
    - group only: only users in the group can access
      - Can create a new group by selecting users from the user list when creating a new collection.
- Good logs
- Update RAG pipeline for BD docs too.
- **Future**:
- Design local-minified version
  - single binary (ish) that runs, indexes the directories you tell it, then you can chat with it.
  - files and indexes stored .rrccp
- Break the vector db into a standalone package.

**Authentication**:

- Notes:
  - The backend can send a jwt token to the frontend that contains the user's email and the security groups they are in.
    - This JWT can be parsed and used to enforce security on the backend.
    - This way they can also use curl and other apps they build to interact with the backend.
    - Jwt should have an expiry and a refresh token.
  - Collection levels: public, private, admin.
    - Public: anyone logged in can access
    - Private: only the creator and selected groups user can access
    - Admin: only users in the admin group can access (including the npe)
  - Through our api, all users can access any collection, but the security groups will filter the documents that are returned.
    - Should not recieve any documents that are not in the security groups of the user.
- Tasks:
  - Outline how security is going to be implemented in the backend.
  - Create a script that defines (as code) the security groups and collection level security permissions (privelege groups and user roles).
  - Create a yaml file that defines the security groups and collection level security permissions (privelege groups and user roles).
    - Creates a list of users with user names and password and their security groups.
    - This is just for testing purposes.

Milvus Security Groups Outline:

```yaml
users:
  - name: "admin_user"
    password: "admin"
    roles:
      - "admin"
  - name: "test_user"
    password: "test"
    roles:
      - "default"

roles:
  - name: "admin"
    collections:
      - "*"
    privilege_groups:
      - "CollectionAdmin"
      - "DatabaseAdmin"
      - "ClusterAdmin"
  - name: "npe"
    collections:
      - "*"
    privilege_groups:
      - "CollectionReadWrite"
      - "DatabaseReadWrite"
      - "ClusterReadWrite"
  - name: "default"
    collections: # None or IRAD, or any public collection (list_collections, describe_collection, collection_description.database_config.access_level == "public")
    privilege_groups:
      - "CollectionReadWrite"
      - "DatabaseReadWrite"
      - "ClusterReadWrite"

security_groups: # also stored as roles in milvus, but never assigned privileges
  - name: "bd"
    collection: "bd"
  - name: "finance"
    collection: "bd"

builtin_privilege_groups:
  - name: "CollectionReadOnly"
  - name: "CollectionReadWrite"
  - name: "CollectionAdmin"
  - name: "DatabaseReadOnly"
  - name: "DatabaseAdmin"
  - name: "ClusterAdmin"


```

```bash
TOKEN="matt:steffen"
CRAWLER_CONFIG='{"name":"standard","embeddings":{"model":"qwen3-embedding:0.6b","base_url":"http://localhost:11434","api_key":"","provider":"ollama","dimension":null},"llm":{"model_name":"gpt-oss:20b","base_url":"http://localhost:11434","system_prompt":null,"ctx_length":32000,"default_timeout":300.0,"provider":"ollama","api_key":"","structured_output":"tools"},"vision_llm":{"model_name":"qwen3-vl:2b","base_url":"http://localhost:11434","system_prompt":null,"ctx_length":32000,"default_timeout":300.0,"provider":"ollama","api_key":"","structured_output":"response_format"},"database":{"provider":"milvus","collection":"athirdtest","partition":null,"access_level":"public","recreate":false,"collection_description":"Standard document collection","host":"localhost","port":19530,"username":"placeholder","password":"placeholder"},"converter":{"type":"pymupdf4llm","vlm_config":{"model_name":"qwen3-vl:2b","base_url":"http://localhost:11434","system_prompt":null,"ctx_length":32000,"default_timeout":300.0,"provider":"ollama","api_key":"","structured_output":"response_format"},"image_prompt":"Describe this image in detail. Focus on the main content, objects, text, and any relevant information useful in a document context.","max_workers":2,"to_markdown_kwargs":{}},"extractor":{"json_schema":{"type":"object","required":["title","authors","year","keywords"],"properties":{"title":{"type":"string","maxLength":500,"description":"Document title."},"authors":{"type":"array","description":"List of authors or contributors.","items":{"type":"string","maxLength":255},"minItems":1},"year":{"type":"integer","description":"Publication year for filtering and sorting.","minimum":1900,"maximum":2100},"document_type":{"type":"string","enum":["report","article","book","whitepaper","manual","presentation","other"],"description":"Broad document category for filtering."},"categories":{"type":"array","description":"High-level subject categories.","items":{"type":"string","maxLength":100}},"keywords":{"type":"array","description":"Searchable keywords describing content.","items":{"type":"string","maxLength":100}},"description":{"type":"string","maxLength":5000,"description":"Brief summary or abstract."}}},"context":"General document collection","structured_output":"json_schema","include_benchmark_questions":false,"num_benchmark_questions":3,"truncate_document_chars":4000,"strict":true},"chunking":{"chunk_size":2000,"overlap":200,"strategy":"naive","preserve_paragraphs":true,"min_chunk_size":100},"metadata_schema":{"type":"object","required":["title","authors","year","keywords"],"properties":{"title":{"type":"string","maxLength":500,"description":"Document title."},"authors":{"type":"array","description":"List of authors or contributors.","items":{"type":"string","maxLength":255},"minItems":1},"year":{"type":"integer","description":"Publication year for filtering and sorting.","minimum":1900,"maximum":2100},"document_type":{"type":"string","enum":["report","article","book","whitepaper","manual","presentation","other"],"description":"Broad document category for filtering."},"categories":{"type":"array","description":"High-level subject categories.","items":{"type":"string","maxLength":100}},"keywords":{"type":"array","description":"Searchable keywords describing content.","items":{"type":"string","maxLength":100}},"description":{"type":"string","maxLength":5000,"description":"Brief summary or abstract."}}},"temp_dir":"tmp/","use_cache":true,"benchmark":false,"generate_benchmark_questions":false,"num_benchmark_questions":3,"security_groups":["public"]}'
curl -X POST http://localhost:8000/v1/collections \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"access_level\": \"public\", \"access_groups\": [], \"crawler_config\": $CRAWLER_CONFIG}"
```

