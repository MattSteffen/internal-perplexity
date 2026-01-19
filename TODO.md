# TODO

**Tasks**:
- Test the creation of a new collection via curl.
  - Modify the api so that a it doesn't take a template, but a whole config. The UI can use the template dropdown to select a template and fill in the details.
- Make good templates
  - in ui, let the users select a template from a dropdown.
- Fix the types in search results make sure things are rendered correctly.
- ensure that the upload and search from the ui use rbac.
- The source has the filename as the base name, this should be used to help determine the title if not provided.
  - in the database, the source should be the basename.
- remove `recreate` from the database config.
- [ ] Uploading:
  - [ ] Make the metadata extraction more robust in crawler
  - [ ] test with new collection, upload, search, chat, permissions
- [ ] Authentication:
  - [ ] Reintegrate oauth again
  - [ ] make proper syncing scrupt for milvus users and groups.
  - [ ] Milvus collections can have 3 settings for authentication:
    - [ ] public: anyone logged in can access
    - [ ] private: only the creator user can access
    - [ ] group only: only users in the group can access
      - [ ] Can create a new group by selecting users from the user list when creating a new collection.
- [ ] Good logs
- [ ] Update RAG pipeline for BD docs too.

- **Future**:
- [ ] Design local-minified version
  - [ ] single binary (ish) that runs, indexes the directories you tell it, then you can chat with it.
  - [ ] files and indexes stored .rrccp
- [ ] Break the vector db into a standalone package.


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