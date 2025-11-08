# TODO

## General
- [ ] Add a deployment system to the page
- [ ] Add a monitoring system to the page

## File Upload
- [ ] Enable multiple file upload.

## Metadata Preview





- Demo Prep:
  - Create 3 users in the database.
    - User 1: Admin
    - User 2 and 3: doc1Viewer, doc2Viewer
  - Create 2 pipelines, one for secure, one for public.
  - Select 4 documents, 2 for secure, 2 for public.
  - Create the Radchat model.
  - Create the Milvus model that has a valve specifically for a collection name.

- Demo walk through:
  - Open Powerpoint presentation.
    - Slide on What LLMs are, how to use them, what to expect.
    - Slide on Uploading documents and Milvus vector search.
    - Show *our* llm stats like tokens per second on the models we support.
  - Sign in to OpenWebUI. Chat with normal LLM. Chat with Radchat.
  - Public collection:
    - Create a new collection.
    - Upload a file to the collection.
    - Edit the metadata for the file.
    - Upload another file without editing the metadata.
    - Open OpenWebUI and chat with those files.
      - What files are available to chat with?
      - What are the titles and authors?
      - `Some interesting questions to ask about those files specifically.`
  - Open Powerpoint again
    - 3 Slides on how we handle security.
  - Secure collection (with security rules):
    - Create a new collection.
    - Upload a file to the collection.
    - Confirm security rules for the file.
    - Upload another file without and use different security rules.
    - Open OpenWebUI and chat with those files with user 1.
      - What files are available to chat with?
      - What are the titles and authors?
      - `Some interesting questions to ask about those files specifically.`
    - Open OpenWebUI and chat with those files with user 2.
      - What files are available to chat with?
      - What are the titles and authors?
      - `Some interesting questions to ask about those files specifically.`


- The pipeline isn't efficient. Currently it doesn't check duplicates, or maybe it does but not based on the file (could use the uuid source stuff which is going to be new every time).

- Notes:
  - The UI: Author should be 'Authors', and should be displayed better as the array. Currently no spacing between the authors.
  - Confirm upload check box is to give an idea of discoverability (I'll upload to the database, then tell the user how many chunks, and on searches how often is it correctly seen).
  - The backend may not accept the already confirmed metadata as part of the upload process.
  - use a drop down to select the document to set the security rules for and metatdata for in the modal for multiple files .
  - In the select users we should have groups of users too (ex: All Users, All Admins, All Editors, All Viewers, etc.). This should be a security role (same thing as group i guess). Then it automatically checks the users in the group.