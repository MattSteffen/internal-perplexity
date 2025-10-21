# Milvus Setup

Outlining process for securing collections and documents in Milvus.

## Key Ideas

- Collections are the containers that hold the documents.
- Documents are the individual items that are stored in a collection.
  - Documents are often chunks of text from a single source.
  - A single IRAD or BD Document can have multiple chunks and thus multiple entries in the database.
- Security groups are our implementation of RBAC row-level access control.
  - We use a Bitmap to index on the required security groups for each document.
  - Each document can have multiple security groups required for access.
  - Milvus search functions DO NOT natively filter on security groups, this is handled at the application level.

See [Milvus RBAC documentation](https://milvus.io/docs/rbac.md) for more details.

## How we implement and enforce security groups

1. When inserting a document, we insert the security groups into the `security_group` field of the document.

- This can be done through the UI when creating a new document or when updating an existing document.

2. Documents that need to be locked down to sepecific users can have a security group set by the document name.

- Users must then have that role assigned to them.

3. Users searching must go through an application with appropriate access.

- Only applications we authorize will have read access to the collections.
- Thus users attempting to search without authorization will receive an error.

4. The application will filter the documents based on the security groups of the user.
