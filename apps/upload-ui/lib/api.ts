// API client functions for backend endpoints

import type {
  Collection,
  CollectionsResponse,
  CollectionInfo,
  ProcessedDocument,
  User,
  SearchRequest,
  SearchResponse,
  UploadResponse,
} from "./types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

/**
 * Get the authentication token from localStorage (username:password format)
 * Falls back to environment variable or default if not available.
 */
function getAuthToken(): string {
  if (typeof window !== "undefined") {
    const username = localStorage.getItem("username");
    const password = localStorage.getItem("password");
    if (username && password) {
      return `${username}:${password}`;
    }
  }
  return process.env.NEXT_PUBLIC_API_AUTH_TOKEN || "matt:steffen";
}

/**
 * Fetch all collections from the backend
 * curl -X GET http://localhost:8000/v1/collections \
 *   -H "Authorization: Bearer matt:steffen"
 * 
 * Example API response (NEW FORMAT):
 * {
 *   "collection_names": ["my_collection", "arxiv2"],
 *   "collections": {
 *     "my_collection": {
 *       "description": "Human readable description",
 *       "metadata_schema": {...},
 *       "num_documents": 100,
 *       "required_roles": ["admin"]
 *     }
 *   }
 * }
 */
export async function fetchCollections(): Promise<Collection[]> {
  const response = await fetch(`${API_BASE_URL}/v1/collections`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${getAuthToken()}`,
    },
  });

  if (!response.ok) {
    let errorMessage = `Failed to fetch collections: ${response.statusText}`;
    try {
      const errorData = await response.json();
      if (errorData.detail || errorData.message) {
        errorMessage = errorData.detail || errorData.message;
      }
    } catch {
      // Ignore JSON parse errors, use default message
    }
    throw new Error(errorMessage);
  }

  const data: CollectionsResponse = await response.json();

  // Transform API response to Collection[] format
  if (!data.collection_names || !Array.isArray(data.collection_names)) {
    console.warn("Unexpected collections response format:", data);
    return [];
  }

  return data.collection_names.map((collectionName) => {
    const collectionInfo: CollectionInfo | undefined = data.collections?.[collectionName];

    // Extract fields from CollectionInfo (new format)
    if (!collectionInfo) {
      // Fallback: return minimal collection info if CollectionInfo is missing
      return {
        name: collectionName,
        description: undefined,
        metadata_schema: undefined,
        num_documents: 0,
        num_chunks: 0,
        num_partitions: 0,
        required_roles: [],
      };
    }

    return {
      name: collectionName,
      description: collectionInfo.description || undefined,
      metadata_schema: collectionInfo.metadata_schema,
      num_documents: collectionInfo.num_documents,
      num_chunks: collectionInfo.num_chunks,
      num_partitions: collectionInfo.num_partitions,
      required_roles: collectionInfo.required_roles,
      access_level: collectionInfo.access_level,
      // Note: security_rules are not part of the API response
      // They may be added later or come from a different source
      security_rules: undefined,
      created_at: undefined,
      updated_at: undefined,
    };
  });
}

/**
 * Process a document to extract metadata (collection-specific)
 * curl -X POST http://localhost:8000/v1/collections/{collection_name}/process \
 *   -H "Authorization: Bearer matt:steffen" \
 *   -F "file=@document.pdf"
 */
export async function processDocument(
  collectionName: string,
  file: File,
): Promise<ProcessedDocument> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(
    `${API_BASE_URL}/v1/collections/${collectionName}/process`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${getAuthToken()}`,
      },
      body: formData,
    },
  );

  if (!response.ok) {
    let errorMessage = `Failed to process document: ${response.statusText}`;
    try {
      const errorData = await response.json();
      if (errorData.detail || errorData.message) {
        errorMessage = errorData.detail || errorData.message;
      }
    } catch {
      // Ignore JSON parse errors, use default message
    }
    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Upload a document to a collection
 * curl -X POST http://localhost:8000/v1/collections/{collection_name}/upload \
 *   -H "Authorization: Bearer matt:steffen" \
 *   -F "file=@document.pdf" \
 *   -F "metadata_override={\"title\":\"Example\",\"author\":\"John Doe\"}"
 */
export async function uploadDocument(
  collectionName: string,
  file: File,
  metadata: Record<string, unknown>,
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("metadata_override", JSON.stringify(metadata));

  const response = await fetch(
    `${API_BASE_URL}/v1/collections/${collectionName}/upload`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${getAuthToken()}`,
      },
      body: formData,
    },
  );

  if (!response.ok) {
    let errorMessage = `Failed to upload document: ${response.statusText}`;
    try {
      const errorData = await response.json();
      if (errorData.detail || errorData.message) {
        errorMessage = errorData.detail || errorData.message;
      }
    } catch {
      // Ignore JSON parse errors, use default message
    }
    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Fetch all security roles from the backend
 * curl -X GET http://localhost:8000/v1/roles \
 *   -H "Authorization: Bearer matt:steffen"
 */
export async function fetchRoles(): Promise<string[]> {
  const response = await fetch(`${API_BASE_URL}/v1/roles`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${getAuthToken()}`,
    },
  });

  if (!response.ok) {
    let errorMessage = `Failed to fetch roles: ${response.statusText}`;
    try {
      const errorData = await response.json();
      if (errorData.detail || errorData.message) {
        errorMessage = errorData.detail || errorData.message;
      }
    } catch {
      // Ignore JSON parse errors, use default message
    }
    throw new Error(errorMessage);
  }

  const data = await response.json();
  // Handle both array response and object with roles property
  if (Array.isArray(data)) {
    return data;
  }
  if (data.roles && Array.isArray(data.roles)) {
    return data.roles;
  }
  return [];
}

/**
 * Fetch all users from the backend
 * curl -X GET http://localhost:8000/v1/users \
 *   -H "Authorization: Bearer matt:steffen"
 */
export async function fetchUsers(): Promise<User[]> {
  const response = await fetch(`${API_BASE_URL}/v1/users`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${getAuthToken()}`,
    },
  });

  if (!response.ok) {
    let errorMessage = `Failed to fetch users: ${response.statusText}`;
    try {
      const errorData = await response.json();
      if (errorData.detail || errorData.message) {
        errorMessage = errorData.detail || errorData.message;
      }
    } catch {
      // Ignore JSON parse errors, use default message
    }
    throw new Error(errorMessage);
  }

  const data = await response.json();
  // Handle both array response and object with users property
  if (Array.isArray(data)) {
    return data;
  }
  if (data.users && Array.isArray(data.users)) {
    return data.users;
  }
  return [];
}

/**
 * Create a new collection with access level
 * curl -X POST http://localhost:8000/v1/collections \
 *   -H "Authorization: Bearer matt:steffen" \
 *   -H "Content-Type: application/json" \
 *   -d '{
 *     "collection_name": "my_collection",
 *     "config_overrides": {"embedding_model": "nomic-embed-text"},
 *     "access_level": "public"
 *   }'
 */
export async function createCollection(
  data: {
    collection_name: string;
    template_name?: string | null;
    custom_config?: Record<string, unknown> | null;
    config_overrides?: Record<string, unknown> | null;
    description?: string | null;
    roles?: string[] | null;
    access_level?: "public" | "private" | "admin";
    metadata_schema?: Record<string, unknown> | null;
  },
): Promise<{ collection_name: string; message: string; roles: string[] }> {
  const response = await fetch(`${API_BASE_URL}/v1/collections`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${getAuthToken()}`,
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    let errorMessage = `Failed to create collection: ${response.statusText}`;
    try {
      const errorData = await response.json();
      if (errorData.detail || errorData.message) {
        errorMessage = errorData.detail || errorData.message;
      }
    } catch {
      // Ignore JSON parse errors, use default message
    }
    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Search a Milvus collection with hybrid search
 * curl -X POST http://localhost:8000/v1/search \
 *   -H "Authorization: Bearer $TOKEN" \
 *   -H "Content-Type: application/json" \
 *   -d '{
 *     "collection": "my_collection",
 *     "text": "query text",
 *     "filters": ["title == \"example\""],
 *     "limit": 100
 *   }'
 */
export async function searchCollection(
  request: SearchRequest,
): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${getAuthToken()}`,
    },
    body: JSON.stringify({
      collection: request.collection,
      text: request.text,
      filters: request.filters || [],
      limit: request.limit || 100,
    }),
  });

  if (!response.ok) {
    let errorMessage = `Failed to search collection: ${response.statusText}`;
    try {
      const errorData = await response.json();
      if (errorData.detail || errorData.message) {
        errorMessage = errorData.detail || errorData.message;
      }
    } catch {
      // Ignore JSON parse errors, use default message
    }
    throw new Error(errorMessage);
  }

  return response.json();
}

