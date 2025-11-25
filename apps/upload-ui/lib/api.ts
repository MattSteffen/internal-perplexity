// API client functions for backend endpoints

import type {
  Collection,
  CollectionsApiResponse,
  ProcessedDocument,
  Role,
  User,
  CollectionMetadataJson,
  PipelineConfig,
  CollectionPermissions,
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
 * Example API response:
 * {
 *   "collections": ["my_collection", "arxiv2"],
 *   "collection_metadata": {
 *     "my_collection": {
 *       "collection_name": "my_collection",
 *       "auto_id": false,
 *       "num_shards": 1,
 *       "description": "",
 *       "fields": [
 *         {
 *           "field_id": 100,
 *           "name": "id",
 *           "description": "",
 *           "type": 5,
 *           "params": {},
 *           "is_primary": true
 *         },
 *         {
 *           "field_id": 103,
 *           "name": "dense",
 *           "description": "",
 *           "type": 101,
 *           "params": { "dim": 5 }
 *         }
 *       ],
 *       "functions": [],
 *       "collection_id": 457696725674011330,
 *       "consistency_level": 2,
 *       "num_partitions": 1,
 *       "enable_dynamic_field": true,
 *       "created_timestamp": 457787511882645509,
 *       "update_timestamp": 457787511882645509
 *     }
 *   }
 * }
 */
// TODO: Should parse certain things from the description and display those.
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

  const data: CollectionsApiResponse = await response.json();

  // Transform API response to Collection[] format
  if (!data.collections || !Array.isArray(data.collections)) {
    console.warn("Unexpected collections response format:", data);
    return [];
  }

  return data.collections.map((collectionName) => {
    const metadata = data.collection_metadata?.[collectionName];

    // Helper function to safely convert Milvus timestamp to ISO string
    // Milvus timestamps are typically in milliseconds since epoch, but can be very large
    // and may exceed JavaScript's MAX_SAFE_INTEGER. We need to convert them properly.
    const convertTimestamp = (timestamp: number | undefined): string | undefined => {
      if (!timestamp) {
        return undefined;
      }

      try {
        let milliseconds: number;
        const maxSafeTimestamp = Number.MAX_SAFE_INTEGER;

        // If timestamp exceeds safe integer range, it's likely in nanoseconds or microseconds
        if (timestamp > maxSafeTimestamp) {
          // Try converting from nanoseconds (divide by 1,000,000)
          const nanosecondsToMs = timestamp / 1_000_000;
          // Try converting from microseconds (divide by 1,000)
          const microsecondsToMs = timestamp / 1_000;

          // Check which conversion gives a reasonable timestamp (between 1970 and 2100)
          const minReasonableMs = 0; // Jan 1, 1970
          const maxReasonableMs = 4102444800000; // Jan 1, 2100

          if (
            nanosecondsToMs >= minReasonableMs &&
            nanosecondsToMs <= maxReasonableMs &&
            nanosecondsToMs <= maxSafeTimestamp
          ) {
            milliseconds = nanosecondsToMs;
          } else if (
            microsecondsToMs >= minReasonableMs &&
            microsecondsToMs <= maxReasonableMs &&
            microsecondsToMs <= maxSafeTimestamp
          ) {
            milliseconds = microsecondsToMs;
          } else {
            // For Milvus hybrid timestamps, extract the physical timestamp part
            // Milvus hybrid timestamp format: (physical_timestamp << 18) | logical_timestamp
            // We can extract physical timestamp by right-shifting by 18 bits
            const physicalTimestamp = Math.floor(timestamp / (1 << 18));
            const physicalMs = physicalTimestamp / 1_000_000; // Convert to milliseconds

            if (
              physicalMs >= minReasonableMs &&
              physicalMs <= maxReasonableMs &&
              physicalMs <= maxSafeTimestamp
            ) {
              milliseconds = physicalMs;
            } else {
              console.warn(
                `Cannot convert timestamp ${timestamp} to a reasonable date value`,
              );
              return undefined;
            }
          }
        } else {
          // Timestamp is within safe range, use as-is (assuming it's already in milliseconds)
          milliseconds = timestamp;
        }

        const date = new Date(milliseconds);
        if (isNaN(date.getTime())) {
          console.warn(
            `Invalid date after conversion: timestamp ${timestamp} -> ${milliseconds} ms`,
          );
          return undefined;
        }

        const isoString = date.toISOString();
        return isoString;
      } catch (error) {
        console.error(`Error converting timestamp ${timestamp}:`, error);
        return undefined;
      }
    };

    // Parse description from JSON string if it exists
    let parsedDescription: string | undefined;
    let pipelineConfig: PipelineConfig | undefined;
    let permissions: CollectionPermissions | undefined;
    let metadataSchema: unknown | undefined;

    if (metadata?.description && typeof metadata.description === "string") {
      try {
        const parsed: CollectionMetadataJson = JSON.parse(metadata.description);
        parsedDescription = parsed.description;
        pipelineConfig = parsed.pipeline_config;
        permissions = parsed.permissions;
        // Parse metadata_schema if it's a JSON string
        if (parsed.metadata_schema) {
          metadataSchema =
            typeof parsed.metadata_schema === "string"
              ? JSON.parse(parsed.metadata_schema)
              : parsed.metadata_schema;
        }
      } catch (error) {
        // If parsing fails, use the description as-is (fallback for old format)
        console.warn(
          `Failed to parse collection metadata JSON for ${collectionName}:`,
          error,
        );
        parsedDescription = metadata.description;
      }
    }

    // Create enhanced metadata object with parsed schema
    const enhancedMetadata = metadata
      ? {
        ...metadata,
        parsed_metadata_schema: metadataSchema,
      }
      : undefined;

    return {
      name: collectionName,
      description: parsedDescription,
      metadata: enhancedMetadata,
      // Note: security_rules are not part of the API response
      // They may be added later or come from a different source
      security_rules: undefined,
      pipeline_config: pipelineConfig,
      permissions: permissions,
      created_at: convertTimestamp(metadata?.created_timestamp),
      updated_at: convertTimestamp(metadata?.update_timestamp),
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
 *   -F "metadata={\"title\":\"Example\",\"author\":\"John Doe\"}"
 */
export async function uploadDocument(
  collectionName: string,
  file: File,
  metadata: Record<string, unknown>,
): Promise<void> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("metadata", JSON.stringify(metadata));

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
}

/**
 * Fetch all security roles from the backend
 * curl -X GET http://localhost:8000/v1/roles \
 *   -H "Authorization: Bearer matt:steffen"
 */
export async function fetchRoles(): Promise<Role[]> {
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
 * Create a new collection with pipeline configuration and permissions
 * curl -X POST http://localhost:8000/v1/collections \
 *   -H "Authorization: Bearer matt:steffen" \
 *   -H "Content-Type: application/json" \
 *   -d '{
 *     "collection_name": "my_collection",
 *     "pipeline_name": "irads",
 *     "config_overrides": {"embedding_model": "nomic-embed-text"},
 *     "default_permissions": "public"
 *   }'
 */
export async function createCollection(
  data: {
    collection_name: string;
    pipeline_name?: string | null;
    custom_config?: Record<string, unknown> | null;
    config_overrides?: Record<string, unknown> | null;
    description?: string | null;
    default_permissions?: "admin_only" | "public";
    metadata_schema?: Record<string, unknown> | null;
  },
): Promise<{ collection_name: string; message: string }> {
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

