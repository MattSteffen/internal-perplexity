// Type definitions for collections and documents

export interface SecurityRule {
  name: string;
  description?: string;
  required: boolean;
}

export interface CollectionField {
  field_id: number;
  name: string;
  description?: string;
  type: number;
  params?: Record<string, unknown>;
  is_primary?: boolean;
  auto_id?: boolean;
}

export interface CollectionFunction {
  name: string;
  id: number;
  type: number;
  description?: string;
  input_field_names?: string;
  output_field_names?: string;
}

export interface CollectionMetadata {
  collection_name: string;
  auto_id?: boolean;
  num_shards?: number;
  description?: string;
  fields: CollectionField[];
  functions?: CollectionFunction[];
  collection_id: number;
  consistency_level?: number;
  num_partitions?: number;
  enable_dynamic_field?: boolean;
  created_timestamp?: number;
  update_timestamp?: number;
}

// New API response format matching backend CollectionsResponse
export interface CollectionInfo {
  description: string;
  metadata_schema: Record<string, unknown>;
  num_documents: number;
  num_chunks: number;
  num_partitions: number;
  required_roles: string[];
  access_level: AccessLevel;
}

export interface CollectionsResponse {
  collection_names: string[];
  collections: Record<string, CollectionInfo>;
}

/** Pipeline template from GET /v1/pipelines (backend PipelineInfo). */
export interface PipelineInfo {
  name: string;
  description: string;
  metadata_schema: Record<string, unknown>;
  chunk_size: number;
  embedding_model: string;
  llm_model: string;
}

export interface PipelinesResponse {
  pipelines: PipelineInfo[];
}

export interface Collection {
  name: string;
  description?: string;
  metadata?: CollectionMetadata;
  security_rules?: SecurityRule[];
  created_at?: string;
  updated_at?: string;
  access_level?: AccessLevel;
  // New fields from CollectionInfo
  metadata_schema?: Record<string, unknown>;
  num_documents?: number;
  num_chunks?: number;
  num_partitions?: number;
  required_roles?: string[];
}

export interface DocumentMetadata {
  title?: string;
  author?: string;
  date?: string;
  [key: string]: unknown;
}

export interface ProcessedDocument {
  metadata: DocumentMetadata;
  markdown_content: string;
  file_name: string;
  file_size?: number;
}

export interface UploadRequest {
  file: File;
  metadata: DocumentMetadata;
  collection_name: string;
}

export interface User {
  id: string;
  name: string;
  roles: string[];
}

export type AccessLevel = "public" | "private" | "group_only" | "admin";

/** Request body for POST /v1/collections. Collection name comes from crawler_config.database.collection. */
export interface CreateCollectionRequest {
  access_level: AccessLevel;
  access_groups?: string[];
  crawler_config: Record<string, unknown>;
}

export interface CollectionMetadataJson {
  description?: string; // Old format - kept for backward compatibility
  library_context?: string; // New format - preferred field for collection description
  metadata_schema?: unknown;
  full_prompt?: string;
  llm_prompt?: string; // New format field name
  collection_config_json?: Record<string, unknown>; // New format field
  access_level?: AccessLevel;
}

export interface Chunk {
  text: string;
  chunk_index: number;
  text_embedding: number[];
  sparse_text_embedding?: number[] | null;
  sparse_metadata_embedding?: number[] | null;
}

// Document interface matching Python Document model from crawler
export interface Document {
  document_id: string;
  source: string;
  content?: string | null; // Base64 encoded bytes in JSON
  markdown?: string | null;
  metadata?: DocumentMetadata | null;
  benchmark_questions?: string[] | null;
  chunks?: Chunk[] | null;
  security_group: string[];
}

// Legacy SearchResult interface - kept for backward compatibility
// but no longer used in SearchResponse
export interface SearchResult {
  source?: string | null;
  chunk_index?: number | null;
  text?: string | null;
  str_metadata?: string | null;
  title?: string | null;
  author?: string | string[] | null;
  date?: string | null;
  keywords?: string[] | null;
  unique_words?: string[] | null;
  distance?: number | null;
}

export interface SearchResponse {
  results: Document[];
  total: number;
}

export interface SearchRequest {
  collection: string;
  text: string;
  filters?: string[];
  limit?: number;
}

export interface UploadResponse {
  message: string;
  collection_name: string;
  document_id?: string | null;
  chunks_created: number;
  processing_time_sec: number;
}

