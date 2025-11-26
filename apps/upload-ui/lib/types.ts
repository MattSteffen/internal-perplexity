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

export interface CollectionsApiResponse {
  collections: string[];
  collection_metadata: Record<string, CollectionMetadata>;
}

export interface Collection {
  name: string;
  description?: string;
  metadata?: CollectionMetadata;
  security_rules?: SecurityRule[];
  created_at?: string;
  updated_at?: string;
  pipeline_config?: PipelineConfig;
  permissions?: CollectionPermissions;
}

export interface DocumentMetadata {
  title?: string;
  author?: string;
  date?: string;
  [key: string]: unknown;
}

export interface ProcessedDocument {
  metadata: DocumentMetadata;
  file_name: string;
  file_size?: number;
}

export interface UploadRequest {
  file: File;
  metadata: DocumentMetadata;
  collection_name: string;
}

export interface Role {
  role: string;
  privileges: string[];
}

export interface User {
  id: string;
  name: string;
  email?: string;
  roles?: string[];
}

export interface PipelineConfig {
  pipeline_name?: string;
  overrides?: Record<string, unknown>;
  full_config?: Record<string, unknown>;
}

export interface CollectionPermissions {
  default: "admin_only" | "public";
}

export interface CollectionMetadataJson {
  description?: string; // Old format - kept for backward compatibility
  library_context?: string; // New format - preferred field for collection description
  metadata_schema?: unknown;
  full_prompt?: string;
  llm_prompt?: string; // New format field name
  collection_config_json?: Record<string, unknown>; // New format field
  pipeline_config?: PipelineConfig;
  permissions?: CollectionPermissions;
}

