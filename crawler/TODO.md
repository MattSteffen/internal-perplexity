# Database Schema Prefixing Refactor

## Overview

This refactor aims to prefix all default/system database fields and metadata keys with `default_` to eliminate conflicts with user-provided metadata. This will remove the need for checking if user metadata contains reserved keys.

## Current State Analysis

### Reserved Keys (database_client.py)

- `document_id` → `default_document_id`
- `chunk_index` → `default_chunk_index`
- `source` → `default_source`
- `text` → `default_text`
- `text_embedding` → `default_text_embedding`
- `text_sparse_embedding` → `default_text_sparse_embedding`
- `metadata` → `default_metadata`
- `metadata_sparse_embedding` → `default_metadata_sparse_embedding`

### Base Schema Fields (milvus_utils.py)

- `id` → `default_id`
- `document_id` → `default_document_id`
- `minio` → `default_minio`
- `chunk_index` → `default_chunk_index`
- `source` → `default_source`
- `text` → `default_text`
- `text_embedding` → `default_text_embedding`
- `text_sparse_embedding` → `default_text_sparse_embedding`
- `metadata` → `default_metadata`
- `metadata_sparse_embedding` → `default_metadata_sparse_embedding`

### Search Output Fields (milvus_benchmarks.py)

- `source` → `default_source`
- `chunk_index` → `default_chunk_index`
- `text` → `default_text`
- `metadata` → `default_metadata`
- `title` → `default_title` (if exists)
- `author` → `default_author` (if exists)
- `date` → `default_date` (if exists)
- `keywords` → `default_keywords` (if exists)
- `unique_words` → `default_unique_words` (if exists)

## Implementation Plan

### Phase 1: Core Schema Changes

#### 1. Update RESERVED Keys List

**File:** `src/crawler/storage/database_client.py`

- Change RESERVED list to use prefixed versions
- Update `sanitize_metadata()` function to use new RESERVED keys
- Update any references to these keys in comments/documentation

#### 2. Update Base Schema Creation

**File:** `src/crawler/storage/milvus_utils.py`

- Modify `_create_base_schema()` to create prefixed field names
- Update all field definitions to use `default_` prefix
- Ensure index creation uses correct field names

#### 3. Update MilvusDB Class

**File:** `src/crawler/storage/milvus_client.py`

- Update all field name references in queries
- Update field name references in insert operations
- Update field name references in duplicate checking
- Update any hardcoded field names in filter expressions

#### 4. Update DatabaseDocument Protocol

**File:** `src/crawler/storage/database_client.py`

- Update `DatabaseDocument` dataclass to use prefixed field names
- Update `from_dict()` method to handle prefixed fields
- Update `__getitem__()` method to map prefixed fields

### Phase 2: Search and Query Operations

#### 5. Update Search Operations

**File:** `src/crawler/storage/milvus_benchmarks.py`

- Update `OUTPUT_FIELDS` list to use prefixed field names
- Update any field name references in search queries
- Update result processing to handle prefixed field names

#### 6. Update Query Operations

**Files:** All files with database queries

- Update filter expressions to use prefixed field names
- Update field selection lists to use prefixed field names
- Update any hardcoded field name references in WHERE clauses

### Phase 3: Metadata Handling

#### 7. Update Metadata Processing

**Files:** `src/crawler/main.py`, extractor files

- Update metadata extraction to store system fields with prefixes
- Update metadata reading to access prefixed fields
- Remove conflict checking logic (no longer needed)

#### 8. Update Schema Validation

**File:** `src/crawler/main.py`

- Remove RESERVED key checking from `sanitize_metadata()`
- Update JSON schema validation to ignore prefixed fields
- Simplify metadata validation logic

### Phase 4: Migration and Testing

#### 9. Migration Support

**File:** `src/crawler/storage/milvus_utils.py`

- Add migration function to rename existing fields
- Create upgrade path from old schema to new schema
- Handle data migration for existing collections

#### 10. Update Tests

**Files:** All test files

- Update test cases to use prefixed field names
- Update test data generation to use prefixed fields
- Update assertions to check for prefixed fields

### Phase 5: Documentation and Cleanup

#### 11. Update Documentation

**Files:** All documentation files

- Update README files to reflect new field naming
- Update docstrings and comments
- Update example code to show prefixed fields

#### 12. Code Cleanup

**Files:** All modified files

- Remove old conflict checking code
- Clean up any TODO comments related to reserved keys
- Update logging messages to reflect new field names

## Benefits After Implementation

1. **No More Conflicts**: User metadata can contain any keys without conflict
2. **Simplified Validation**: No need to check for reserved key conflicts
3. **Cleaner Code**: Remove complex conflict resolution logic
4. **Better User Experience**: Users can use any metadata keys they want
5. **Future-Proof**: Easy to add new system fields without worrying about conflicts

## Migration Strategy

### For New Collections

- Automatically use prefixed field names
- No migration needed

### For Existing Collections

- Provide migration script to rename fields
- Update collection schema in-place
- Handle data migration transparently

## Testing Strategy

1. **Unit Tests**: Test each component with prefixed field names
2. **Integration Tests**: Test full pipeline with prefixed fields
3. **Migration Tests**: Test migration from old to new schema
4. **Backward Compatibility**: Ensure old code still works during transition

## Rollback Plan

If issues arise:

1. Keep old field names as aliases during transition period
2. Provide rollback script to revert field names
3. Ensure backward compatibility during rollout

## Implementation Order

1. Start with core schema changes (Phase 1)
2. Update search operations (Phase 2)
3. Update metadata handling (Phase 3)
4. Add migration support (Phase 4)
5. Update documentation (Phase 5)

## Success Criteria

- [ ] All system fields use `default_` prefix
- [ ] User metadata can contain any keys without conflict
- [ ] No reserved key checking needed
- [ ] All tests pass with new field names
- [ ] Migration works for existing collections
- [ ] Documentation updated
- [ ] Backward compatibility maintained during transition
