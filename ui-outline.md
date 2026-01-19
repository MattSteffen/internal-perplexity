# RBAC Implementation Guide for Milvus-backed RAG Application

## 1) Threat Model + Security Invariants

### Must-Never-Happen Items

| ID | Threat | Mitigation Layer |
|----|--------|------------------|
| T1 | Unauthorized document returned to user | App-enforced Milvus filter (mandatory) |
| T2 | Filter bypass or omission | Centralized query builder; no raw Milvus access |
| T3 | Privilege escalation via `security_groups` tampering on write | Write-path validation; allowed-to-assign policy |
| T4 | Collection access without collection-level permission | App-level collection gate (checked before Milvus call) |
| T5 | Inference via error messages (e.g., "doc exists but denied") | Uniform "not found" responses; no count leakage |
| T6 | Inference via timing (filtered vs no-match) | Constant-time-ish response path; avoid early-exit |
| T7 | Stale group membership grants access after revocation | Bounded cache TTL; fail-closed on LDAP errors |
| T8 | Client-supplied filter injection | Server constructs all filters; client params are data only |
| T9 | Enumeration of collections user cannot access | App controls collection list; no Milvus discovery exposed |

### Enforcement Boundaries

| Check | Enforced By | Notes |
|-------|-------------|-------|
| Collection-level permission | **Application** | Before any Milvus call |
| Document-level permission | **Milvus filter** (constructed by app) | `security_groups` intersection |
| Write validation (`security_groups` assignment) | **Application** | Before insert/upsert |
| User identity | **Application** (LDAP) | Token → username resolution |
| Group membership | **Application** (LDAP + cache) | Source of truth = LDAP |

**Milvus is untrusted for authorization.** It executes filters but cannot enforce policy; all policy logic lives in the application.

---

## 2) Authorization Model

### Collection Permission Model

```
Permission levels (ordered):
  admin  > rw  > r  > none

  admin : create/drop collection, manage schema, full CRUD, assign any security_groups
  rw    : insert/upsert/delete docs, query/search (with doc filter)
  r     : query/search only (with doc filter)
  none  : no access (deny)
```

### Document Permission Model

**Field**: `security_groups` (array of strings)

**Semantics**:
- User can read doc iff `intersection(doc.security_groups, user.effective_groups) ≠ ∅`
- If `security_groups` is **missing or empty**: **DENY** (fail-safe default)
- Groups are string identifiers (stable IDs recommended over display names)

### LDAP Group Naming Convention

```
Namespace: milvus:<collection>:<permission>
           milvus:doc:<group_id>

Collection-level groups (grant collection permission):
  milvus:contracts:admin   → admin on "contracts" collection
  milvus:contracts:rw      → read/write on "contracts" collection
  milvus:contracts:r       → read-only on "contracts" collection

Document-level groups (used in security_groups field):
  milvus:doc:legal-team
  milvus:doc:finance-team
  milvus:doc:project-alpha
  milvus:doc:confidential-hr

Tagging permission groups (who can assign a doc-group on write):
  milvus:tag:legal-team       → can assign "milvus:doc:legal-team" to docs
  milvus:tag:confidential-hr  → can assign "milvus:doc:confidential-hr" to docs
```

### Example Mapping Table

| User | LDAP Groups | Collection: `contracts` | Collection: `hr_docs` | Doc Groups (for filtering) |
|------|-------------|-------------------------|----------------------|---------------------------|
| alice | `milvus:contracts:rw`, `milvus:hr_docs:r`, `milvus:doc:legal-team`, `milvus:tag:legal-team` | rw | r | legal-team |
| bob | `milvus:contracts:r`, `milvus:doc:finance-team` | r | none | finance-team |
| admin_carol | `milvus:contracts:admin`, `milvus:hr_docs:admin`, `milvus:doc:*` (or explicit list) | admin | admin | all |
| eve | (none) | none | none | (none) |

---

## 3) LDAP Sync Strategy

### Strategy Comparison

| Strategy | Latency | Revocation Speed | LDAP Load | Complexity |
|----------|---------|------------------|-----------|------------|
| **On-demand + cache** | Low (cache hit) / Medium (miss) | ≤ cache TTL | Per-request on miss | Medium |
| **Periodic bulk sync** | Very low | ≤ sync interval | Batch, predictable | Higher |
| **Hybrid** | Low | Bounded by shorter of TTL/sync | Balanced | Highest |

### Recommended: On-Demand with Short-TTL Cache

**Justification**:
- Simpler to implement and reason about
- Revocation speed directly controlled by TTL
- No need for full user enumeration (LDAP may restrict this)
- Works well for moderate QPS (<1000 req/s); beyond that, consider hybrid

### Specification

```yaml
cache:
  backend: redis  # or in-process with size bound
  key_pattern: "ldap:groups:{username}"
  ttl: 300        # 5 minutes (revocation window)
  max_staleness: 300  # same as TTL; no grace
  negative_cache_ttl: 60  # cache "user has no groups" briefly

ldap_outage:
  policy: fail_closed  # deny all requests if LDAP unreachable AND cache miss
  grace_period: 0      # no grace; security > availability
  # Alternative: allow 1 TTL window of cached data, then fail closed

performance:
  expected_groups_per_user: 10-50
  max_groups_per_user: 500  # sanity limit; reject if exceeded
  ldap_timeout: 3s
  cache_refresh_threshold: 60  # refresh if <60s remaining on TTL (background)
```

### Revocation Flow

```
1. Admin removes user from LDAP group
2. User's next request after cache expiry:
   - Cache miss or TTL expired
   - Fresh LDAP lookup returns updated groups
   - New groups cached; old access revoked
3. Worst-case revocation delay = cache TTL (300s)
```

### LDAP Outage Handling

```python
def get_user_groups(username: str) -> list[str]:
    cached = cache.get(f"ldap:groups:{username}")
    if cached and not cached.expired:
        return cached.groups
    
    try:
        groups = ldap_client.get_groups(username, timeout=3.0)
        cache.set(f"ldap:groups:{username}", groups, ttl=300)
        return groups
    except LDAPError:
        # FAIL CLOSED: do not use stale cache during outage
        log.error("LDAP unreachable", username=username)
        raise AuthorizationError("Unable to verify permissions; try again later")
```

---

## 4) Enforcement Flow (End-to-End Request Path)

### Sequence Diagram

```
┌──────────┐    ┌─────────────┐    ┌───────────┐    ┌───────┐    ┌────────┐
│  Client  │    │  API Layer  │    │ AuthZ Svc │    │ Cache │    │  LDAP  │
└────┬─────┘    └──────┬──────┘    └─────┬─────┘    └───┬───┘    └────┬───┘
     │                 │                 │              │             │
     │ POST /search    │                 │              │             │
     │ {collection,    │                 │              │             │
     │  query_vec,     │                 │              │             │
     │  token}         │                 │              │             │
     │────────────────>│                 │              │             │
     │                 │                 │              │             │
     │                 │ 1. resolve_user(token)         │             │
     │                 │────────────────>│              │             │
     │                 │                 │ 2. get_groups(user)        │
     │                 │                 │─────────────>│             │
     │                 │                 │    [cache miss]            │
     │                 │                 │              │─────────────>
     │                 │                 │              │  LDAP lookup│
     │                 │                 │              │<────────────│
     │                 │                 │<─────────────│             │
     │                 │                 │   groups[]   │             │
     │                 │                 │              │             │
     │                 │ 3. check_collection_access(user, collection, "r")
     │                 │<────────────────│              │             │
     │                 │   {allowed: true}              │             │
     │                 │                 │              │             │
     │                 │ 4. build_doc_filter(groups)    │             │
     │                 │────────────────>│              │             │
     │                 │<────────────────│              │             │
     │                 │   filter_expr   │              │             │
     │                 │                 │              │             │
     │                 │                 │              │             │
     │                 │  ┌────────┐     │              │             │
     │                 │  │ Milvus │     │              │             │
     │                 │  └───┬────┘     │              │             │
     │                 │      │          │              │             │
     │                 │ 5. search(collection, vec, filter=filter_expr)
     │                 │─────>│          │              │             │
     │                 │<─────│          │              │             │
     │                 │  results        │              │             │
     │                 │                 │              │             │
     │ 6. results      │                 │              │             │
     │<────────────────│                 │              │             │
     │                 │                 │              │             │
```

### Request Flow (Pseudocode)

```python
from fastapi import Request, HTTPException
from functools import wraps

class RBACEnforcer:
    def __init__(self, ldap_client, cache, milvus_client):
        self.ldap = ldap_client
        self.cache = cache
        self.milvus = milvus_client
    
    def resolve_user(self, request: Request) -> str:
        """Extract username from token or header."""
        token = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if not token:
            raise HTTPException(401, "Missing authentication")
        
        # Validate token with LDAP or token service
        username = self.ldap.validate_token(token)
        if not username:
            raise HTTPException(401, "Invalid token")
        return username
    
    def get_effective_groups(self, username: str) -> set[str]:
        """Fetch groups with caching. Fail closed on errors."""
        cache_key = f"ldap:groups:{username}"
        cached = self.cache.get(cache_key)
        
        if cached is not None:
            return set(cached)
        
        try:
            groups = self.ldap.get_user_groups(username)
            if len(groups) > 500:
                raise HTTPException(403, "Group limit exceeded")
            self.cache.set(cache_key, list(groups), ex=300)
            return set(groups)
        except LDAPConnectionError:
            # FAIL CLOSED
            raise HTTPException(503, "Authorization service unavailable")
    
    def get_collection_permission(
        self, groups: set[str], collection: str
    ) -> str:
        """Derive highest permission from groups."""
        if f"milvus:{collection}:admin" in groups:
            return "admin"
        if f"milvus:{collection}:rw" in groups:
            return "rw"
        if f"milvus:{collection}:r" in groups:
            return "r"
        return "none"
    
    def require_collection_access(
        self, username: str, groups: set[str], collection: str, min_level: str
    ):
        """Gate check. Raises if insufficient permission."""
        levels = {"none": 0, "r": 1, "rw": 2, "admin": 3}
        actual = self.get_collection_permission(groups, collection)
        
        if levels[actual] < levels[min_level]:
            # Log attempt but return generic error (no info leakage)
            log.warning("access_denied", user=username, collection=collection,
                        required=min_level, actual=actual)
            raise HTTPException(403, "Access denied")
    
    def build_doc_filter(self, groups: set[str]) -> str:
        """Build Milvus filter expression. NEVER skippable."""
        doc_groups = [g for g in groups if g.startswith("milvus:doc:")]
        
        if not doc_groups:
            # User has no doc-level groups → cannot read any docs
            # Return impossible filter
            return "1 == 0"
        
        # Escape group names to prevent injection
        escaped = [self._escape_string(g) for g in doc_groups]
        group_list = ", ".join(f'"{g}"' for g in escaped)
        
        # array_contains_any for array field
        return f"array_contains_any(security_groups, [{group_list}])"
    
    def _escape_string(self, s: str) -> str:
        """Escape special characters for Milvus expression."""
        return s.replace("\\", "\\\\").replace('"', '\\"')
```

---

## 5) Collection-Level RBAC Enforcement

### Permission Matrix

| Operation | Required Level | Additional Checks |
|-----------|----------------|-------------------|
| `search` / `query` | `r` | Doc filter applied |
| `get` (by ID) | `r` | Doc filter applied (fetch then filter) |
| `insert` | `rw` | Validate `security_groups` assignment |
| `upsert` | `rw` | Validate `security_groups` assignment |
| `delete` | `rw` | Only delete docs user can read (filter-scoped) |
| `create_collection` | `admin` | — |
| `drop_collection` | `admin` | — |
| `create_index` | `admin` | — |
| `describe_collection` | `r` | Metadata only, no doc access |

### Enforcement Implementation

```python
class MilvusGateway:
    """
    All Milvus operations go through this gateway.
    Direct Milvus client access is forbidden elsewhere in the codebase.
    """
    
    def __init__(self, milvus_client, rbac: RBACEnforcer):
        self._milvus = milvus_client  # Single service credential
        self._rbac = rbac
    
    def search(
        self,
        username: str,
        groups: set[str],
        collection: str,
        vector: list[float],
        top_k: int = 10,
        client_filter: str | None = None  # IGNORED for security
    ) -> list[dict]:
        # 1. Collection gate
        self._rbac.require_collection_access(username, groups, collection, "r")
        
        # 2. Build mandatory doc filter (NEVER uses client_filter)
        doc_filter = self._rbac.build_doc_filter(groups)
        
        # 3. Execute with enforced filter
        results = self._milvus.search(
            collection_name=collection,
            data=[vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            expr=doc_filter,  # Mandatory, server-constructed
            output_fields=["id", "text", "security_groups"]
        )
        
        # 4. Log for audit
        log.info("milvus_search",
            user=username,
            collection=collection,
            groups_hash=hashlib.sha256(str(sorted(groups)).encode()).hexdigest()[:16],
            filter_applied=doc_filter,
            result_count=len(results[0]) if results else 0
        )
        
        return self._format_results(results)
    
    def insert(
        self,
        username: str,
        groups: set[str],
        collection: str,
        documents: list[dict]
    ) -> list[str]:
        # 1. Collection gate (need rw)
        self._rbac.require_collection_access(username, groups, collection, "rw")
        
        # 2. Validate security_groups on each document
        for doc in documents:
            self._validate_security_groups_assignment(username, groups, doc)
        
        # 3. Execute insert
        ids = self._milvus.insert(collection_name=collection, data=documents)
        
        log.info("milvus_insert", user=username, collection=collection,
                 doc_count=len(documents))
        return ids
    
    def delete(
        self,
        username: str,
        groups: set[str],
        collection: str,
        ids: list[str]
    ) -> int:
        # 1. Collection gate
        self._rbac.require_collection_access(username, groups, collection, "rw")
        
        # 2. Fetch docs to verify user can read them (owns them for delete)
        doc_filter = self._rbac.build_doc_filter(groups)
        id_list = ", ".join(f'"{id}"' for id in ids)
        combined_filter = f"id in [{id_list}] and ({doc_filter})"
        
        # 3. Delete only matching (readable) docs
        result = self._milvus.delete(
            collection_name=collection,
            expr=combined_filter
        )
        
        log.info("milvus_delete", user=username, collection=collection,
                 requested=len(ids), deleted=result.delete_count)
        return result.delete_count
    
    def _validate_security_groups_assignment(
        self,
        username: str,
        groups: set[str],
        doc: dict
    ):
        """Prevent privilege escalation on write."""
        # Detailed in Section 7
        pass
```

### Tightness Guarantee

**Critical invariant**: Doc-level access alone is insufficient.

```python
# WRONG (never do this):
def bad_search(user, collection, vector):
    # Missing collection check! User could query any collection
    # if they happen to have overlapping doc groups
    filter = build_doc_filter(user.groups)
    return milvus.search(collection, vector, filter)

# CORRECT:
def good_search(user, collection, vector):
    # Collection check FIRST
    require_collection_access(user, collection, "r")  # ← Gate
    # Then doc filter
    filter = build_doc_filter(user.groups)
    return milvus.search(collection, vector, filter)
```

---

## 6) Document-Level RBAC in Milvus (Mechanical Details)

### Milvus Schema for `security_groups`

```python
from pymilvus import FieldSchema, DataType

fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="security_groups", dtype=DataType.ARRAY,
                element_type=DataType.VARCHAR, max_capacity=50, max_length=128),
    FieldSchema(name="created_at", dtype=DataType.INT64),
    FieldSchema(name="updated_at", dtype=DataType.INT64),
]
```

### Filter Construction

```python
def build_doc_filter(groups: set[str]) -> str:
    """
    Build Milvus filter for document-level RBAC.
    
    Uses array_contains_any for set intersection semantics.
    """
    # Extract only doc-level groups
    doc_groups = sorted([g for g in groups if g.startswith("milvus:doc:")])
    
    if not doc_groups:
        # No doc groups → impossible filter (deny all)
        return "array_length(security_groups) < 0"  # Always false
    
    # Escape and format
    escaped = [escape_milvus_string(g) for g in doc_groups]
    group_list = ", ".join(f'"{g}"' for g in escaped)
    
    # Milvus array_contains_any: true if ANY element matches
    return f"array_contains_any(security_groups, [{group_list}])"

def escape_milvus_string(s: str) -> str:
    """Escape for Milvus string literals."""
    # Milvus uses backslash escaping
    return (s
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("'", "\\'"))
```

### Handling Large Group Lists

**Problem**: Users with 100+ groups create huge filter expressions.

**Mitigations**:

```python
MAX_GROUPS_IN_FILTER = 100

def build_doc_filter_with_limit(groups: set[str]) -> str:
    doc_groups = sorted([g for g in groups if g.startswith("milvus:doc:")])
    
    if len(doc_groups) > MAX_GROUPS_IN_FILTER:
        # Option 1: Use group compression (precomputed effective IDs)
        compressed = get_compressed_group_ids(doc_groups)
        if len(compressed) <= MAX_GROUPS_IN_FILTER:
            return build_filter_from_ids(compressed)
        
        # Option 2: Fail safe with warning
        log.warning("group_limit_exceeded", count=len(doc_groups))
        raise HTTPException(400, 
            "Too many group memberships; contact admin for role consolidation")
    
    return build_standard_filter(doc_groups)
```

**Group Compression Strategy**:

```sql
-- Precompute "effective group IDs" that represent group combinations
-- Store in local DB

CREATE TABLE group_rollups (
    rollup_id VARCHAR(64) PRIMARY KEY,  -- e.g., "rollup:abc123"
    member_groups JSONB,                  -- ["milvus:doc:a", "milvus:doc:b", ...]
    created_at TIMESTAMP
);

-- User's effective_rollup_ids are computed and cached
-- Filter uses rollup IDs instead of raw groups when beneficial
```

### Group Renames/Deletes (Stable IDs)

**Recommendation**: Use stable IDs, not display names.

```python
# LDAP stores:
#   cn=legal-team,ou=groups,dc=corp → groupId: "grp-a1b2c3"
#
# security_groups field uses stable IDs:
#   ["milvus:doc:grp-a1b2c3", "milvus:doc:grp-x9y8z7"]
#
# If group is renamed in LDAP, ID stays constant → no doc updates needed

# Local mapping table (optional, for display):
CREATE TABLE group_aliases (
    group_id VARCHAR(64) PRIMARY KEY,      -- "grp-a1b2c3"
    canonical_name VARCHAR(256),           -- "milvus:doc:grp-a1b2c3"
    display_name VARCHAR(256),             -- "Legal Team"
    ldap_dn VARCHAR(512)
);
```

### Guaranteed Filter Application

```python
class SecureMilvusClient:
    """
    Wrapper that guarantees doc filter is always applied.
    The raw milvus client is private and inaccessible.
    """
    
    def __init__(self, milvus_client):
        self.__milvus = milvus_client  # Name-mangled, not directly accessible
    
    def search(
        self,
        collection: str,
        vectors: list,
        mandatory_filter: str,  # Required parameter
        **kwargs
    ):
        if not mandatory_filter or mandatory_filter.strip() == "":
            raise SecurityViolation("Document filter is required")
        
        # Reject any attempt to pass expr in kwargs
        if "expr" in kwargs:
            raise SecurityViolation("Cannot override mandatory filter")
        
        return self.__milvus.search(
            collection_name=collection,
            data=vectors,
            expr=mandatory_filter,
            **kwargs
        )
```

---

## 7) Write-Path + Preventing Privilege Escalation

### Who Can Write

| Permission | Can Insert | Can Update | Can Delete |
|------------|------------|------------|------------|
| `admin` | ✓ | ✓ | ✓ |
| `rw` | ✓ | ✓ (own docs) | ✓ (own docs) |
| `r` | ✗ | ✗ | ✗ |
| `none` | ✗ | ✗ | ✗ |

### Allowed-to-Assign Model

**Problem**: A writer with `milvus:doc:public` membership should not be able to tag docs with `milvus:doc:confidential-hr` (escalation).

**Solution**: Separate tagging permissions.

```
To assign security_groups value X to a document, user must have:
  1. Write permission on the collection (rw or admin), AND
  2. Membership in the tagging group: milvus:tag:X
     OR
     admin permission on the collection (admins can assign any group)
```

### Validation Implementation

```python
def validate_security_groups_assignment(
    username: str,
    user_groups: set[str],
    collection: str,
    doc: dict
):
    """
    Validate that user is allowed to assign the given security_groups.
    Called before insert/upsert.
    """
    doc_security_groups = doc.get("security_groups", [])
    
    # Deny empty security_groups (would create inaccessible doc)
    if not doc_security_groups:
        raise HTTPException(400, 
            "security_groups is required and cannot be empty")
    
    # Check each group assignment
    is_admin = f"milvus:{collection}:admin" in user_groups
    
    for group in doc_security_groups:
        # Validate format
        if not group.startswith("milvus:doc:"):
            raise HTTPException(400, 
                f"Invalid security_group format: {group}")
        
        # Check assignment permission
        tag_permission = f"milvus:tag:{group.removeprefix('milvus:doc:')}"
        
        if not is_admin and tag_permission not in user_groups:
            log.warning("unauthorized_tag_attempt",
                user=username,
                attempted_group=group,
                has_tag_perm=False
            )
            raise HTTPException(403,
                f"Not authorized to assign group: {group}")
    
    # Ensure user can read what they're creating (sanity check)
    user_doc_groups = {g for g in user_groups if g.startswith("milvus:doc:")}
    if not user_doc_groups.intersection(doc_security_groups):
        raise HTTPException(400,
            "You must be able to read documents you create")
```

### Update Flow (Changing Doc ACL)

```python
def update_document_acl(
    username: str,
    user_groups: set[str],
    collection: str,
    doc_id: str,
    new_security_groups: list[str]
):
    """
    Update security_groups on existing document.
    """
    # 1. Verify collection write access
    require_collection_access(username, user_groups, collection, "rw")
    
    # 2. Verify user can currently read the doc (ownership check)
    doc_filter = build_doc_filter(user_groups)
    existing = milvus.query(
        collection_name=collection,
        expr=f'id == "{escape(doc_id)}" and ({doc_filter})',
        output_fields=["id", "security_groups"]
    )
    
    if not existing:
        # Either doesn't exist or user can't read it
        raise HTTPException(404, "Document not found")
    
    # 3. Validate new assignments
    validate_security_groups_assignment(
        username, user_groups, collection,
        {"security_groups": new_security_groups}
    )
    
    # 4. Perform update (Milvus upsert or delete+insert)
    # Note: Milvus doesn't support partial updates; must re-insert
    old_doc = existing[0]
    old_doc["security_groups"] = new_security_groups
    old_doc["updated_at"] = int(time.time())
    
    milvus.upsert(collection_name=collection, data=[old_doc])
    
    log.info("acl_updated",
        user=username,
        collection=collection,
        doc_id=doc_id,
        old_groups=existing[0]["security_groups"],
        new_groups=new_security_groups
    )
```

---

## 8) Data/Schema Recommendations

### Milvus Collection Schema

```python
from pymilvus import CollectionSchema, FieldSchema, DataType

def create_rag_collection_schema(collection_name: str) -> CollectionSchema:
    fields = [
        # Primary key
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=64
        ),
        
        # Vector embedding
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=1536  # Adjust for your model
        ),
        
        # Document content reference
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=65535
        ),
        
        # Source reference (external doc store)
        FieldSchema(
            name="source_uri",
            dtype=DataType.VARCHAR,
            max_length=1024
        ),
        
        # RBAC: document-level access control
        FieldSchema(
            name="security_groups",
            dtype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=50,     # Max groups per doc
            max_length=128       # Max length per group name
        ),
        
        # Optional: tenant isolation (if multi-tenant)
        FieldSchema(
            name="tenant_id",
            dtype=DataType.VARCHAR,
            max_length=64
        ),
        
        # Timestamps
        FieldSchema(name="created_at", dtype=DataType.INT64),
        FieldSchema(name="updated_at", dtype=DataType.INT64),
        
        # Metadata
        FieldSchema(
            name="metadata",
            dtype=DataType.JSON
        ),
    ]
    
    return CollectionSchema(
        fields=fields,
        description=f"RAG collection: {collection_name}",
        enable_dynamic_field=False  # Strict schema
    )
```

### Local Database Schema (PostgreSQL)

```sql
-- User group cache (can also use Redis, but this is for persistence/audit)
CREATE TABLE user_group_cache (
    username VARCHAR(256) PRIMARY KEY,
    groups JSONB NOT NULL,                    -- ["milvus:doc:x", "milvus:contracts:r", ...]
    fetched_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    ldap_dn VARCHAR(512)
);

CREATE INDEX idx_user_group_cache_expires ON user_group_cache(expires_at);

-- Collection policy (authoritative mapping of collections to required groups)
CREATE TABLE collection_policy (
    collection_name VARCHAR(256) PRIMARY KEY,
    admin_groups JSONB NOT NULL,              -- ["milvus:contracts:admin"]
    rw_groups JSONB NOT NULL,                 -- ["milvus:contracts:rw"]
    r_groups JSONB NOT NULL,                  -- ["milvus:contracts:r"]
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Alternative: derive from naming convention, no table needed

-- Group alias mapping (stable IDs ↔ names)
CREATE TABLE group_registry (
    group_id VARCHAR(64) PRIMARY KEY,         -- "grp-a1b2c3"
    canonical_name VARCHAR(256) UNIQUE,       -- "milvus:doc:grp-a1b2c3"
    display_name VARCHAR(256),                -- "Legal Team"
    ldap_dn VARCHAR(512),                     -- "cn=legal-team,ou=groups,dc=corp"
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Audit log
CREATE TABLE rbac_audit_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    username VARCHAR(256) NOT NULL,
    action VARCHAR(64) NOT NULL,              -- "search", "insert", "delete", "acl_change"
    collection VARCHAR(256),
    doc_id VARCHAR(64),
    groups_hash VARCHAR(32),                  -- First 16 chars of SHA256
    filter_applied TEXT,
    result_count INT,
    denied BOOLEAN DEFAULT FALSE,
    denial_reason VARCHAR(256),
    client_ip INET,
    request_id VARCHAR(64)
);

CREATE INDEX idx_audit_timestamp ON rbac_audit_log(timestamp);
CREATE INDEX idx_audit_username ON rbac_audit_log(username);
CREATE INDEX idx_audit_collection ON rbac_audit_log(collection);
```

### Example Records

```sql
-- user_group_cache
INSERT INTO user_group_cache VALUES (
    'alice',
    '["milvus:contracts:rw", "milvus:hr_docs:r", "milvus:doc:legal-team", "milvus:tag:legal-team"]',
    '2024-01-15 10:30:00',
    '2024-01-15 10:35:00',
    'uid=alice,ou=users,dc=corp'
);

-- collection_policy
INSERT INTO collection_policy VALUES (
    'contracts',
    '["milvus:contracts:admin"]',
    '["milvus:contracts:rw"]',
    '["milvus:contracts:r"]',
    NOW(), NOW()
);

-- group_registry
INSERT INTO group_registry VALUES (
    'grp-legal-001',
    'milvus:doc:legal-team',
    'Legal Team',
    'cn=legal-team,ou=groups,dc=corp',
    TRUE,
    NOW()
);
```

### Redis Cache Structure

```python
# Key patterns for Redis cache

# User groups (primary cache)
# Key: ldap:groups:{username}
# Value: JSON array of group names
# TTL: 300 seconds

redis.setex(
    "ldap:groups:alice",
    300,
    '["milvus:contracts:rw","milvus:doc:legal-team"]'
)

# Optional: Group membership reverse index (for bulk revocation)
# Key: ldap:group_members:{group_name}
# Value: Set of usernames
# TTL: 3600 seconds (less critical, longer TTL ok)

redis.sadd("ldap:group_members:milvus:doc:legal-team", "alice", "bob")
redis.expire("ldap:group_members:milvus:doc:legal-team", 3600)
```

---

## 9) Revocation, Auditing, and Operations

### Revocation Timeline

```
Event: Admin removes 'alice' from 'milvus:doc:legal-team' in LDAP

Timeline:
  T+0s     : LDAP updated
  T+0-300s : Alice's cached groups still include legal-team (stale)
  T+300s   : Cache TTL expires
  T+300s+  : Next request triggers fresh LDAP lookup
  T+300s+  : Alice's queries no longer return legal-team docs

Worst case revocation: 300 seconds (cache TTL)
```

### Token/Session Strategy

```python
class SessionManager:
    """
    Even with LDAP tokens, re-validate groups periodically.
    Tokens prove identity; groups prove authorization.
    """
    
    def __init__(self, cache, ldap):
        self.cache = cache
        self.ldap = ldap
        self.GROUP_REFRESH_INTERVAL = 300  # seconds
    
    def get_session(self, token: str) -> Session:
        # Validate token (proves identity)
        username = self.ldap.validate_token(token)
        if not username:
            raise Unauthorized()
        
        # Always check groups freshness
        session_key = f"session:{token}"
        session = self.cache.get(session_key)
        
        if session:
            session = Session.from_json(session)
            # Re-fetch groups if stale
            if session.groups_fetched_at < time.time() - self.GROUP_REFRESH_INTERVAL:
                session.groups = self.fetch_fresh_groups(username)
                session.groups_fetched_at = time.time()
                self.cache.set(session_key, session.to_json(), ex=3600)
        else:
            # New session
            groups = self.fetch_fresh_groups(username)
            session = Session(
                username=username,
                groups=groups,
                groups_fetched_at=time.time()
            )
            self.cache.set(session_key, session.to_json(), ex=3600)
        
        return session
```

### Audit Logging

```python
import structlog
import hashlib

log = structlog.get_logger()

def audit_log(
    action: str,
    username: str,
    collection: str | None = None,
    groups: set[str] | None = None,
    filter_applied: str | None = None,
    result_count: int | None = None,
    denied: bool = False,
    denial_reason: str | None = None,
    doc_id: str | None = None,
    request_id: str | None = None,
    client_ip: str | None = None
):
    """Emit structured audit log."""
    
    groups_hash = None
    if groups:
        groups_hash = hashlib.sha256(
            str(sorted(groups)).encode()
        ).hexdigest()[:16]
    
    log.info(
        "rbac_audit",
        action=action,
        username=username,
        collection=collection,
        doc_id=doc_id,
        groups_hash=groups_hash,
        filter_applied=filter_applied,
        result_count=result_count,
        denied=denied,
        denial_reason=denial_reason,
        request_id=request_id,
        client_ip=client_ip,
        timestamp=datetime.utcnow().isoformat()
    )
    
    # Also persist to DB for compliance
    db.execute("""
        INSERT INTO rbac_audit_log 
        (username, action, collection, doc_id, groups_hash, 
         filter_applied, result_count, denied, denial_reason,
         client_ip, request_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (username, action, collection, doc_id, groups_hash,
          filter_applied, result_count, denied, denial_reason,
          client_ip, request_id))
```

### LDAP Downtime Handling

```python
class ResilientLDAPClient:
    def __init__(self, primary_ldap, fallback_ldap=None):
        self.primary = primary_ldap
        self.fallback = fallback_ldap
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )
    
    def get_user_groups(self, username: str) -> list[str]:
        # Try primary
        try:
            if self.circuit_breaker.is_closed():
                groups = self.primary.get_groups(username, timeout=3.0)
                self.circuit_breaker.record_success()
                return groups
        except LDAPError as e:
            self.circuit_breaker.record_failure()
            log.warning("ldap_primary_failed", error=str(e))
        
        # Try fallback if configured
        if self.fallback:
            try:
                return self.fallback.get_groups(username, timeout=3.0)
            except LDAPError:
                log.warning("ldap_fallback_failed")
        
        # FAIL CLOSED - no cached fallback during outage
        raise LDAPUnavailable("All LDAP servers unavailable")
```

### Runbook: Onboarding New Collection

```markdown
## Runbook: Create New Milvus Collection with RBAC

### Prerequisites
- [ ] Collection name approved (lowercase, alphanumeric + hyphens)
- [ ] LDAP groups created:
  - `milvus:{collection}:admin`
  - `milvus:{collection}:rw`
  - `milvus:{collection}:r`
- [ ] Initial users assigned to groups in LDAP

### Steps

1. **Create LDAP Groups**
   ```bash
   ldapadd -x -D "cn=admin,dc=corp" -W <<EOF
   dn: cn=milvus:newcollection:admin,ou=groups,dc=corp
   objectClass: groupOfNames
   cn: milvus:newcollection:admin
   member: uid=admin_carol,ou=users,dc=corp

   dn: cn=milvus:newcollection:rw,ou=groups,dc=corp
   objectClass: groupOfNames
   cn: milvus:newcollection:rw
   member: uid=alice,ou=users,dc=corp

   dn: cn=milvus:newcollection:r,ou=groups,dc=corp
   objectClass: groupOfNames
   cn: milvus:newcollection:r
   member: uid=bob,ou=users,dc=corp
   EOF
   ```

2. **Create Milvus Collection**
   ```python
   from app.milvus_admin import create_collection
   
   # Must be run by app admin (service account or admin user)
   create_collection(
       name="newcollection",
       schema=create_rag_collection_schema("newcollection"),
       index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 1024}}
   )
   ```

3. **Register Collection Policy** (if using policy table)
   ```sql
   INSERT INTO collection_policy VALUES (
       'newcollection',
       '["milvus:newcollection:admin"]',
       '["milvus:newcollection:rw"]', 
       '["milvus:newcollection:r"]',
       NOW(), NOW()
   );
   ```

4. **Verify Access**
   ```bash
   # As alice (rw)
   curl -H "Authorization: Bearer $ALICE_TOKEN" \
        -X POST /api/collections/newcollection/search \
        -d '{"query": "test"}'
   # Should return empty results (no docs yet)

   # As eve (no access)
   curl -H "Authorization: Bearer $EVE_TOKEN" \
        -X POST /api/collections/newcollection/search \
        -d '{"query": "test"}'
   # Should return 403
   ```

5. **Document in Wiki**
   - Collection purpose
   - Owner team
   - Associated LDAP groups
   - Data classification level
```

### Runbook: Onboarding New Document Group

```markdown
## Runbook: Create New Document Security Group

### Steps

1. **Create LDAP Group**
   ```bash
   ldapadd -x -D "cn=admin,dc=corp" -W <<EOF
   dn: cn=milvus:doc:project-gamma,ou=groups,dc=corp
   objectClass: groupOfNames
   cn: milvus:doc:project-gamma
   member: uid=alice,ou=users,dc=corp
   member: uid=bob,ou=users,dc=corp
   EOF
   ```

2. **Create Tagging Permission Group**
   ```bash
   ldapadd -x -D "cn=admin,dc=corp" -W <<EOF
   dn: cn=milvus:tag:project-gamma,ou=groups,dc=corp
   objectClass: groupOfNames
   cn: milvus:tag:project-gamma
   member: uid=alice,ou=users,dc=corp
   EOF
   ```

3. **Register Group** (optional, for tracking)
   ```sql
   INSERT INTO group_registry VALUES (
       'grp-gamma-001',
       'milvus:doc:project-gamma',
       'Project Gamma Team',
       'cn=milvus:doc:project-gamma,ou=groups,dc=corp',
       TRUE, NOW()
   );
   ```

4. **Users Can Now**
   - Alice: Create docs with `security_groups: ["milvus:doc:project-gamma"]`
   - Alice & Bob: Read docs tagged with `milvus:doc:project-gamma`
   - Bob: Cannot tag new docs (no tagging permission)
```

---

## 10) Concrete Examples

### Example LDAP Groups Setup

**Collection: `contracts`**
```
cn=milvus:contracts:admin,ou=groups,dc=corp
  members: admin_carol

cn=milvus:contracts:rw,ou=groups,dc=corp
  members: alice

cn=milvus:contracts:r,ou=groups,dc=corp
  members: bob, charlie
```

**Collection: `hr_docs`**
```
cn=milvus:hr_docs:admin,ou=groups,dc=corp
  members: admin_carol, hr_manager

cn=milvus:hr_docs:rw,ou=groups,dc=corp
  members: hr_specialist

cn=milvus:hr_docs:r,ou=groups,dc=corp
  members: (empty - no general read access)
```

**Document Groups**
```
cn=milvus:doc:legal-team,ou=groups,dc=corp
  members: alice, legal_counsel

cn=milvus:doc:finance-team,ou=groups,dc=corp
  members: bob, cfo

cn=milvus:doc:hr-confidential,ou=groups,dc=corp
  members: hr_manager, hr_specialist

cn=milvus:doc:all-employees,ou=groups,dc=corp
  members: alice, bob, charlie, ...everyone
```

### Example Document `security_groups` Values

```python
# Contract visible to legal team only
doc1 = {
    "id": "contract-001",
    "text": "Confidential merger agreement...",
    "embedding": [...],
    "security_groups": ["milvus:doc:legal-team"],
}

# Financial report visible to finance and legal
doc2 = {
    "id": "finance-q4-2024",
    "text": "Q4 financial results...",
    "embedding": [...],
    "security_groups": ["milvus:doc:finance-team", "milvus:doc:legal-team"],
}

# Company-wide announcement
doc3 = {
    "id": "announcement-001",
    "text": "Holiday schedule...",
    "embedding": [...],
    "security_groups": ["milvus:doc:all-employees"],
}

# HR confidential - only HR can see
doc4 = {
    "id": "hr-salary-bands",
    "text": "2024 salary bands...",
    "embedding": [...],
    "security_groups": ["milvus:doc:hr-confidential"],
}
```

### Example Request + Derived Filter

**Scenario**: Alice searches the `contracts` collection

```python
# Alice's LDAP groups:
alice_groups = {
    "milvus:contracts:rw",      # Collection: read-write
    "milvus:doc:legal-team",    # Doc group
    "milvus:tag:legal-team",    # Can tag legal docs
}

# Request
request = {
    "collection": "contracts",
    "query_vector": [0.1, 0.2, ...],
    "top_k": 10
}

# Step 1: Resolve user
username = "alice"

# Step 2: Fetch groups (from cache or LDAP)
groups = alice_groups

# Step 3: Check collection permission
# "milvus:contracts:rw" in groups → permission = "rw" ≥ "r" ✓

# Step 4: Build doc filter
doc_groups = ["milvus:doc:legal-team"]
filter_expr = 'array_contains_any(security_groups, ["milvus:doc:legal-team"])'

# Step 5: Execute Milvus search
results = milvus.search(
    collection_name="contracts",
    data=[[0.1, 0.2, ...]],
    expr='array_contains_any(security_groups, ["milvus:doc:legal-team"])',
    limit=10
)

# Returns: doc1, doc2 (both have legal-team)
# Filtered out: doc3 (all-employees - alice not in that group in this collection)
```

### Example Denied Scenarios

**Scenario A: Collection Denied**

```python
# Bob's groups:
bob_groups = {
    "milvus:contracts:r",       # Read-only on contracts
    "milvus:doc:finance-team",  # Doc group
    # No hr_docs permission!
}

# Bob tries to search hr_docs
request = {"collection": "hr_docs", "query_vector": [...]}

# Step 3: Check collection permission
# No "milvus:hr_docs:*" in bob_groups → permission = "none"

# Result: HTTP 403 "Access denied"
# Log: access_denied, user=bob, collection=hr_docs, required=r, actual=none
```

**Scenario B: Document Filtered Out**

```python
# Charlie's groups:
charlie_groups = {
    "milvus:contracts:r",       # Read-only on contracts
    "milvus:doc:all-employees", # Only general access
}

# Charlie searches contracts
request = {"collection": "contracts", "query_vector": [...]}

# Step 3: Collection check passes (has "r")

# Step 4: Build filter
filter_expr = 'array_contains_any(security_groups, ["milvus:doc:all-employees"])'

# Step 5: Execute
# Returns: doc3 only (all-employees)
# doc1, doc2 filtered out (require legal-team or finance-team)

# Charlie never knows doc1, doc2 exist
```

**Scenario C: Write Privilege Escalation Blocked**

```python
# Bob tries to insert a doc with legal-team tag
bob_groups = {
    "milvus:contracts:r",       # Read-only! Not rw
    "milvus:doc:finance-team",
}

request = {
    "collection": "contracts",
    "document": {
        "id": "new-doc",
        "text": "...",
        "security_groups": ["milvus:doc:legal-team"]  # Bob shouldn't tag this
    }
}

# Step 1: Check collection permission for "rw"
# bob has "r" only → HTTP 403 "Access denied"

# Even if Bob had rw:
bob_groups_v2 = {
    "milvus:contracts:rw",
    "milvus:doc:finance-team",
    # No milvus:tag:legal-team!
}

# Step 2 (if rw check passed): Validate security_groups assignment
# "milvus:doc:legal-team" requires "milvus:tag:legal-team"
# Bob doesn't have it → HTTP 403 "Not authorized to assign group: milvus:doc:legal-team"
```

---

## Summary: Security Checklist

| # | Check | Implementation |
|---|-------|----------------|
| 1 | User identity verified | LDAP token validation |
| 2 | Groups fetched server-side | LDAP lookup with bounded cache |
| 3 | Collection permission checked | App-level gate before Milvus call |
| 4 | Doc filter always applied | Centralized query builder, mandatory param |
| 5 | Filter constructed server-side | Never from client input |
| 6 | Write validation enforced | security_groups assignment check |
| 7 | Deny-by-default | Empty groups = no access; missing field = no access |
| 8 | Revocation bounded | Cache TTL = 300s max staleness |
| 9 | LDAP outage = fail closed | No stale cache fallback |
| 10 | Full audit trail | Structured logs + DB persistence |





## 1) Threat model + security invariants

### Assets
- **Embeddings + chunk text** (or pointers to chunk text) stored in Milvus.
- **Metadata**, especially `security_groups`, doc IDs, source URIs, and any tenant/org identifiers.
- **Collection existence** and configuration (names, schema, index params).
- **LDAP identity + group membership**.

### Trust boundaries
- **Client is untrusted/malicious**: may omit/alter filters, request arbitrary collections, attempt prompt injection, etc.
- **Application server is the policy enforcement point (PEP)**.
- **LDAP is the source of truth** for identity and group membership.
- **Milvus is not a policy authority** in this design; it executes searches with filters you provide.

### “Must never happen” (explicit)
1. **Unauthorized document returned**
   - A chunk/doc whose `security_groups` does **not** intersect the user’s effective LDAP groups must never be returned.
2. **Filter bypass or omission**
   - No code path may issue a Milvus `search/query/get` without a server-constructed mandatory filter.
   - Client-supplied filter expressions must never be passed through.
3. **Privilege escalation via writes**
   - A writer must not be able to write documents tagged with broader/more privileged `security_groups` than they are authorized to assign.
   - A writer must not change an existing doc’s ACL to grant access to additional groups unless explicitly authorized.
4. **Inference/leakage**
   - No leaking via:
     - error messages (“group X not allowed”, “collection exists but unauthorized”),
     - counts/aggregations (“you have 0 docs in collection” vs “collection not found”),
     - timing side-channels (fast reject vs slow filtered search),
     - metadata fields (returning `security_groups`, internal IDs, or other sensitive attributes).
5. **Deny-by-default**
   - Missing/empty ACL metadata must **not** accidentally become public.
6. **Revocation must work**
   - When LDAP removes a user from a group, access must be cut off within a defined max staleness window; during uncertainty (LDAP down), fail closed (or tightly bounded grace with compensating controls).

### What is enforced where
- **Application enforces:**
  - Collection-level RBAC (`admin/rw/r/none`) on every operation.
  - Construction and mandatory attachment of doc-level filter based on LDAP groups.
  - Write-path validation of `security_groups` assignments.
  - Response shaping (no sensitive metadata, no revealing errors).
- **Milvus enforces (as execution engine):**
  - Document-level filtering only insofar as it correctly applies the **filter expression** the app passes (e.g., `expr=...`).

Security invariant to code against:
- **Every Milvus read query must include an ACL filter AND must only be reachable after collection-level `r` check.**
- **Every Milvus write must only be reachable after collection-level `rw/admin` check AND must validate ACL/tagging policy.**

---

## 2) Authorization model (precise)

### Collection permission model
Per collection \(C\), user \(U\) has exactly one effective permission:
- `admin`: full control (read/write + schema management if you expose it)
- `rw`: read + write (insert/upsert/delete/update ACL subject to tagging rules)
- `r`: read-only (search/query)
- `none`: no access

Rules:
- `admin` ⊇ `rw` ⊇ `r` ⊇ `none`
- Deny by default: if no rule matches, permission is `none`.

### Document permission model
Each document/chunk row stored in Milvus must include:
- `security_groups`: **array/set of group identifiers** (strings or stable IDs)

Read rule:
- doc is readable iff:
  - `security_groups` intersects `effective_groups(U)`:
    - \(security\_groups(doc) \cap groups(U) \neq \emptyset\)

Missing/empty `security_groups`:
- **Default: deny**
  - If `security_groups` is missing, null, or empty array => not readable by anyone unless you *explicitly* define a “public” group and still store it as a group entry.
- Recommended explicit public access:
  - Use a real group value, e.g. `doc:public`, and include it in docs intended to be public. Do **not** treat empty as public.

### Mapping LDAP groups → collection permissions and doc ACL groups

#### Proposed naming conventions (recommended)
Use two namespaces:

1) **Collection RBAC groups** (drives `admin/rw/r`):
- `milvus:{collection}:admin`
- `milvus:{collection}:rw`
- `milvus:{collection}:r`

2) **Document ACL groups** (values stored in `security_groups`):
- `doc:{domain}:{name}` (stable semantic groups)
  - Examples: `doc:finance:payroll`, `doc:eng:platform`, `doc:legal:contracts`

Why separate namespaces:
- Collection RBAC is about *ability to use the collection at all*.
- Doc ACL groups are about *which content within the collection is visible*.
- Keeping them separate avoids accidental “having r on collection implies doc group membership” or vice versa.

#### Example mapping table
Assume collections: `hr_policies`, `eng_runbooks`

| LDAP group | Meaning | Collection perm effect | Doc ACL use |
|---|---|---|---|
| `milvus:hr_policies:r` | can search HR policies collection | `hr_policies` => `r` | none |
| `milvus:hr_policies:rw` | can write to HR policies | `hr_policies` => `rw` | none |
| `milvus:eng_runbooks:admin` | full admin for eng runbooks | `eng_runbooks` => `admin` | none |
| `doc:finance:payroll` | can read payroll-related docs (any collection that uses it) | none | appears in `security_groups` |
| `doc:eng:platform` | platform docs access | none | appears in `security_groups` |

Effective policy:
- To read from collection `C`: must have `milvus:C:r` (or `rw/admin`), **and** returned docs must match doc ACL filter derived from `doc:*` groups (or whatever set you choose).

---

## 3) LDAP sync strategy (choose + justify)

You need both correctness (revocation) and performance. Two viable strategies:

### Strategy A: On-demand LDAP group lookup (per request/session) + caching
- On each request (or session start), query LDAP for user’s groups.
- Cache results (in-memory + shared cache like Redis).

Pros:
- Simple, minimal moving parts.
- Near-real-time revocation if TTL is short.

Cons:
- LDAP load at high QPS.
- Outage handling becomes critical (must fail closed or bounded grace).

### Strategy B: Periodic sync from LDAP into local DB/cache
- Background job syncs users/groups and policies into a local store.
- Requests only hit local store.

Pros:
- Decouples request latency from LDAP.
- Better for high QPS.

Cons:
- Revocation lag unless you sync very frequently.
- Requires correct incremental sync, tombstones, and monitoring.

### Strategy C (recommended): Hybrid — on-demand with short TTL + “revocation-aware” periodic refresh
Recommended given “security and revocation” constraints:
- **On-demand fetch** on cache miss/expiry for the requesting user.
- **Short TTL** to bound staleness.
- **Background refresh** for hot users and/or periodic validation to reduce LDAP load.
- Optional: ingest LDAP change notifications (if your directory supports it) to proactively invalidate caches.

#### Concrete recommended parameters
- Cache TTL: **5 minutes** (default)
- Soft TTL (serve-stale-while-refresh): **1 minute** (optional)
- Maximum acceptable staleness window: **≤ 5 minutes**
- Revocation:
  - If user removed from a group in LDAP, access is cut off at next cache refresh, worst-case 5 minutes.
  - If you can consume LDAP change events: invalidate immediately (best).
- LDAP outage behavior:
  - **Fail closed by default**: if you cannot confirm groups for a request and cache is expired, deny.
  - Optional controlled grace: if cache is still within TTL and LDAP is down, you may serve using cached groups **only until TTL**, then deny. Never extend beyond TTL.
- Performance considerations:
  - Store groups as compact IDs (see section 6) to keep filter size manageable.
  - Use Redis or similar shared cache for multi-instance app servers.
  - Track per-user group cardinality; very large memberships may require compression (roles, hashing).

---

## 4) Enforcement flow (end-to-end request path)

### Step-by-step sequence for a read/search request
1) **Resolve username**
   - If request has LDAP username: accept (after auth).
   - If request has LDAP token: resolve token → username via LDAP/introspection endpoint.
   - Normalize username (case, realm) to canonical form.

2) **Fetch effective LDAP groups**
   - `groups = get_groups(username)`:
     - Check shared cache (Redis) first.
     - If miss/expired: query LDAP, compute `effective_groups`, write cache with TTL.
   - Compute:
     - `collection_roles`: membership in `milvus:{collection}:{r|rw|admin}`
     - `doc_groups`: membership in `doc:*` (or your chosen namespace)

3) **Collection-level permission check**
   - Determine requested operation (`search`, `query`, `insert`, `delete`, etc).
   - Compute `perm = collection_permission(username, collection, groups)`
   - If perm insufficient: deny (generic error).

4) **Build mandatory doc-level filter**
   - From `doc_groups` (and optionally allow `doc:public` if you use it).
   - Construct Milvus filter expression only from server-side data.

5) **Execute Milvus search/query with mandatory filter**
   - Use the single shared Milvus service credential.
   - Always pass `expr=<mandatory_filter>` (or equivalent) in the Milvus call.
   - Never merge client filters; if you allow “extra filters”, AND them with mandatory filter server-side.

6) **Post-process + return**
   - Strip sensitive fields (`security_groups`, internal partition keys, etc.).
   - Enforce top-k bounds and rate limits.
   - Emit audit logs.

### ASCII sequence diagram
```text
Client
  |
  | 1) Request (collection=C, query=..., token/user)
  v
API Server (PEP)
  |-- 1a) Resolve token -> username (LDAP)
  |
  |-- 2) Get groups(username)
  |       |-- check Redis cache
  |       |-- (miss) query LDAP -> groups -> cache(TTL=5m)
  |
  |-- 3) Check collection RBAC (milvus:C:r/rw/admin)
  |       |-- if deny -> generic 403
  |
  |-- 4) Build mandatory expr from doc groups
  |
  |-- 5) Milvus search(collection=C, expr=MANDATORY_ACL_EXPR, vector=...)
  v
Milvus (shared credential)
  |
  |-- applies expr filter during search
  v
API Server
  |
  |-- 6) sanitize results + audit log
  v
Client
```

---

## 5) How to enforce collection-level RBAC (with “tightness”)

“Tightness” requirement: **doc-level access never implies collection read**. Therefore:
- You must check collection permission **before** any Milvus operation, regardless of doc groups.

### Define operation checks
Let `perm ∈ {admin, rw, r, none}`.

#### Read operations (require at least `r`)
- `search` (vector similarity): require `perm ∈ {r, rw, admin}`
- `query` (metadata query): require `perm ∈ {r, rw, admin}` and still apply doc ACL filter
- `get by id` / `fetch`: require `perm ∈ {r, rw, admin}` and apply doc ACL filter (never fetch raw without expr)

#### Write operations (require `rw` or `admin`)
- `insert` / `upsert`: require `perm ∈ {rw, admin}` plus tagging validation (section 7)
- `delete`: require `perm ∈ {rw, admin}` plus scope rule (only delete docs you are allowed to affect; see section 7)
- `update ACL/metadata`: require `perm ∈ {rw, admin}` plus strict ACL change rules

#### Admin operations (require `admin`)
(Only if your app exposes them; otherwise keep internal)
- `create collection`, `drop collection`, `create index`, `load/release`, schema changes:
  - require `perm = admin` for that collection (or a global admin group)
  - recommended: **do not expose these in the RAG API**; handle via infra pipeline.

### Where collection permissions come from
Recommended primary approach: **direct group naming convention** in LDAP:
- `milvus:{collection}:r`
- `milvus:{collection}:rw`
- `milvus:{collection}:admin`

Decision logic:
- if in `admin` group => `admin`
- else if in `rw` group => `rw`
- else if in `r` group => `r`
- else => `none`

Alternative (if you need central policy): `collection_policy` table mapping collection → required LDAP group(s) for each level. Still deny-by-default.

---

## 6) How to enforce document-level RBAC in Milvus (mechanically)

### Schema approach for `security_groups`
Use an **array field** if your Milvus version supports array + membership operations efficiently; otherwise store as:
- `security_group_ids`: array<int64> (preferred for compact filters)
or
- `security_groups`: array<string> (simpler, larger filter payload)

**Recommendation:** use **stable numeric IDs** for groups to:
- reduce filter expression size,
- avoid rename issues,
- speed comparisons.

Maintain a `group_alias` mapping (name → id), see section 8.

### Building the filter expression
You need “any overlap” semantics:
- readable iff at least one of the user’s group IDs is in the doc’s group IDs array.

Expression patterns depend on Milvus expr features available in your deployment. Common patterns:
- `array_contains_any(security_group_ids, [1,2,3])`
- or OR-chaining: `array_contains(security_group_ids, 1) OR array_contains(security_group_ids, 2) ...`

If your Milvus does not support `array_contains_any`, you’ll generate OR-chains with a hard cap and mitigation (below).

#### Mandatory filter must also deny missing/empty
If empty arrays are possible, ensure the expression doesn’t match them accidentally. Typically overlap checks won’t match empty anyway, but explicitly deny null/missing if relevant.

### Large group lists: limits + mitigations
Problems:
- LDAP users may have hundreds/thousands of groups → huge filter strings → latency and possible expr limits.
Mitigations (recommended in order):
1) **Use stable numeric IDs** (shrinks payload).
2) **Role compression**: instead of raw LDAP groups, derive a smaller set of “doc ACL groups” relevant to your app (e.g., only `doc:*` namespace).
3) **Precompute effective group IDs** and cache them.
4) **Group hashing is not sufficient alone** (collisions). Only use hashing if you also include collision-safe salt+namespace and accept complexity; better to use stable IDs.
5) **Hard cap**:
   - If `doc_group_ids` exceeds a threshold (e.g., 500), require the user be assigned to coarser roles or move to a policy redesign; do not silently truncate (truncation becomes a security bug).

### Group renames/deletes
- Store **IDs** in documents; maintain alias table from LDAP group name → stable ID.
- On rename: update alias only; no doc rewrite needed.
- On delete: stop issuing that ID in user effective groups; docs tagged with that ID become unreadable (safe default).

### Cross-collection reuse of doc groups
- Works naturally if `doc:*` groups are global semantics.
- Still require collection `r` first (tightness constraint).

### Pseudocode (Python) — centralized query builder that cannot be skipped
Key design: **there is exactly one internal function** that can call Milvus for reads, and it *always* injects the ACL expr.

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass(frozen=True)
class AuthContext:
    username: str
    effective_groups: List[str]          # raw LDAP group names
    doc_group_ids: List[int]             # derived stable IDs for doc ACL
    groups_hash: str                     # for audit logs
    collection_perm: str                 # "admin" | "rw" | "r" | "none"


class AuthorizationError(Exception):
    pass


def require_collection_read(ctx: AuthContext) -> None:
    if ctx.collection_perm not in ("r", "rw", "admin"):
        raise AuthorizationError("forbidden")


def build_acl_expr_from_group_ids(group_ids: List[int]) -> str:
    # Deny-by-default: if no doc groups, the user can read nothing.
    if not group_ids:
        return "false"

    # Prefer array_contains_any if supported; otherwise OR-chain.
    # Keep this as the only place that knows the Milvus expression syntax.
    ids = ",".join(str(i) for i in group_ids)
    return f"array_contains_any(security_group_ids, [{ids}])"


def combine_expr(mandatory_acl_expr: str, extra_expr: Optional[str]) -> str:
    # Client must never provide expr; extra_expr is server-generated only.
    if extra_expr is None or extra_expr.strip() == "":
        return mandatory_acl_expr
    return f"({mandatory_acl_expr}) and ({extra_expr})"


def milvus_search_secured(
    milvus_client,
    collection: str,
    ctx: AuthContext,
    vector: List[float],
    top_k: int,
    server_extra_expr: Optional[str] = None,
):
    require_collection_read(ctx)

    acl_expr = build_acl_expr_from_group_ids(ctx.doc_group_ids)
    expr = combine_expr(acl_expr, server_extra_expr)

    # Enforce sane bounds (avoid inference + resource abuse)
    top_k = min(max(top_k, 1), 50)

    return milvus_client.search(
        collection_name=collection,
        data=[vector],
        limit=top_k,
        expr=expr,
        output_fields=["doc_id", "chunk_id", "source", "score_hint"],
    )
```

Non-bypassable patterns to implement:
- Do **not** expose raw Milvus client to handlers.
- Make secured functions the only import path in your codebase (enforced by code review + static checks).
- Add runtime assertions: if `expr` doesn’t contain your ACL field name, fail the request.

---

## 7) Write-path + preventing privilege escalation

### Who can write
- Require `collection_perm ∈ {rw, admin}` for:
  - insert/upsert
  - delete
  - update metadata / ACL

### Core risk: writer assigns broader `security_groups`
If writers can set arbitrary `security_groups`, they can:
- make docs readable by groups they don’t control,
- exfiltrate by tagging docs as `doc:public` or a broad group.

### Recommended “allowed-to-assign” policy
Separate “can write” from “can tag with group X”.

Use LDAP groups like:
- `milvus:{collection}:tag:{doc_group}` (fine-grained)
  - e.g., `milvus:hr_policies:tag:doc:finance:payroll`
- Optional coarse:
  - `milvus:{collection}:tag:any` (highly privileged; avoid if possible)

Rules:
- On write, the server computes `assignable_doc_groups` for the writer from LDAP.
- Validate that `requested_security_groups` (or IDs) is a subset of `assignable_doc_groups`.
- Deny by default if doc has no security groups.

Also consider:
- Disallow tagging with collection RBAC groups entirely (different namespace).
- For updates, prevent expanding ACL unless user has explicit tag permission for the newly added groups.

### Validation pseudocode (Python)
```python
class ValidationError(Exception):
    pass


def derive_assignable_doc_group_ids(effective_groups: list[str], collection: str) -> set[int]:
    # Example: groups like "milvus:{collection}:tag:{doc_group_name}"
    prefix = f"milvus:{collection}:tag:"
    assignable_names = set()

    for g in effective_groups:
        if g.startswith(prefix):
            assignable_names.add(g[len(prefix):])

    # Map doc group names -> stable IDs
    return {group_name_to_id(name) for name in assignable_names if is_doc_group(name)}


def validate_security_groups_on_write(
    ctx: AuthContext,
    collection: str,
    requested_group_ids: list[int],
) -> None:
    if ctx.collection_perm not in ("rw", "admin"):
        raise AuthorizationError("forbidden")

    if not requested_group_ids:
        # deny-by-default: require explicit ACL
        raise ValidationError("missing_security_groups")

    assignable = derive_assignable_doc_group_ids(ctx.effective_groups, collection)

    req = set(requested_group_ids)
    if not req.issubset(assignable):
        raise AuthorizationError("forbidden")


def validate_acl_update(
    ctx: AuthContext,
    collection: str,
    old_group_ids: list[int],
    new_group_ids: list[int],
) -> None:
    validate_security_groups_on_write(ctx, collection, new_group_ids)

    old = set(old_group_ids)
    new = set(new_group_ids)
    added = new - old

    if added:
        # Must be explicitly allowed to add those groups (already ensured by subset)
        pass

    # Optional: also restrict removals if you need retention controls
```

### Delete/update scope (avoid destructive escalation)
For delete/update operations, require:
- `rw/admin` on collection AND
- server-side constraint that the target docs are within the user’s *assignable* scope or at least within their *readable* scope (depending on business needs).
A safe default is:
- only allow delete/update for docs whose `security_group_ids` is a subset of writer’s `assignable` groups.

### ACL change implications
- If you store ACL purely as metadata fields in Milvus, updating ACL is a metadata update (no re-embedding needed).
- Ensure the update is atomic and that subsequent reads immediately respect new ACL (Milvus query consistency settings may matter; test).

---

## 8) Data/schema recommendations

### Milvus collection schema (minimum)
Per chunk/document row:
- `pk`: int64 / uuid (primary key)
- `doc_id`: string (source document identifier)
- `chunk_id`: int32 (chunk index)
- `embedding`: float vector
- `source`: string (URI/path)
- `created_at`: int64 (epoch ms)
- `updated_at`: int64
- `security_group_ids`: array<int64>  **(recommended)**
  - or `security_groups`: array<string> if IDs aren’t feasible
- Optional but often useful:
  - `tenant_id`: string/int (if multi-tenant)
  - `doc_version`: string/int
  - `content_hash`: string (integrity/debug)

Do **not** return `security_group_ids` to clients.

### Local DB/cache (recommended)
Even if you use on-demand LDAP, you still want a local store for:
- group name ↔ stable ID mapping
- optional collection policy overrides
- caching

#### Tables/keys
1) `user_group_cache` (Redis or DB)
- `username` (pk)
- `groups` (compressed list of LDAP group strings or IDs)
- `doc_group_ids` (list<int64>)
- `fetched_at`
- `expires_at`
- `etag`/`version` optional (if LDAP provides)

Example record:
```text
username="matt"
groups=["milvus:hr_policies:r", "doc:finance:payroll", "doc:eng:platform"]
doc_group_ids=[1012, 2044]
fetched_at=1734112800000
expires_at=1734113100000
```

2) `collection_policy` (optional if not purely naming convention)
- `collection` (pk)
- `r_group` (string) e.g. `milvus:{collection}:r`
- `rw_group`
- `admin_group`
- `tag_prefix` (string) e.g. `milvus:{collection}:tag:`

Example:
```text
collection="hr_policies"
r_group="milvus:hr_policies:r"
rw_group="milvus:hr_policies:rw"
admin_group="milvus:hr_policies:admin"
tag_prefix="milvus:hr_policies:tag:"
```

3) `group_alias` (stable IDs)
- `group_name` (pk) e.g. `doc:finance:payroll`
- `group_id` (int64) e.g. 1012
- `status` (active/deprecated)
- `created_at`, `updated_at`

Example:
```text
group_name="doc:finance:payroll"
group_id=1012
status="active"
```

---

## 9) Revocation, auditing, and operations

### Revocation (“user removed from group”)
- Bound by cache TTL (recommended 5 minutes).
- If LDAP supports change notifications:
  - subscribe and invalidate `user_group_cache` entries on membership change.
- If not:
  - rely on TTL + background refresh for hot users.

### Token/session strategy
Even if clients present an LDAP token:
- Do not treat token as authorization proof beyond identity.
- Still resolve username and **re-check groups** at least every TTL window.
- If you maintain app sessions, store:
  - `username`
  - `groups_hash`
  - `groups_fetched_at`
But still enforce TTL and refresh.

### Logs to emit (audit)
For every Milvus operation:
- `request_id`
- `username`
- `collection`
- `operation` (search/query/insert/delete)
- `collection_perm`
- `doc_groups_hash` (hash of sorted doc group IDs; not raw list)
- `acl_expr_hash` (hash of expr string)
- `result_count_returned`
- `top_k_requested`, `top_k_effective`
- `latency_ms` (total + milvus call)
- `decision` (allow/deny + reason code)

Avoid logging raw group lists unless in restricted debug logs.

### Handling LDAP downtime safely
Recommended policy:
- If cached entry is valid (not expired): allow using cached groups.
- If expired and LDAP is unreachable: **deny** (fail closed).
- Expose operational metric: `authz_ldap_errors_total`, `authz_denies_due_to_ldap_unavailable_total`.

### Runbook: onboarding a new collection + groups
1) Create Milvus collection via infra pipeline (not via app API).
2) Define LDAP collection RBAC groups:
   - `milvus:{collection}:r`, `:rw`, `:admin`
3) Define tagging groups for writers:
   - `milvus:{collection}:tag:{doc_group}`
4) Define/allocate doc groups (in LDAP + `group_alias` mapping):
   - `doc:{domain}:{name}` → stable `group_id`
5) Update any `collection_policy` entry if you use that table.
6) Deploy and test:
   - user with `r` sees only docs with matching doc groups
   - user without `r` denied even if they have doc groups
   - writer cannot tag unauthorized groups

---

## 10) Concrete examples

### Example LDAP groups (two collections, differing roles)

Collections:
- `hr_policies`
- `eng_runbooks`

Users:
- Alice (HR analyst)
  - `milvus:hr_policies:r`
  - `doc:hr:general`
  - `doc:finance:payroll`
- Bob (Engineer)
  - `milvus:eng_runbooks:r`
  - `doc:eng:platform`
- Carol (HR editor)
  - `milvus:hr_policies:rw`
  - `milvus:hr_policies:tag:doc:hr:general`
  - (no permission to tag payroll)

`group_alias`:
- `doc:hr:general` → 3001
- `doc:finance:payroll` → 1012
- `doc:eng:platform` → 2044

### Example doc `security_groups`
In `hr_policies`:
- Doc A (general HR policy): `security_group_ids=[3001]`
- Doc B (payroll policy): `security_group_ids=[1012]`

In `eng_runbooks`:
- Doc C (platform runbook): `security_group_ids=[2044]`

### Example request + derived filter
Alice searches `hr_policies`:

- Collection check:
  - Alice has `milvus:hr_policies:r` → allowed (`r`)
- Effective doc group IDs:
  - `[3001, 1012]`
- Mandatory expr:
  - `array_contains_any(security_group_ids, [3001,1012])`

Milvus search is executed with that expr; results may include Doc A and Doc B.

### Example denied scenarios

1) **Collection denied**
Bob tries to search `hr_policies`:
- Bob lacks `milvus:hr_policies:r` → `none`
- Deny before Milvus call (generic 403). Even though Bob has `doc:eng:platform`, it does not matter.

2) **Doc filtered out**
Alice searches `eng_runbooks`:
- Alice lacks `milvus:eng_runbooks:r` → denied at collection level (even if some docs were tagged with `doc:finance:payroll`).

If Alice *did* have `milvus:eng_runbooks:r` but lacked `doc:eng:platform`, then:
- Allowed to query collection, but expr is `array_contains_any(security_group_ids, [3001,1012])`
- Doc C has `[2044]` → no intersection → not returned.

3) **Write privilege escalation blocked**
Carol (HR editor) attempts to insert a payroll doc into `hr_policies` with `security_group_ids=[1012]`:
- Collection perm is `rw` (passes)
- Assignable groups derived from `milvus:hr_policies:tag:*`:
  - only `doc:hr:general` → 3001
- Requested includes 1012 → not subset → deny (generic 403/validation error)

---

### Implementation checklist (tie it together)
1) Define LDAP group naming conventions (`milvus:{collection}:{r|rw|admin}`, `milvus:{collection}:tag:{doc_group}`, `doc:*`).
2) Implement `get_username()` (token→username) and canonicalization.
3) Implement `get_groups(username)` with Redis cache (TTL=5m) + LDAP fetch.
4) Implement `collection_permission(groups, collection)` decision function (deny default).
5) Implement `doc_group_ids = map_doc_groups_to_ids(groups)` using `group_alias`.
6) Build a **single secured Milvus read API** that always injects ACL expr and is the only way handlers can read.
7) Build secured write APIs that validate `security_group_ids` against `assignable` tagging groups.
8) Add audit logs + metrics + alerting for LDAP failures and unauthorized attempts.
9) Add tests:
   - no expr => request fails
   - missing/empty `security_groups` => unreadable
   - collection denied even if doc groups match
   - writer cannot tag broader groups
10) Operationalize onboarding/runbook and periodic review of group mappings.

If you tell me your Milvus version + SDK (pymilvus version) and whether array membership functions like `array_contains_any` are supported in your deployment, I can pin the exact expr syntax and provide drop-in code for the read/write wrappers.