package storage

import (
	"context"
	"fmt"
	"strconv"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"go-crawler/internal/config"
	"go-crawler/pkg/interfaces"
)

// MilvusClient implements the DatabaseClient interface for Milvus
type MilvusClient struct {
	config         *config.DatabaseConfig
	embeddingDim   int
	metadataSchema map[string]interface{}
	client         client.Client
	collectionName string
}

// NewMilvusClient creates a new Milvus database client
func NewMilvusClient(cfg *config.DatabaseConfig, embeddingDim int, metadataSchema map[string]interface{}) (*MilvusClient, error) {
	if cfg.Collection == "" {
		return nil, fmt.Errorf("collection name is required")
	}

	ctx := context.Background()
	c, err := client.NewClient(ctx, client.Config{
		Address: cfg.GetURI(),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Milvus: %v", err)
	}

	mc := &MilvusClient{
		config:         cfg,
		embeddingDim:   embeddingDim,
		metadataSchema: metadataSchema,
		client:         c,
		collectionName: cfg.Collection,
	}

	return mc, nil
}

// CreateCollection creates the Milvus collection with the appropriate schema
func (m *MilvusClient) CreateCollection(ctx context.Context, recreate bool) error {
	// Check if collection exists
	exists, err := m.client.HasCollection(ctx, m.collectionName)
	if err != nil {
		return fmt.Errorf("failed to check collection existence: %v", err)
	}

	if exists && !recreate {
		return nil // Collection already exists
	}

	if exists && recreate {
		// Drop existing collection
		if err := m.client.DropCollection(ctx, m.collectionName); err != nil {
			return fmt.Errorf("failed to drop existing collection: %v", err)
		}
	}

	// Create new collection
	schema, err := m.buildSchema()
	if err != nil {
		return fmt.Errorf("failed to build schema: %v", err)
	}

	if err := m.client.CreateCollection(ctx, schema, entity.DefaultShardNumber); err != nil {
		return fmt.Errorf("failed to create collection: %v", err)
	}

	// Create index on text_embedding field
	index, err := entity.NewIndexFlat(entity.COSINE)
	if err != nil {
		return fmt.Errorf("failed to create index: %v", err)
	}

	if err := m.client.CreateIndex(ctx, m.collectionName, "text_embedding", index, false); err != nil {
		return fmt.Errorf("failed to create index: %v", err)
	}

	// Load collection into memory
	if err := m.client.LoadCollection(ctx, m.collectionName, false); err != nil {
		return fmt.Errorf("failed to load collection: %v", err)
	}

	return nil
}

// InsertDocuments inserts documents into the Milvus collection
func (m *MilvusClient) InsertDocuments(ctx context.Context, docs []interfaces.Document) error {
	if len(docs) == 0 {
		return nil
	}

	// Prepare data columns
	textColumn := make([]string, len(docs))
	embeddingColumn := make([][]float32, len(docs))
	chunkIndexColumn := make([]int64, len(docs))
	sourceColumn := make([]string, len(docs))

	// Dynamic columns for metadata
	metadataColumns := make(map[string][]interface{})

	for i, doc := range docs {
		textColumn[i] = doc.Text
		// Convert float64 to float32
		embeddingColumn[i] = make([]float32, len(doc.TextEmbedding))
		for j, v := range doc.TextEmbedding {
			embeddingColumn[i][j] = float32(v)
		}
		chunkIndexColumn[i] = int64(doc.ChunkIndex)
		sourceColumn[i] = doc.Source

		// Extract metadata fields
		for key, value := range doc.Metadata {
			if metadataColumns[key] == nil {
				metadataColumns[key] = make([]interface{}, len(docs))
			}
			metadataColumns[key][i] = value
		}
	}

	// Build column data
	columnData := []entity.Column{
		entity.NewColumnString("text", textColumn),
		entity.NewColumnFloatVector("text_embedding", m.embeddingDim, embeddingColumn),
		entity.NewColumnInt64("chunk_index", chunkIndexColumn),
		entity.NewColumnString("source", sourceColumn),
	}

	// Add metadata columns
	for fieldName, values := range metadataColumns {
		col := m.createColumnForField(fieldName, values)
		if col != nil {
			columnData = append(columnData, col)
		}
	}

	// Insert data
	_, err := m.client.Insert(ctx, m.collectionName, "", columnData...)
	if err != nil {
		return fmt.Errorf("failed to insert documents: %v", err)
	}

	// Flush to ensure data is persisted
	if err := m.client.Flush(ctx, m.collectionName, false); err != nil {
		return fmt.Errorf("failed to flush collection: %v", err)
	}

	return nil
}

// CheckDuplicate checks if a document chunk already exists
func (m *MilvusClient) CheckDuplicate(ctx context.Context, source string, chunkIndex int) (bool, error) {
	// Build filter expression
	filter := fmt.Sprintf("source == '%s' && chunk_index == %d", source, chunkIndex)

	// Query to check existence
	_, err := m.client.Query(ctx, m.collectionName, []string{}, filter, []string{"id"})
	if err != nil {
		// If query fails, assume it's not a duplicate
		return false, nil
	}

	// If we get results, it's a duplicate
	results, err := m.client.Query(ctx, m.collectionName, []string{}, filter, []string{"id"})
	if err != nil {
		return false, fmt.Errorf("failed to query for duplicates: %v", err)
	}

	return len(results) > 0, nil
}

// Search performs vector search on the collection
func (m *MilvusClient) Search(ctx context.Context, query string, limit int, filters map[string]interface{}) ([]interfaces.SearchResult, error) {
	// For now, this is a placeholder. In a full implementation, we'd need to:
	// 1. Embed the query using the embedder
	// 2. Perform vector search
	// 3. Filter results
	// 4. Return formatted results

	return nil, fmt.Errorf("search not implemented yet - requires embedder integration")
}

// Close closes the Milvus client connection
func (m *MilvusClient) Close() error {
	if m.client != nil {
		return m.client.Close()
	}
	return nil
}

// buildSchema creates the Milvus collection schema
func (m *MilvusClient) buildSchema() (*entity.Schema, error) {
	schema := &entity.Schema{
		CollectionName: m.collectionName,
		Description:    "Document collection for vector search",
		Fields:         []*entity.Field{},
	}

	// Add ID field (auto-generated)
	idField := &entity.Field{
		Name:        "id",
		DataType:    entity.FieldTypeInt64,
		PrimaryKey:  true,
		AutoID:      true,
		Description: "Primary key",
	}
	schema.Fields = append(schema.Fields, idField)

	// Add core fields
	coreFields := []*entity.Field{
		{
			Name:        "text",
			DataType:    entity.FieldTypeVarChar,
			TypeParams:  map[string]string{"max_length": "65535"},
			Description: "Document text chunk",
		},
		{
			Name:        "text_embedding",
			DataType:    entity.FieldTypeFloatVector,
			TypeParams:  map[string]string{"dim": strconv.Itoa(m.embeddingDim)},
			Description: "Text embedding vector",
		},
		{
			Name:        "chunk_index",
			DataType:    entity.FieldTypeInt64,
			Description: "Index of chunk within document",
		},
		{
			Name:        "source",
			DataType:    entity.FieldTypeVarChar,
			TypeParams:  map[string]string{"max_length": "1000"},
			Description: "Source file path",
		},
	}

	for _, field := range coreFields {
		schema.Fields = append(schema.Fields, field)
	}

	// Add metadata fields based on schema
	metadataFields, err := m.buildMetadataFields()
	if err != nil {
		return nil, err
	}

	schema.Fields = append(schema.Fields, metadataFields...)

	return schema, nil
}

// buildMetadataFields creates fields for metadata based on the JSON schema
func (m *MilvusClient) buildMetadataFields() ([]*entity.Field, error) {
	var fields []*entity.Field

	properties, ok := m.metadataSchema["properties"].(map[string]interface{})
	if !ok {
		return fields, nil
	}

	for fieldName, fieldDef := range properties {
		fieldDefMap, ok := fieldDef.(map[string]interface{})
		if !ok {
			continue
		}

		fieldType, ok := fieldDefMap["type"].(string)
		if !ok {
			continue
		}

		field, err := m.createFieldForType(fieldName, fieldType, fieldDefMap)
		if err != nil {
			return nil, fmt.Errorf("failed to create field %s: %v", fieldName, err)
		}

		if field != nil {
			fields = append(fields, field)
		}
	}

	return fields, nil
}

// createFieldForType creates a Milvus field based on JSON schema type
func (m *MilvusClient) createFieldForType(fieldName, fieldType string, fieldDef map[string]interface{}) (*entity.Field, error) {
	switch fieldType {
	case "string":
		maxLength := "1000" // default
		if maxLen, ok := fieldDef["maxLength"].(float64); ok {
			maxLength = strconv.Itoa(int(maxLen))
		}
		return &entity.Field{
			Name:        fieldName,
			DataType:    entity.FieldTypeVarChar,
			TypeParams:  map[string]string{"max_length": maxLength},
			Description: fmt.Sprintf("String field: %s", fieldName),
		}, nil

	case "integer":
		return &entity.Field{
			Name:        fieldName,
			DataType:    entity.FieldTypeInt64,
			Description: fmt.Sprintf("Integer field: %s", fieldName),
		}, nil

	case "number":
		return &entity.Field{
			Name:        fieldName,
			DataType:    entity.FieldTypeFloat,
			Description: fmt.Sprintf("Number field: %s", fieldName),
		}, nil

	case "boolean":
		return &entity.Field{
			Name:        fieldName,
			DataType:    entity.FieldTypeBool,
			Description: fmt.Sprintf("Boolean field: %s", fieldName),
		}, nil

	case "array":
		// Handle array types
		items, ok := fieldDef["items"].(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("array field %s missing items definition", fieldName)
		}

		itemType, ok := items["type"].(string)
		if !ok {
			return nil, fmt.Errorf("array field %s items missing type", fieldName)
		}

		switch itemType {
		case "string":
			return &entity.Field{
				Name:        fieldName,
				DataType:    entity.FieldTypeVarChar,
				TypeParams:  map[string]string{"max_length": "10000"}, // Store as JSON string
				Description: fmt.Sprintf("String array field: %s", fieldName),
			}, nil
		default:
			return nil, fmt.Errorf("unsupported array item type: %s", itemType)
		}

	default:
		return nil, fmt.Errorf("unsupported field type: %s", fieldType)
	}
}

// createColumnForField creates a column for a metadata field
func (m *MilvusClient) createColumnForField(fieldName string, values []interface{}) entity.Column {
	if len(values) == 0 {
		return nil
	}

	// Determine the type from the first non-nil value
	var sampleValue interface{}
	for _, v := range values {
		if v != nil {
			sampleValue = v
			break
		}
	}

	if sampleValue == nil {
		return nil
	}

	_ = sampleValue // Mark as used for type assertion below

	switch v := sampleValue.(type) {
	case string:
		stringValues := make([]string, len(values))
		for i, val := range values {
			if val != nil {
				stringValues[i] = val.(string)
			}
		}
		return entity.NewColumnString(fieldName, stringValues)

	case int, int64:
		intValues := make([]int64, len(values))
		for i, val := range values {
			if val != nil {
				switch v := val.(type) {
				case int:
					intValues[i] = int64(v)
				case int64:
					intValues[i] = v
				}
			}
		}
		return entity.NewColumnInt64(fieldName, intValues)

	case float64:
		floatValues := make([]float32, len(values))
		for i, val := range values {
			if val != nil {
				floatValues[i] = float32(val.(float64))
			}
		}
		return entity.NewColumnFloat(fieldName, floatValues)

	case bool:
		boolValues := make([]bool, len(values))
		for i, val := range values {
			if val != nil {
				boolValues[i] = val.(bool)
			}
		}
		return entity.NewColumnBool(fieldName, boolValues)

	case []interface{}:
		// Convert array to JSON string
		jsonValues := make([]string, len(values))
		for i, val := range values {
			if val != nil {
				// This is a simplified approach - in practice, you'd want proper JSON marshaling
				jsonValues[i] = fmt.Sprintf("%v", val)
			}
		}
		return entity.NewColumnString(fieldName, jsonValues)

	default:
		// Convert to string as fallback
		stringValues := make([]string, len(values))
		for i, val := range values {
			if val != nil {
				stringValues[i] = fmt.Sprintf("%v", val)
			}
		}
		return entity.NewColumnString(fieldName, stringValues)
	}
}
