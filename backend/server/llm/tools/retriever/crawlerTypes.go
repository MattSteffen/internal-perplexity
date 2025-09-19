package retriever

import "time"

/*
What are the attributes that I will have as default?
How should I handle the rest?
*/
type (
	// MilvusDocument represents a single document chunk as stored/retrieved from Milvus
	MilvusDocument struct {
		ID          int64          `json:"id"`
		DocumentID  string         `json:"document_id"`
		Source      string         `json:"source"`
		ChunkIndex  int            `json:"chunk_index"`
		Metadata    map[string]any `json:"metadata,omitempty"`
		Title       string         `json:"title,omitempty"`
		Author      []string       `json:"author,omitempty"`
		Date        time.Time      `json:"date,omitempty"` // YYYY format
		Keywords    []string       `json:"keywords,omitempty"`
		UniqueWords []string       `json:"unique_words,omitempty"`
		Text        string         `json:"text,omitempty"`
		Distance    *float64       `json:"distance,omitempty"` // nullable value
	}

	// ConsolidatedDocument represents merged chunks from the same source
	ConsolidatedDocument struct {
		ID          int64     `json:"id"`
		DocumentID  string    `json:"document_id"`
		Source      string    `json:"source"`
		Title       string    `json:"title,omitempty"`
		Author      []string  `json:"author,omitempty"`
		Date        time.Time `json:"date,omitempty"`
		Keywords    []string  `json:"keywords,omitempty"`
		UniqueWords []string  `json:"unique_words,omitempty"`
		Text        string    `json:"text,omitempty"`
		AvgDistance *float64  `json:"avg_distance,omitempty"`
	}

	// Citation object for user-facing outputs
	Citation struct {
		ID         int64          `json:"id"`
		DocumentID string         `json:"document_id"`
		Source     CitationSource `json:"source"`
		Document   []string       `json:"document"` // usually a markdown-prettified version
		Metadata   map[string]any `json:"metadata"`
		Distance   *float64       `json:"avg_distance,omitempty"`
	}

	CitationSource struct {
		Name string `json:"name"`
		URL  string `json:"url,omitempty"`
	}

	// SearchInput is used when constructing tool calls
	SearchInput struct {
		Queries []string `json:"queries,omitempty"`
		Filters []string `json:"filters,omitempty"`
	}

	// SearchResult wraps Milvus search responses
	SearchResult struct {
		Documents []MilvusDocument `json:"documents"`
	}

	// MetadataResponse gives a high-level summary (title/author/date) for quick queries
	MetadataResponse struct {
		Title  string    `json:"title"`
		Author []string  `json:"author"`
		Date   time.Time `json:"date"`
	}
)
