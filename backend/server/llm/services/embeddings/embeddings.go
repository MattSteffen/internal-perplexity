package embeddings

import "context"

type EmbeddingsRegistry interface {
	GetEmbeddings(ctx context.Context, text string) ([]float64, error)
}
