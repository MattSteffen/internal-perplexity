package ollama

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"slices"
)

/*
curl -X GET http://localhost:11434/api/tags | jq
{
  "models": [
    {
      "name": "gpt-oss:20b",
      "model": "gpt-oss:20b",
      "modified_at": "2025-08-07T14:16:32.768491498-07:00",
      "size": 13780173839,
      "digest": "f2b8351c629c005bd3f0a0e3046f905afcbffede19b648e4bd7c884cdfd63af6",
      "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "gptoss",
        "families": [
          "gptoss"
        ],
        "parameter_size": "20.9B",
        "quantization_level": "MXFP4"
      }
    }
  ]
}

*/

type ModelFamily string

const (
	ModelFamilyBert   ModelFamily = "bert"
	ModelFamilyQwen3  ModelFamily = "qwen3"
	ModelFamilyLlama  ModelFamily = "llama"
	ModelFamilyGemma3 ModelFamily = "gemma3"
	ModelFamilyGPTOss ModelFamily = "gptoss"
)

var SupportedModelFamilies = []ModelFamily{
	ModelFamilyQwen3,
	ModelFamilyLlama,
	ModelFamilyGemma3,
	ModelFamilyGPTOss,
}

var UnsupportedModelFamilies = []ModelFamily{
	ModelFamilyBert,
}

type Model struct {
	Name       string
	Model      string
	ModifiedAt string
	Size       int
	Digest     string
	Details    ModelDetails
}

type ModelDetails struct {
	ParentModel       string
	Format            string
	Family            ModelFamily
	Families          []ModelFamily
	ParameterSize     string
	QuantizationLevel string
}

func (m *Model) IsSupported(model string) bool {
	return slices.Contains(SupportedModelFamilies, m.Details.Family)
}

func GetModels(baseURL string) ([]Model, error) {
	url := fmt.Sprintf("%s/api/tags", baseURL)
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to get models: %s", resp.Status)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %v", err)
	}

	var models struct {
		Models []Model `json:"models"`
	}
	if err := json.Unmarshal(body, &models); err != nil {
		return nil, fmt.Errorf("failed to parse response: %v, body: %s", err, string(body))
	}

	return models.Models, nil
}
