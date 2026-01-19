"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface CreateCollectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCreate: (data: CreateCollectionData) => Promise<void>;
}

export interface CreateCollectionData {
  collection_name: string;
  template_name?: string | null;
  custom_config: Record<string, unknown> | null;
  config_overrides: Record<string, unknown> | null;
  description: string | null;
  roles?: string[] | null;
  access_level: "public" | "private" | "admin";
  metadata_schema: Record<string, unknown> | null;
}

const getDefaultConfig = () => {
  const ollamaBaseUrl = process.env.NEXT_PUBLIC_OLLAMA_BASE_URL || "http://localhost:11434";
  return {
    embeddings: {
      type: "ollama",
      model: "all-minilm:v2",
      base_url: ollamaBaseUrl,
    },
    llm: {
      type: "ollama",
      model_name: "llama3.2:3b",
      base_url: ollamaBaseUrl,
      structured_output: null,
    },
    vision_llm: {
      type: "ollama",
      model_name: "granite3.2-vision:latest",
      base_url: ollamaBaseUrl,
    },
    database: {
      provider: "milvus",
      collection: "",
      host: "localhost",
      port: 19530,
      username: "root",
      password: "Milvus",
      recreate: false,
    },
    converter: {
      type: "pymupdf4llm",
    },
    extractor: {
      type: "basic",
      llm: {
        type: "ollama",
        model_name: "llama3.2:3b",
        base_url: ollamaBaseUrl,
      },
    },
    chunking: {
      type: "fixed_size",
      chunk_size: 10000,
    },
    metadata_schema: {},
    temp_dir: "tmp/",
    benchmark: false,
    generate_benchmark_questions: false,
    num_benchmark_questions: 3,
  };
};

const DEFAULT_CONFIG = getDefaultConfig();

export function CreateCollectionModal({
  isOpen,
  onClose,
  onCreate,
}: CreateCollectionModalProps) {
  const [collectionName, setCollectionName] = useState("");
  const [description, setDescription] = useState("");
  const [pipelineType, setPipelineType] = useState<"predefined" | "custom">("predefined");
  const [configOverrides, setConfigOverrides] = useState("{}");
  const [customConfig, setCustomConfig] = useState(JSON.stringify(DEFAULT_CONFIG, null, 2));
  const [accessLevel, setAccessLevel] = useState<"public" | "private" | "admin">("public");
  const [metadataSchema, setMetadataSchema] = useState("{}");
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!collectionName.trim()) {
      setError("Collection name is required");
      return;
    }

    try {
      setIsCreating(true);

      let parsedConfigOverrides: Record<string, unknown> | null = null;
      let parsedCustomConfig: Record<string, unknown> | null = null;
      let parsedMetadataSchema: Record<string, unknown> | null = null;

      if (pipelineType === "predefined") {
        if (configOverrides.trim()) {
          parsedConfigOverrides = JSON.parse(configOverrides);
        }
      } else {
        if (customConfig.trim()) {
          parsedCustomConfig = JSON.parse(customConfig);
        }
      }

      if (metadataSchema.trim()) {
        parsedMetadataSchema = JSON.parse(metadataSchema);
      }

      const data: CreateCollectionData = {
        collection_name: collectionName.trim(),
        custom_config: parsedCustomConfig,
        config_overrides: parsedConfigOverrides,
        description: description.trim() || null,
        access_level: accessLevel,
        metadata_schema: parsedMetadataSchema,
      };

      await onCreate(data);

      // Reset form
      setCollectionName("");
      setDescription("");
      setPipelineType("predefined");
      setConfigOverrides("{}");
      setCustomConfig(JSON.stringify(DEFAULT_CONFIG, null, 2));
      setAccessLevel("public");
      setMetadataSchema("{}");
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create collection");
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="w-full max-w-3xl max-h-[90vh] overflow-y-auto rounded-lg border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900 p-6 shadow-lg">
        <div className="mb-6 flex items-center justify-between">
          <h2 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
            Create New Collection
          </h2>
          <button
            onClick={onClose}
            className="rounded-lg p-1 text-zinc-500 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-300"
          >
            <svg
              className="h-5 w-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {error && (
          <div className="mb-4 rounded-lg border border-red-300 bg-red-50 p-3 dark:border-red-800 dark:bg-red-950/30">
            <p className="text-sm text-red-800 dark:text-red-300">{error}</p>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="mb-1 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
              Collection Name <span className="text-red-500">*</span>
            </label>
            <Input
              type="text"
              value={collectionName}
              onChange={(e) => setCollectionName(e.target.value)}
              placeholder="my_collection"
              required
              className="w-full"
            />
          </div>

          <div>
            <label className="mb-1 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
              Description
            </label>
            <Input
              type="text"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Optional description of the collection"
              className="w-full"
            />
          </div>

          <div>
            <label className="mb-1 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
              Pipeline Type
            </label>
            <div className="flex gap-4">
              <label className="flex cursor-pointer items-center gap-2">
                <input
                  type="radio"
                  name="pipelineType"
                  value="predefined"
                  checked={pipelineType === "predefined"}
                  onChange={(e) => setPipelineType(e.target.value as "predefined" | "custom")}
                  className="h-4 w-4 text-blue-600 focus:ring-2 focus:ring-blue-500/20"
                />
                <span className="text-sm text-zinc-700 dark:text-zinc-300">Predefined Pipeline</span>
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <input
                  type="radio"
                  name="pipelineType"
                  value="custom"
                  checked={pipelineType === "custom"}
                  onChange={(e) => setPipelineType(e.target.value as "predefined" | "custom")}
                  className="h-4 w-4 text-blue-600 focus:ring-2 focus:ring-blue-500/20"
                />
                <span className="text-sm text-zinc-700 dark:text-zinc-300">Custom Config</span>
              </label>
            </div>
          </div>

          {pipelineType === "predefined" ? (
            <>
              <div>
                <label className="mb-1 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                  Config Overrides (JSON, optional)
                </label>
                <textarea
                  value={configOverrides}
                  onChange={(e) => setConfigOverrides(e.target.value)}
                  placeholder='{"embedding_model": "nomic-embed-text"}'
                  rows={4}
                  className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 font-mono text-sm text-zinc-700 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300"
                />
              </div>
            </>
          ) : (
            <div>
              <label className="mb-1 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                Custom Config (JSON)
              </label>
              <textarea
                value={customConfig}
                onChange={(e) => setCustomConfig(e.target.value)}
                rows={12}
                className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 font-mono text-sm text-zinc-700 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300"
              />
            </div>
          )}

          <div>
            <label className="mb-1 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
              Access Level
            </label>
            <select
              value={accessLevel}
              onChange={(e) => setAccessLevel(e.target.value as "public" | "private" | "admin")}
              className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-700 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300"
            >
              <option value="public">Public</option>
              <option value="private">Private</option>
              <option value="admin">Admin</option>
            </select>
          </div>

          <div>
            <label className="mb-1 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
              Metadata Schema (JSON, optional)
            </label>
            <textarea
              value={metadataSchema}
              onChange={(e) => setMetadataSchema(e.target.value)}
              placeholder='{"type": "object", "properties": {...}}'
              rows={6}
              className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 font-mono text-sm text-zinc-700 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300"
            />
          </div>

          <div className="flex justify-end gap-3 pt-4">
            <Button
              type="button"
              variant="outline"
              onClick={onClose}
              disabled={isCreating}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={isCreating}>
              {isCreating ? "Creating..." : "Create Collection"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}

