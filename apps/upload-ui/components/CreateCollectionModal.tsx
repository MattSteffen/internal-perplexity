"use client";

import { useEffect, useState } from "react";
import type { AccessLevel, CreateCollectionRequest, PipelineInfo } from "@/lib/types";
import { fetchPipelines, fetchPipelineConfig } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface CreateCollectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCreate: (data: CreateCollectionRequest) => Promise<void>;
}

const getDefaultConfig = (collectionName: string) => {
  const ollamaBaseUrl = process.env.NEXT_PUBLIC_OLLAMA_BASE_URL || "http://localhost:11434";
  return {
    name: collectionName,
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
      collection: collectionName,
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

const ACCESS_LEVELS: { value: AccessLevel; label: string }[] = [
  { value: "public", label: "Public" },
  { value: "private", label: "Private" },
  { value: "group_only", label: "Group only" },
  { value: "admin", label: "Admin" },
];

export function CreateCollectionModal({
  isOpen,
  onClose,
  onCreate,
}: CreateCollectionModalProps) {
  const [collectionName, setCollectionName] = useState("");
  const [accessLevel, setAccessLevel] = useState<AccessLevel>("public");
  const [accessGroupsStr, setAccessGroupsStr] = useState("");
  const [crawlerConfigStr, setCrawlerConfigStr] = useState("");
  const [pipelines, setPipelines] = useState<PipelineInfo[]>([]);
  const [selectedPipelineName, setSelectedPipelineName] = useState("");
  const [loadingPipelines, setLoadingPipelines] = useState(false);
  const [loadingConfig, setLoadingConfig] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // When modal opens, fetch pipeline list and reset pipeline/config state
  useEffect(() => {
    if (!isOpen) return;
    setSelectedPipelineName("");
    setCrawlerConfigStr("");
    setError(null);
    setLoadingPipelines(true);
    fetchPipelines()
      .then((list) => setPipelines(list))
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load pipelines"))
      .finally(() => setLoadingPipelines(false));
  }, [isOpen]);

  const handlePipelineChange = (name: string) => {
    setSelectedPipelineName(name);
    if (!name) {
      setCrawlerConfigStr("");
      return;
    }
    setLoadingConfig(true);
    setError(null);
    fetchPipelineConfig(name)
      .then((config) => setCrawlerConfigStr(JSON.stringify(config, null, 2)))
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load pipeline config"))
      .finally(() => setLoadingConfig(false));
  };

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!collectionName.trim()) {
      setError("Collection name is required");
      return;
    }

    if (accessLevel === "group_only") {
      const groups = accessGroupsStr
        .split(/[\s,]+/)
        .map((s) => s.trim())
        .filter(Boolean);
      if (groups.length === 0) {
        setError("At least one access group is required when access level is Group only");
        return;
      }
    }

    try {
      setIsCreating(true);

      let crawlerConfig: Record<string, unknown>;
      if (crawlerConfigStr.trim()) {
        try {
          crawlerConfig = JSON.parse(crawlerConfigStr) as Record<string, unknown>;
        } catch {
          setError("Crawler config must be valid JSON");
          return;
        }
      } else {
        crawlerConfig = getDefaultConfig(collectionName.trim()) as Record<string, unknown>;
      }

      // Ensure database.collection matches collection name
      const db = (crawlerConfig.database as Record<string, unknown>) ?? {};
      crawlerConfig.database = { ...db, collection: collectionName.trim() };
      if (!crawlerConfig.name) crawlerConfig.name = collectionName.trim();

      const access_groups =
        accessLevel === "group_only"
          ? accessGroupsStr
              .split(/[\s,]+/)
              .map((s) => s.trim())
              .filter(Boolean)
          : [];

      const data: CreateCollectionRequest = {
        access_level: accessLevel,
        access_groups,
        crawler_config: crawlerConfig,
      };

      await onCreate(data);

      setCollectionName("");
      setAccessLevel("public");
      setAccessGroupsStr("");
      setCrawlerConfigStr("");
      setSelectedPipelineName("");
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create collection");
    } finally {
      setIsCreating(false);
    }
  };

  const accessGroupsPlaceholder =
    accessLevel === "group_only"
      ? "Comma- or space-separated role names (e.g. engineers, research)"
      : "Not used unless access level is Group only";

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
              Access Level
            </label>
            <select
              value={accessLevel}
              onChange={(e) => setAccessLevel(e.target.value as AccessLevel)}
              className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-700 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300"
            >
              {ACCESS_LEVELS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          {accessLevel === "group_only" && (
            <div>
              <label className="mb-1 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                Access Groups <span className="text-red-500">*</span>
              </label>
              <Input
                type="text"
                value={accessGroupsStr}
                onChange={(e) => setAccessGroupsStr(e.target.value)}
                placeholder={accessGroupsPlaceholder}
                className="w-full"
              />
            </div>
          )}

          <div>
            <label className="mb-1 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
              Pipeline template
            </label>
            <select
              value={selectedPipelineName}
              onChange={(e) => handlePipelineChange(e.target.value)}
              disabled={loadingPipelines}
              className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-700 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300"
            >
              <option value="">
                {loadingPipelines ? "Loading templates…" : "No template"}
              </option>
              {pipelines.map((p) => (
                <option key={p.name} value={p.name} title={p.description || undefined}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="mb-1 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
              Crawler Config (JSON)
            </label>
            <textarea
              value={crawlerConfigStr}
              onChange={(e) => setCrawlerConfigStr(e.target.value)}
              placeholder={
                loadingConfig
                  ? "Loading config…"
                  : "Leave empty to use defaults, or select a pipeline template above. database.collection is set from collection name on submit."
              }
              rows={12}
              disabled={loadingConfig}
              className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 font-mono text-sm text-zinc-700 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300 disabled:opacity-70"
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
