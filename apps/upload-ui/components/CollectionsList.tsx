"use client";

import { useState, useMemo } from "react";
import type { Collection } from "@/lib/types";
import { Input } from "@/components/ui/input";

interface CollectionsListProps {
  collections: Collection[];
  selectedCollection: string | null;
  onSelectCollection: (name: string) => void;
  onRefresh?: () => void;
  isRefreshing?: boolean;
  onCreateCollection?: () => void;
}

type FilterType = "all" | "public" | "private" | "admin";

export function CollectionsList({
  collections,
  selectedCollection,
  onSelectCollection,
  onRefresh,
  isRefreshing = false,
  onCreateCollection,
}: CollectionsListProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [filterType, setFilterType] = useState<FilterType>("all");

  // Safety check: ensure collections is always an array
  const safeCollections = useMemo(
    () => (Array.isArray(collections) ? collections : []),
    [collections]
  );

  // Filter collections based on search query and filter type
  const filteredCollections = useMemo(() => {
    return safeCollections.filter((collection) => {
      // Search filter: match name or description
      const matchesSearch =
        searchQuery === "" ||
        collection.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (collection.description &&
          collection.description.toLowerCase().includes(searchQuery.toLowerCase()));

      if (!matchesSearch) return false;

      // Filter by access_level
      if (filterType === "all") {
        return true;
      }
      return collection.access_level === filterType;
    });
  }, [safeCollections, searchQuery, filterType]);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
          Collections
        </h2>
        <div className="flex items-center gap-2">
          {onRefresh && (
            <button
              onClick={onRefresh}
              disabled={isRefreshing}
              className="rounded-lg border border-zinc-300 bg-white px-3 py-1.5 text-sm font-medium text-zinc-700 transition-colors hover:bg-zinc-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
              title="Refresh collections"
            >
              <svg
                className={`h-4 w-4 ${isRefreshing ? "animate-spin" : ""}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
            </button>
          )}
          {onCreateCollection && (
            <button
              onClick={onCreateCollection}
              className="rounded-lg border border-zinc-300 bg-white px-3 py-1.5 text-sm font-medium text-zinc-700 transition-colors hover:bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
              title="Create new collection"
            >
              <svg
                className="h-4 w-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 4v16m8-8H4"
                />
              </svg>
            </button>
          )}
        </div>
      </div>

      {/* Search bar */}
      <div className="relative">
        <Input
          type="text"
          placeholder="Search collections by name or description..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full"
        />
        {searchQuery && (
          <button
            onClick={() => setSearchQuery("")}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-300"
            aria-label="Clear search"
          >
            <svg
              className="h-4 w-4"
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
        )}
      </div>

      {/* Filter buttons */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-sm font-medium text-zinc-700 dark:text-zinc-300">Access:</span>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setFilterType("all")}
            className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${filterType === "all"
              ? "border-blue-500 bg-blue-50 text-blue-700 dark:border-blue-400 dark:bg-blue-950/30 dark:text-blue-300"
              : "border-zinc-300 bg-white text-zinc-700 hover:bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
              }`}
          >
            All
          </button>
          <button
            onClick={() => setFilterType("public")}
            className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${filterType === "public"
              ? "border-blue-500 bg-blue-50 text-blue-700 dark:border-blue-400 dark:bg-blue-950/30 dark:text-blue-300"
              : "border-zinc-300 bg-white text-zinc-700 hover:bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
              }`}
          >
            Public
          </button>
          <button
            onClick={() => setFilterType("private")}
            className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${filterType === "private"
              ? "border-blue-500 bg-blue-50 text-blue-700 dark:border-blue-400 dark:bg-blue-950/30 dark:text-blue-300"
              : "border-zinc-300 bg-white text-zinc-700 hover:bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
              }`}
          >
            Private
          </button>
          <button
            onClick={() => setFilterType("admin")}
            className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${filterType === "admin"
              ? "border-blue-500 bg-blue-50 text-blue-700 dark:border-blue-400 dark:bg-blue-950/30 dark:text-blue-300"
              : "border-zinc-300 bg-white text-zinc-700 hover:bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
              }`}
          >
            Admin
          </button>
        </div>
      </div>

      {/* Collections list */}
      {filteredCollections.length === 0 ? (
        <div className="rounded-lg border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-center text-zinc-500 dark:text-zinc-400">
            {safeCollections.length === 0
              ? "No collections found"
              : "No collections match your search or filter"}
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {filteredCollections.map((collection) => (
            <button
              key={collection.name}
              onClick={() => onSelectCollection(collection.name)}
              className={`w-full rounded-lg border p-4 text-left transition-colors ${selectedCollection === collection.name
                ? "border-blue-500 bg-blue-50 dark:border-blue-400 dark:bg-blue-950/30"
                : "border-zinc-200 bg-white hover:border-zinc-300 hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900 dark:hover:border-zinc-700 dark:hover:bg-zinc-800"
                }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <h3 className="font-semibold text-zinc-900 dark:text-zinc-100">
                    {collection.name}
                  </h3>
                  {collection.description && (
                    <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
                      {collection.description}
                    </p>
                  )}
                  <div className="mt-2 flex flex-wrap gap-2">
                    {collection.num_documents !== undefined && (
                      <span className="rounded-full bg-gray-100 px-2 py-1 text-xs text-gray-800 dark:bg-gray-900/30 dark:text-gray-300">
                        {collection.num_documents} {collection.num_documents === 1 ? "document" : "documents"}
                      </span>
                    )}
                    {collection.num_chunks !== undefined && (
                      <span className="rounded-full bg-slate-100 px-2 py-1 text-xs text-slate-800 dark:bg-slate-900/30 dark:text-slate-300">
                        {collection.num_chunks} {collection.num_chunks === 1 ? "chunk" : "chunks"}
                      </span>
                    )}
                    {collection.num_partitions !== undefined && (
                      <span className="rounded-full bg-indigo-100 px-2 py-1 text-xs text-indigo-800 dark:bg-indigo-900/30 dark:text-indigo-300">
                        {collection.num_partitions} {collection.num_partitions === 1 ? "partition" : "partitions"}
                      </span>
                    )}
                    {collection.required_roles && collection.required_roles.length > 0 && (
                      <span className="rounded-full bg-purple-100 px-2 py-1 text-xs text-purple-800 dark:bg-purple-900/30 dark:text-purple-300">
                        Roles: {collection.required_roles.join(", ")}
                      </span>
                    )}
                    {collection.access_level && (
                      <span
                        className={`rounded-full px-2 py-1 text-xs ${
                          collection.access_level === "public"
                            ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300"
                            : collection.access_level === "private"
                            ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300"
                            : "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300"
                        }`}
                      >
                        {collection.access_level === "public"
                          ? "Public"
                          : collection.access_level === "private"
                          ? "Private"
                          : "Admin"}
                      </span>
                    )}
                    {collection.security_rules &&
                      collection.security_rules.length > 0 &&
                      collection.security_rules.map((rule, idx) => (
                        <span
                          key={idx}
                          className={`rounded-full px-2 py-1 text-xs ${rule.required
                            ? "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300"
                            : "bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300"
                            }`}
                        >
                          {rule.name}
                          {rule.required && " *"}
                        </span>
                      ))}
                  </div>
                </div>
                {selectedCollection === collection.name && (
                  <svg
                    className="ml-2 h-5 w-5 text-blue-500 dark:text-blue-400"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M5 13l4 4L19 7"
                    />
                  </svg>
                )}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

