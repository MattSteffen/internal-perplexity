"use client";

import { useState } from "react";
import { searchCollection } from "@/lib/api";
import type { Document } from "@/lib/types";

export interface UseSearchReturn {
  query: string;
  setQuery: (query: string) => void;
  results: Document[];
  isSearching: boolean;
  error: string | null;
  performSearch: () => Promise<void>;
}

/**
 * Handles search query state, results, and API calls.
 * @param selectedCollection - The collection to search in (null if none selected)
 */
export function useSearch(selectedCollection: string | null): UseSearchReturn {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Document[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const performSearch = async () => {
    if (!selectedCollection || !query.trim()) return;
    setIsSearching(true);
    setError(null);
    try {
      const response = await searchCollection({
        collection: selectedCollection,
        text: query.trim(),
        filters: [],
        limit: 100,
      });
      setResults(response.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
      setResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  return { query, setQuery, results, isSearching, error, performSearch };
}

