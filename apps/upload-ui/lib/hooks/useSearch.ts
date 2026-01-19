"use client";

import { useState, useEffect, useRef } from "react";
import { searchCollection } from "@/lib/api";
import type { Document } from "@/lib/types";

interface CachedSearchState {
  query: string;
  results: Document[];
}

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
 * Caches search state per collection and restores it when switching collections.
 * @param selectedCollection - The collection to search in (null if none selected)
 */
export function useSearch(selectedCollection: string | null): UseSearchReturn {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Document[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Cache search state per collection
  const cacheRef = useRef<Map<string, CachedSearchState>>(new Map());
  const previousCollectionRef = useRef<string | null>(null);

  // Handle collection changes: save current state and restore cached state
  useEffect(() => {
    // Save current state to cache for previous collection
    if (previousCollectionRef.current) {
      cacheRef.current.set(previousCollectionRef.current, {
        query,
        results,
      });
    }

    // Restore cached state for new collection or clear if none
    if (selectedCollection) {
      const cached = cacheRef.current.get(selectedCollection);
      if (cached) {
        setQuery(cached.query);
        setResults(cached.results);
      } else {
        // New collection: clear state
        setQuery("");
        setResults([]);
      }
    } else {
      // No collection selected: clear display but keep cache
      setQuery("");
      setResults([]);
    }

    // Clear error when switching collections
    setError(null);
    
    // Update previous collection ref
    previousCollectionRef.current = selectedCollection;
  }, [selectedCollection]);

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
      // Update cache with new results
      cacheRef.current.set(selectedCollection, {
        query,
        results: response.results,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
      setResults([]);
      // Update cache even on error (empty results)
      cacheRef.current.set(selectedCollection, {
        query,
        results: [],
      });
    } finally {
      setIsSearching(false);
    }
  };

  return { query, setQuery, results, isSearching, error, performSearch };
}

