"use client";

import { Button } from "@/components/ui/button";
import type { UseSearchReturn } from "@/lib/hooks";

interface SearchPanelProps {
  selectedCollection: string | null;
  search: UseSearchReturn;
}

/**
 * Search panel component with input, button, and results display.
 */
export function SearchPanel({ selectedCollection, search }: SearchPanelProps) {
  return (
    <div>
      <h2 className="mb-4 text-xl font-semibold text-zinc-900 dark:text-zinc-100">
        Search in {selectedCollection || "..."}
      </h2>
      <div className="flex gap-2 mb-4">
        <input
          className="w-full rounded-lg border border-zinc-300 px-3 py-2 text-sm dark:bg-zinc-800 dark:border-zinc-700"
          placeholder="Search documents..."
          value={search.query}
          onChange={(e) => search.setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && search.performSearch()}
          disabled={!selectedCollection || search.isSearching}
        />
        <Button
          variant="outline"
          onClick={search.performSearch}
          disabled={!selectedCollection || !search.query || search.isSearching}
        >
          {search.isSearching ? "..." : "Search"}
        </Button>
      </div>
      {search.error && (
        <div className="p-3 text-red-600 bg-red-50 rounded-lg text-sm">{search.error}</div>
      )}

      <div className="space-y-2 max-h-[600px] overflow-y-auto">
        {search.results.map((res, i) => (
          <div
            key={i}
            className="p-4 bg-white border rounded-lg hover:shadow-md dark:bg-zinc-900 dark:border-zinc-800"
          >
            <h3 className="font-semibold">{res.metadata?.title || res.source || "Untitled"}</h3>
            <p className="text-sm text-zinc-500 line-clamp-2">{res.chunks?.[0] || ""}</p>
          </div>
        ))}
        {search.results.length === 0 && search.query && !search.isSearching && !search.error && (
          <p className="text-sm text-zinc-500">No results found.</p>
        )}
      </div>
    </div>
  );
}

