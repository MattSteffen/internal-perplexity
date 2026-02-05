"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import type { Document } from "@/lib/types";
import { getDocumentDisplayInfo } from "@/lib/utils";
import type { UseSearchReturn as UseSearchReturnType } from "@/lib/hooks";
import { SearchDocumentModal } from "./SearchDocumentModal";

interface SearchPanelProps {
  selectedCollection: string | null;
  search: UseSearchReturnType;
}

/**
 * Search panel component with input, button, results display (title/authors or first line of markdown),
 * and click-to-open document modal with metadata and markdown.
 */
export function SearchPanel({ selectedCollection, search }: SearchPanelProps) {
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(
    null,
  );
  const [modalOpen, setModalOpen] = useState(false);

  const openDocument = (doc: Document) => {
    setSelectedDocument(doc);
    setModalOpen(true);
  };

  const closeModal = () => {
    setModalOpen(false);
    setSelectedDocument(null);
  };

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
        <div className="p-3 text-red-600 bg-red-50 rounded-lg text-sm dark:bg-red-900/20 dark:text-red-400">
          {search.error}
        </div>
      )}

      <div className="space-y-2 max-h-[600px] overflow-y-auto">
        {search.results.map((res, i) => {
          const { displayTitle, displayAuthors } =
            getDocumentDisplayInfo(res);
          return (
            <button
              type="button"
              key={res.document_id ?? i}
              className="w-full text-left p-4 bg-white border rounded-lg hover:shadow-md dark:bg-zinc-900 dark:border-zinc-800 focus:outline-none focus:ring-2 focus:ring-zinc-400 dark:focus:ring-zinc-600"
              onClick={() => openDocument(res)}
            >
              <h3 className="font-semibold text-zinc-900 dark:text-zinc-100">
                {displayTitle}
              </h3>
              {displayAuthors ? (
                <p className="text-sm text-zinc-600 dark:text-zinc-400 mt-0.5">
                  {displayAuthors}
                </p>
              ) : null}
            </button>
          );
        })}
        {search.results.length === 0 &&
          search.query &&
          !search.isSearching &&
          !search.error && (
            <p className="text-sm text-zinc-500">No results found.</p>
          )}
      </div>

      <SearchDocumentModal
        isOpen={modalOpen}
        onClose={closeModal}
        document={selectedDocument}
      />
    </div>
  );
}
