"use client";

import type { Document } from "@/lib/types";
import { getDocumentDisplayInfo } from "@/lib/utils";

interface SearchDocumentModalProps {
  isOpen: boolean;
  onClose: () => void;
  document: Document | null;
}

/**
 * Modal showing a search result document: metadata (title, authors) and markdown body.
 */
export function SearchDocumentModal({
  isOpen,
  onClose,
  document: doc,
}: SearchDocumentModalProps) {
  if (!isOpen || !doc) return null;

  const { displayTitle, displayAuthors } = getDocumentDisplayInfo(doc);
  const metadata = doc.metadata ?? {};
  const title = metadata.title ?? displayTitle;
  const authors = displayAuthors;
  const date = metadata.date != null ? String(metadata.date) : undefined;
  const markdown = doc.markdown?.trim() ?? "";

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-4xl max-h-[90vh] overflow-y-auto rounded-lg border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900 p-6 shadow-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-6 flex items-center justify-between">
          <h2 className="text-xl font-bold text-zinc-900 dark:text-zinc-100 truncate pr-8">
            {displayTitle}
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1 text-zinc-500 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-300"
            aria-label="Close"
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

        <div className="space-y-6">
          {/* Metadata section: title and authors (at least); optional date and other fields */}
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-zinc-600 dark:text-zinc-400">
              Metadata
            </h3>
            <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-800/50 space-y-2">
              <div>
                <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
                  Title
                </span>
                <p className="text-zinc-900 dark:text-zinc-100">{title || "—"}</p>
              </div>
              <div>
                <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
                  Authors
                </span>
                <p className="text-zinc-900 dark:text-zinc-100">
                  {authors || "—"}
                </p>
              </div>
              {date && (
                <div>
                  <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
                    Date
                  </span>
                  <p className="text-zinc-900 dark:text-zinc-100">{date}</p>
                </div>
              )}
              {doc.source && (
                <div>
                  <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
                    Source
                  </span>
                  <p className="text-sm text-zinc-700 dark:text-zinc-300 truncate">
                    {doc.source}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Markdown section */}
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-zinc-600 dark:text-zinc-400">
              Content
            </h3>
            <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-800">
              {markdown ? (
                <pre className="whitespace-pre-wrap font-sans text-sm text-zinc-700 dark:text-zinc-300 overflow-x-auto">
                  {markdown}
                </pre>
              ) : (
                <p className="text-sm text-zinc-500 dark:text-zinc-400">
                  No content
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
