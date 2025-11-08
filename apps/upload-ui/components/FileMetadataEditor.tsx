"use client";

import { useState, useEffect } from "react";
import type { DocumentMetadata } from "@/lib/types";

interface FileMetadata {
  file: File;
  metadata: DocumentMetadata;
}

interface FileMetadataEditorProps {
  files: FileMetadata[];
  onMetadataChange: (fileIndex: number, metadata: DocumentMetadata) => void;
}

export function FileMetadataEditor({
  files,
  onMetadataChange,
}: FileMetadataEditorProps) {
  const [expandedFile, setExpandedFile] = useState<number | null>(null);

  if (files.length === 0) {
    return null;
  }

  return (
    <div className="mt-6 rounded-lg border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
      <h3 className="mb-4 text-lg font-semibold text-zinc-900 dark:text-zinc-100">
        Edit File Metadata
      </h3>
      <div className="space-y-3">
        {files.map((fileMetadata, index) => (
          <FileMetadataItem
            key={index}
            file={fileMetadata.file}
            metadata={fileMetadata.metadata}
            isExpanded={expandedFile === index}
            onToggle={() =>
              setExpandedFile(expandedFile === index ? null : index)
            }
            onMetadataChange={(metadata) => onMetadataChange(index, metadata)}
          />
        ))}
      </div>
    </div>
  );
}

interface FileMetadataItemProps {
  file: File;
  metadata: DocumentMetadata;
  isExpanded: boolean;
  onToggle: () => void;
  onMetadataChange: (metadata: DocumentMetadata) => void;
}

function FileMetadataItem({
  file,
  metadata,
  isExpanded,
  onToggle,
  onMetadataChange,
}: FileMetadataItemProps) {
  const [editedMetadata, setEditedMetadata] =
    useState<DocumentMetadata>(metadata);
  const [additionalFields, setAdditionalFields] = useState<
    Record<string, string>
  >({});

  useEffect(() => {
    setEditedMetadata(metadata);
    const additional: Record<string, string> = {};
    Object.entries(metadata).forEach(([key, value]) => {
      if (!["title", "author", "date"].includes(key)) {
        additional[key] =
          typeof value === "string" ? value : JSON.stringify(value);
      }
    });
    setAdditionalFields(additional);
  }, [metadata]);

  useEffect(() => {
    // Notify parent of changes
    onMetadataChange(editedMetadata);
  }, [editedMetadata, onMetadataChange]);

  const handleMetadataChange = (key: string, value: string) => {
    setEditedMetadata((prev) => ({ ...prev, [key]: value }));
  };

  const handleAdditionalFieldChange = (key: string, value: string) => {
    setAdditionalFields((prev) => ({ ...prev, [key]: value }));
    try {
      const parsed = JSON.parse(value);
      setEditedMetadata((prev) => ({ ...prev, [key]: parsed }));
    } catch {
      setEditedMetadata((prev) => ({ ...prev, [key]: value }));
    }
  };

  const handleAddField = () => {
    const newKey = `field_${Date.now()}`;
    setAdditionalFields((prev) => ({ ...prev, [newKey]: "" }));
  };

  const handleRemoveField = (key: string) => {
    setAdditionalFields((prev) => {
      const newFields = { ...prev };
      delete newFields[key];
      return newFields;
    });
    setEditedMetadata((prev) => {
      const newMetadata = { ...prev };
      delete newMetadata[key];
      return newMetadata;
    });
  };

  return (
    <div className="rounded-lg border border-zinc-200 dark:border-zinc-700">
      <button
        onClick={onToggle}
        className="w-full px-4 py-3 text-left transition-colors hover:bg-zinc-50 dark:hover:bg-zinc-800"
      >
        <div className="flex items-center justify-between">
          <span className="font-medium text-zinc-900 dark:text-zinc-100">
            {file.name}
          </span>
          <svg
            className={`h-5 w-5 text-zinc-500 transition-transform ${
              isExpanded ? "rotate-180" : ""
            }`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </div>
      </button>

      {isExpanded && (
        <div className="border-t border-zinc-200 p-4 dark:border-zinc-700">
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
                Title
              </label>
              <input
                type="text"
                value={editedMetadata.title || ""}
                onChange={(e) => handleMetadataChange("title", e.target.value)}
                className="mt-1 w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100 dark:focus:border-blue-400"
                placeholder="Enter title"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
                Author
              </label>
              <input
                type="text"
                value={editedMetadata.author || ""}
                onChange={(e) =>
                  handleMetadataChange("author", e.target.value)
                }
                className="mt-1 w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100 dark:focus:border-blue-400"
                placeholder="Enter author"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
                Date
              </label>
              <input
                type="text"
                value={editedMetadata.date || ""}
                onChange={(e) => handleMetadataChange("date", e.target.value)}
                className="mt-1 w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100 dark:focus:border-blue-400"
                placeholder="Enter date"
              />
            </div>

            <div>
              <div className="mb-2 flex items-center justify-between">
                <label className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
                  Additional Metadata
                </label>
                <button
                  onClick={handleAddField}
                  className="text-xs text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
                >
                  + Add Field
                </button>
              </div>
              <div className="space-y-2">
                {Object.entries(additionalFields).map(([key, value]) => (
                  <div key={key} className="flex gap-2">
                    <input
                      type="text"
                      value={key}
                      onChange={(e) => {
                        const oldKey = key;
                        const newKey = e.target.value;
                        setAdditionalFields((prev) => {
                          const newFields = { ...prev };
                          delete newFields[oldKey];
                          newFields[newKey] = value;
                          return newFields;
                        });
                        setEditedMetadata((prev) => {
                          const newMetadata = { ...prev };
                          const oldValue = prev[oldKey];
                          delete newMetadata[oldKey];
                          newMetadata[newKey] = oldValue;
                          return newMetadata;
                        });
                      }}
                      className="w-1/3 rounded-lg border border-zinc-300 bg-white px-2 py-1 text-xs text-zinc-900 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100 dark:focus:border-blue-400"
                      placeholder="Field name"
                    />
                    <input
                      type="text"
                      value={value}
                      onChange={(e) =>
                        handleAdditionalFieldChange(key, e.target.value)
                      }
                      className="flex-1 rounded-lg border border-zinc-300 bg-white px-2 py-1 text-xs text-zinc-900 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100 dark:focus:border-blue-400"
                      placeholder="Field value"
                    />
                    <button
                      onClick={() => handleRemoveField(key)}
                      className="rounded-lg px-2 py-1 text-xs text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-950/30"
                    >
                      Remove
                    </button>
                  </div>
                ))}
                {Object.keys(additionalFields).length === 0 && (
                  <p className="text-xs text-zinc-500 dark:text-zinc-400">
                    No additional metadata fields. Click "Add Field" to add one.
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

