"use client";

import { useState, useEffect } from "react";
import type { DocumentMetadata } from "@/lib/types";
import { SecurityUserForm } from "./SecurityUserForm";

interface FileProcessingStatus {
  file: File;
  metadata: DocumentMetadata;
  security?: { groupId: string | null; userIds: string[] };
  isProcessing: boolean;
  isReady: boolean;
}

interface MetadataPreviewProps {
  metadata: DocumentMetadata;
  fileName: string;
  fileSize?: number;
  onConfirm: (editedMetadata: DocumentMetadata) => void;
  onCancel: () => void;
  isUploading?: boolean;
  showSecurity?: boolean;
  selectedGroupId?: string | null;
  selectedUserIds?: string[];
  onGroupChange?: (groupId: string | null) => void;
  onUsersChange?: (userIds: string[]) => void;
  onCreateGroup?: () => void;
  // Multi-file support
  files?: FileProcessingStatus[];
  onMetadataUpdate?: (file: File, metadata: DocumentMetadata) => void;
  onAllDocumentsMetadataUpdate?: (metadata: DocumentMetadata) => void;
  onSecurityUpdate?: (file: File, groupId: string | null, userIds: string[]) => void;
  onAllDocumentsSecurityUpdate?: (groupId: string | null, userIds: string[]) => void;
  selectedFile?: File | null;
  onSelectedFileChange?: (file: File | null) => void;
}

export function MetadataPreview({
  metadata,
  fileName,
  fileSize,
  onConfirm,
  onCancel,
  isUploading = false,
  showSecurity = false,
  selectedGroupId = null,
  selectedUserIds = [],
  onGroupChange,
  onUsersChange,
  onCreateGroup,
  files,
  onMetadataUpdate,
  onAllDocumentsMetadataUpdate,
  onSecurityUpdate,
  onAllDocumentsSecurityUpdate,
  selectedFile: propSelectedFile,
  onSelectedFileChange,
}: MetadataPreviewProps) {
  const [editedMetadata, setEditedMetadata] = useState<DocumentMetadata>(metadata);
  const [additionalFields, setAdditionalFields] = useState<Record<string, string>>({});
  const [currentSelectedFile, setCurrentSelectedFile] = useState<File | null>(
    propSelectedFile || null
  );
  const [editedGroupId, setEditedGroupId] = useState<string | null>(selectedGroupId);
  const [editedUserIds, setEditedUserIds] = useState<string[]>(selectedUserIds);
  const isMultiFile = files && files.length > 1;

  // Update currentSelectedFile when prop changes
  useEffect(() => {
    if (propSelectedFile !== undefined) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setCurrentSelectedFile(propSelectedFile);
    }
  }, [propSelectedFile]);

  // Helper function to create a deep copy of metadata
  const deepCopyMetadata = (meta: DocumentMetadata): DocumentMetadata => {
    const copy: DocumentMetadata = {};
    Object.entries(meta).forEach(([key, value]) => {
      if (Array.isArray(value)) {
        copy[key] = [...value];
      } else if (value && typeof value === "object") {
        copy[key] = JSON.parse(JSON.stringify(value));
      } else {
        copy[key] = value;
      }
    });
    return copy;
  };

  // Update metadata and security when selected file changes
  useEffect(() => {
    if (isMultiFile && currentSelectedFile && files) {
      // Individual file selected - show that file's metadata and security (deep copied)
      const fileStatus = files.find((f) => f.file === currentSelectedFile);
      if (fileStatus && fileStatus.isReady) {
        const copiedMetadata = deepCopyMetadata(fileStatus.metadata);
        // eslint-disable-next-line react-hooks/set-state-in-effect
        setEditedMetadata(copiedMetadata);
        // Extract additional fields
        const additional: Record<string, string> = {};
        Object.entries(copiedMetadata).forEach(([key, value]) => {
          if (!["title", "author", "date"].includes(key)) {
            additional[key] = typeof value === "string" ? value : JSON.stringify(value);
          }
        });
        setAdditionalFields(additional);
        // Update security settings
        if (fileStatus.security) {
          setEditedGroupId(fileStatus.security.groupId);
          setEditedUserIds([...fileStatus.security.userIds]);
        } else {
          setEditedGroupId(null);
          setEditedUserIds([]);
        }
      }
    } else if (isMultiFile && currentSelectedFile === null && files) {
      // "All Documents" selected - show template from first ready file (deep copied)
      // This is just for display; when saved, it will overwrite all files
      const firstReady = files.find((f) => f.isReady);
      if (firstReady) {
        const copiedMetadata = deepCopyMetadata(firstReady.metadata);
        setEditedMetadata(copiedMetadata);
        const additional: Record<string, string> = {};
        Object.entries(copiedMetadata).forEach(([key, value]) => {
          if (!["title", "author", "date"].includes(key)) {
            additional[key] = typeof value === "string" ? value : JSON.stringify(value);
          }
        });
        setAdditionalFields(additional);
        // Update security settings from first file
        if (firstReady.security) {
          setEditedGroupId(firstReady.security.groupId);
          setEditedUserIds([...firstReady.security.userIds]);
        } else {
          setEditedGroupId(null);
          setEditedUserIds([]);
        }
      }
    } else {
      // Single file mode
      const copiedMetadata = deepCopyMetadata(metadata);
      setEditedMetadata(copiedMetadata);
      // Extract additional fields (not title, author, date)
      const additional: Record<string, string> = {};
      Object.entries(copiedMetadata).forEach(([key, value]) => {
        if (!["title", "author", "date"].includes(key)) {
          additional[key] = typeof value === "string" ? value : JSON.stringify(value);
        }
      });
      setAdditionalFields(additional);
      // Use prop values for single file mode
      setEditedGroupId(selectedGroupId);
      setEditedUserIds([...selectedUserIds]);
    }
  }, [metadata, currentSelectedFile, files, isMultiFile, selectedGroupId, selectedUserIds]);

  const handleFileSelect = (file: File | null) => {
    setCurrentSelectedFile(file);
    if (onSelectedFileChange) {
      onSelectedFileChange(file);
    }
    // The useEffect will handle updating the metadata display
    // This ensures we always show the correct metadata for the selected file
  };

  const formatFileSize = (bytes?: number): string => {
    if (!bytes) return "Unknown size";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const handleMetadataChange = (key: string, value: string | string[]) => {
    setEditedMetadata((prev) => ({ ...prev, [key]: value }));
  };

  const handleAdditionalFieldChange = (key: string, value: string) => {
    setAdditionalFields((prev) => ({ ...prev, [key]: value }));
    // Try to parse as JSON, otherwise store as string
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

  const handleConfirm = () => {
    if (isMultiFile && files) {
      if (currentSelectedFile === null) {
        // "All Documents" selected - overwrite all files with this metadata and security (deep copied)
        if (onAllDocumentsMetadataUpdate) {
          onAllDocumentsMetadataUpdate(deepCopyMetadata(editedMetadata));
        }
        if (onAllDocumentsSecurityUpdate) {
          onAllDocumentsSecurityUpdate(editedGroupId, [...editedUserIds]);
        }
      } else {
        // Individual file selected - update only that specific file (deep copied)
        if (onMetadataUpdate) {
          onMetadataUpdate(currentSelectedFile, deepCopyMetadata(editedMetadata));
        }
        if (onSecurityUpdate) {
          onSecurityUpdate(currentSelectedFile, editedGroupId, [...editedUserIds]);
        }
      }
    } else {
      // Single file mode
      onConfirm(deepCopyMetadata(editedMetadata));
      // Security is handled via props in single file mode
    }
  };

  // Format author field for display (handle array of strings)
  const formatAuthorForDisplay = (author: unknown): string => {
    if (Array.isArray(author)) {
      return author.join(", ");
    }
    if (typeof author === "string") {
      // Handle comma-separated string without spaces
      return author.split(",").map((a) => a.trim()).join(", ");
    }
    return String(author || "");
  };

  // Parse author input back to array format
  const parseAuthorInput = (value: string): string[] => {
    return value.split(",").map((a) => a.trim()).filter((a) => a.length > 0);
  };

  const allFilesReady = files ? files.every((f) => f.isReady) : true;
  const displayFileName = currentSelectedFile
    ? currentSelectedFile.name
    : isMultiFile
      ? "All Documents"
      : fileName;
  const displayFileSize = currentSelectedFile
    ? currentSelectedFile.size
    : fileSize;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="w-full max-w-4xl max-h-[90vh] overflow-y-auto rounded-lg border border-zinc-200 bg-white shadow-xl dark:border-zinc-800 dark:bg-zinc-900">
        <div className="border-b border-zinc-200 p-6 dark:border-zinc-800">
          <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
            Review Document{isMultiFile ? "s" : ""}
          </h2>
          {isMultiFile && files ? (
            <div className="mt-4">
              <label className="mb-2 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                Select Document
              </label>
              <select
                value={currentSelectedFile ? currentSelectedFile.name : "all"}
                onChange={(e) => {
                  if (e.target.value === "all") {
                    handleFileSelect(null);
                  } else {
                    const file = files.find((f) => f.file.name === e.target.value)?.file;
                    if (file) {
                      handleFileSelect(file);
                    }
                  }
                }}
                className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100 dark:focus:border-blue-400"
              >
                <option value="all">All Documents</option>
                {files.map((fileStatus) => (
                  <option key={fileStatus.file.name} value={fileStatus.file.name}>
                    {fileStatus.file.name}
                    {fileStatus.isProcessing
                      ? " (Processing...)"
                      : fileStatus.isReady
                        ? ""
                        : " (Loading...)"}
                  </option>
                ))}
              </select>
              {currentSelectedFile && (
                <div className="mt-2 flex items-center gap-2 text-xs text-zinc-500 dark:text-zinc-400">
                  {files.find((f) => f.file === currentSelectedFile)?.isProcessing && (
                    <>
                      <div className="h-3 w-3 animate-spin rounded-full border-2 border-zinc-300 border-t-blue-600 dark:border-zinc-700 dark:border-t-blue-400"></div>
                      <span>Processing metadata...</span>
                    </>
                  )}
                </div>
              )}
              <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
                {displayFileName} ({formatFileSize(displayFileSize)})
              </p>
            </div>
          ) : (
            <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
              {fileName} ({formatFileSize(fileSize)})
            </p>
          )}
        </div>

        <div className="p-6 space-y-6">
          {showSecurity && onGroupChange && onUsersChange && (
            <div>
              <h3 className="mb-4 text-lg font-semibold text-zinc-900 dark:text-zinc-100">
                Security & User Assignment
              </h3>
              <SecurityUserForm
                selectedGroupId={editedGroupId || null}
                selectedUserIds={editedUserIds || []}
                onGroupChange={(groupId) => {
                  setEditedGroupId(groupId);
                  if (isMultiFile && currentSelectedFile === null) {
                    // "All Documents" - don't update parent yet, wait for confirm
                  } else if (isMultiFile && currentSelectedFile) {
                    // Individual file - update immediately
                    if (onSecurityUpdate) {
                      onSecurityUpdate(currentSelectedFile, groupId, editedUserIds);
                    }
                  } else {
                    // Single file mode - update parent
                    onGroupChange(groupId);
                  }
                }}
                onUsersChange={(userIds) => {
                  setEditedUserIds(userIds);
                  if (isMultiFile && currentSelectedFile === null) {
                    // "All Documents" - don't update parent yet, wait for confirm
                  } else if (isMultiFile && currentSelectedFile) {
                    // Individual file - update immediately
                    if (onSecurityUpdate) {
                      onSecurityUpdate(currentSelectedFile, editedGroupId, userIds);
                    }
                  } else {
                    // Single file mode - update parent
                    onUsersChange(userIds);
                  }
                }}
                onCreateGroup={onCreateGroup}
              />
            </div>
          )}

          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
              Metadata
            </h3>
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
                Authors
              </label>
              <input
                type="text"
                value={formatAuthorForDisplay(editedMetadata.author)}
                onChange={(e) => {
                  // Store as array for consistency with backend format
                  const authors = parseAuthorInput(e.target.value);
                  handleMetadataChange("author", authors.length > 0 ? authors : e.target.value);
                }}
                className="mt-1 w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100 dark:focus:border-blue-400"
                placeholder="Enter authors (comma-separated)"
              />
              <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                Separate multiple authors with commas
              </p>
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
                      onChange={(e) => handleAdditionalFieldChange(key, e.target.value)}
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
                    No additional metadata fields. Click &quot;Add Field&quot; to add one.
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="border-t border-zinc-200 p-6 dark:border-zinc-800">
          {showSecurity && editedUserIds.length === 0 && (
            <p className="mb-3 text-sm text-red-600 dark:text-red-400">
              Please select at least one user before uploading
            </p>
          )}
          {isMultiFile && !allFilesReady && (
            <p className="mb-3 text-sm text-amber-600 dark:text-amber-400">
              Please wait for all documents to finish processing before submitting
            </p>
          )}
          <div className="flex justify-end gap-3">
            <button
              onClick={onCancel}
              disabled={isUploading}
              className="rounded-lg border border-zinc-300 bg-white px-4 py-2 text-sm font-medium text-zinc-700 transition-colors hover:bg-zinc-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
            >
              Cancel
            </button>
            <button
              onClick={handleConfirm}
              disabled={
                isUploading ||
                (showSecurity && editedUserIds.length === 0) ||
                (isMultiFile && !allFilesReady)
              }
              className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-blue-500 dark:hover:bg-blue-600"
            >
              {isUploading
                ? "Uploading..."
                : isMultiFile
                  ? currentSelectedFile === null
                    ? "Apply to All Documents"
                    : `Update ${currentSelectedFile.name}`
                  : `Submit ${fileName}`}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

