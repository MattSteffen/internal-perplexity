"use client";

import { useEffect, useState, useRef } from "react";
import { CollectionsList } from "@/components/CollectionsList";
import { FileUpload } from "@/components/FileUpload";
import { MetadataPreview } from "@/components/MetadataPreview";
import { SecurityUserForm } from "@/components/SecurityUserForm";
import { FileMetadataEditor } from "@/components/FileMetadataEditor";
import { SubmitSection } from "@/components/SubmitSection";
import { CollectionSchemaDisplay } from "@/components/CollectionSchemaDisplay";
import { CreateCollectionModal } from "@/components/CreateCollectionModal";
import { Button } from "@/components/ui/button";
import { fetchCollections, processDocument, uploadDocument, createCollection } from "@/lib/api";
import type { Collection, DocumentMetadata } from "@/lib/types";
import type { CreateCollectionData } from "@/components/CreateCollectionModal";

export default function Home() {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [selectedCollection, setSelectedCollection] = useState<string | null>(
    null,
  );
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [processedMetadata, setProcessedMetadata] =
    useState<DocumentMetadata | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<{
    current: number;
    total: number;
    fileName: string;
  } | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [selectedGroupId, setSelectedGroupId] = useState<string | null>(null);
  const [selectedUserIds, setSelectedUserIds] = useState<string[]>([]);
  const [fileMetadataMap, setFileMetadataMap] = useState<
    Map<File, DocumentMetadata>
  >(new Map());
  const [fileSecurityMap, setFileSecurityMap] = useState<
    Map<File, { groupId: string | null; userIds: string[] }>
  >(new Map());
  const [fileProcessingStatus, setFileProcessingStatus] = useState<
    Map<File, { isProcessing: boolean; isReady: boolean }>
  >(new Map());
  const [isProcessingMetadata, setIsProcessingMetadata] = useState(false);
  const [confirmSecurity, setConfirmSecurity] = useState(true);
  const [confirmMetadata, setConfirmMetadata] = useState(false);
  const [confirmUpload, setConfirmUpload] = useState(false);
  const [clearDropzone, setClearDropzone] = useState(false);
  const [modalSelectedFile, setModalSelectedFile] = useState<File | null>(null);
  const [isCreateCollectionModalOpen, setIsCreateCollectionModalOpen] = useState(false);
  const firstFileReadyRef = useRef(false);

  useEffect(() => {
    loadCollections();
  }, []);

  const loadCollections = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const data = await fetchCollections();
      // Ensure data is an array
      setCollections(Array.isArray(data) ? data : []);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load collections",
      );
      setCollections([]); // Reset to empty array on error
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileSelect = async (files: File[]) => {
    if (!selectedCollection) {
      alert("Please select a collection first");
      return;
    }

    // Store selected files
    setSelectedFiles(files);
    console.log("selectedFiles", selectedFiles.length, selectedFiles);

    // Initialize processing status for all files
    const statusMap = new Map<File, { isProcessing: boolean; isReady: boolean }>();
    files.forEach((file) => {
      statusMap.set(file, { isProcessing: false, isReady: false });
    });
    setFileProcessingStatus(statusMap);

    // If either confirmMetadata or confirmSecurity is checked, process and show modal
    if (confirmMetadata || confirmSecurity) {
      // Process metadata if confirmMetadata is checked
      if (confirmMetadata) {
        try {
          setIsProcessingMetadata(true);
          setError(null);
          const metadataMap = new Map<File, DocumentMetadata>();
          firstFileReadyRef.current = false; // Reset ref for this batch

          // Process all files in parallel
          const processingPromises = files.map(async (file) => {
            // Mark as processing
            setFileProcessingStatus((prev) => {
              const newMap = new Map(prev);
              newMap.set(file, { isProcessing: true, isReady: false });
              return newMap;
            });

            try {
              const result = await processDocument(selectedCollection, file);
              metadataMap.set(file, result.metadata);

              // Mark as ready
              setFileProcessingStatus((prev) => {
                const newMap = new Map(prev);
                newMap.set(file, { isProcessing: false, isReady: true });
                return newMap;
              });

              // Show modal when first file is ready (use ref to ensure only first one sets it)
              if (!firstFileReadyRef.current) {
                firstFileReadyRef.current = true;
                setSelectedFile(file);
                setProcessedMetadata(result.metadata);
                setModalSelectedFile(null); // Start with "All Documents" selected
              }
            } catch (err) {
              // Mark as failed (not ready)
              setFileProcessingStatus((prev) => {
                const newMap = new Map(prev);
                newMap.set(file, { isProcessing: false, isReady: false });
                return newMap;
              });
              throw err;
            }
          });

          await Promise.all(processingPromises);
          setFileMetadataMap(metadataMap);

          // Initialize security map for all files
          const securityMap = new Map<File, { groupId: string | null; userIds: string[] }>();
          files.forEach((file) => {
            securityMap.set(file, { groupId: null, userIds: [] });
          });
          setFileSecurityMap(securityMap);

          // Always ensure modal is shown after processing completes
          // Use the first file that has metadata (meaning it was successfully processed)
          const firstReadyFile = files.find((file) => metadataMap.has(file)) || files[0];
          if (firstReadyFile) {
            setSelectedFile(firstReadyFile);
            setProcessedMetadata(metadataMap.get(firstReadyFile) || {});
            setModalSelectedFile(null);
          }
        } catch (err) {
          setError(
            err instanceof Error
              ? err.message
              : "Failed to process document metadata",
          );
          setSelectedFile(null);
          setProcessedMetadata(null);
          setSelectedFiles([]);
          setFileMetadataMap(new Map());
          setFileProcessingStatus(new Map());
          setModalSelectedFile(null);
        } finally {
          setIsProcessingMetadata(false);
        }
      } else {
        // If only security is checked, just show modal without processing
        const metadataMap = new Map<File, DocumentMetadata>();
        const securityMap = new Map<File, { groupId: string | null; userIds: string[] }>();
        files.forEach((file) => {
          metadataMap.set(file, {});
          securityMap.set(file, { groupId: null, userIds: [] });
          statusMap.set(file, { isProcessing: false, isReady: true });
        });
        setFileMetadataMap(metadataMap);
        setFileSecurityMap(securityMap);
        setFileProcessingStatus(statusMap);

        // Show modal for security confirmation
        setSelectedFile(files[0]);
        setProcessedMetadata({});
        setModalSelectedFile(null);
      }
    } else {
      // If neither checkbox is checked, just store empty metadata (ready for direct upload)
      const metadataMap = new Map<File, DocumentMetadata>();
      const securityMap = new Map<File, { groupId: string | null; userIds: string[] }>();
      files.forEach((file) => {
        metadataMap.set(file, {});
        securityMap.set(file, { groupId: null, userIds: [] });
        statusMap.set(file, { isProcessing: false, isReady: true });
      });
      setFileMetadataMap(metadataMap);
      setFileSecurityMap(securityMap);
      setFileProcessingStatus(statusMap);
      setSelectedFile(null);
      setProcessedMetadata(null);
      setModalSelectedFile(null);
    }
  };

  const handleMetadataUpdate = (file: File, metadata: DocumentMetadata) => {
    setFileMetadataMap((prev) => {
      const newMap = new Map(prev);
      // Create a deep copy to ensure each file has its own independent metadata
      const copiedMetadata: DocumentMetadata = {};
      Object.entries(metadata).forEach(([key, value]) => {
        if (Array.isArray(value)) {
          copiedMetadata[key] = [...value];
        } else if (value && typeof value === "object") {
          copiedMetadata[key] = JSON.parse(JSON.stringify(value));
        } else {
          copiedMetadata[key] = value;
        }
      });
      newMap.set(file, copiedMetadata);
      return newMap;
    });
    // Also update processedMetadata if it's the selected file
    if (selectedFile === file || modalSelectedFile === file) {
      setProcessedMetadata(metadata);
    }
  };

  const handleAllDocumentsMetadataUpdate = (metadata: DocumentMetadata) => {
    // Apply metadata to all files - this overwrites individual file metadata
    setFileMetadataMap((prev) => {
      const newMap = new Map(prev);
      selectedFiles.forEach((file) => {
        // Deep copy to ensure each file gets its own independent copy
        const copiedMetadata: DocumentMetadata = {};
        Object.entries(metadata).forEach(([key, value]) => {
          if (Array.isArray(value)) {
            copiedMetadata[key] = [...value];
          } else if (value && typeof value === "object") {
            copiedMetadata[key] = JSON.parse(JSON.stringify(value));
          } else {
            copiedMetadata[key] = value;
          }
        });
        newMap.set(file, copiedMetadata);
      });
      return newMap;
    });
  };

  const handleSecurityUpdate = (file: File, groupId: string | null, userIds: string[]) => {
    setFileSecurityMap((prev) => {
      const newMap = new Map(prev);
      // Deep copy to ensure each file has its own independent security settings
      newMap.set(file, {
        groupId,
        userIds: [...userIds], // Copy array
      });
      return newMap;
    });
    // Also update global state if it's the selected file
    if (selectedFile === file || modalSelectedFile === file) {
      setSelectedGroupId(groupId);
      setSelectedUserIds([...userIds]);
    }
  };

  const handleAllDocumentsSecurityUpdate = (groupId: string | null, userIds: string[]) => {
    // Apply security settings to all files - this overwrites individual file security settings
    setFileSecurityMap((prev) => {
      const newMap = new Map(prev);
      selectedFiles.forEach((file) => {
        // Deep copy to ensure each file gets its own independent copy
        newMap.set(file, {
          groupId,
          userIds: [...userIds], // Copy array
        });
      });
      return newMap;
    });
  };

  const handleSingleFileMetadataUpdate = async (metadata: DocumentMetadata) => {
    if (!selectedFile || !selectedCollection) {
      return;
    }

    // Get security settings for this file
    const fileSecurity = fileSecurityMap.get(selectedFile) || { groupId: null, userIds: [] };

    // Validate that at least one user is selected if security confirmation is enabled
    if (confirmSecurity && fileSecurity.userIds.length === 0) {
      alert("Please select at least one user");
      return;
    }

    // Update metadata
    setProcessedMetadata(metadata);
    handleMetadataUpdate(selectedFile, metadata);

    // Upload the file with updated metadata
    try {
      setIsUploading(true);
      setError(null);
      await uploadDocument(selectedCollection, selectedFile, metadata);
      // Reset state after successful upload
      setSelectedFile(null);
      setProcessedMetadata(null);
      setSelectedFiles([]);
      setSelectedGroupId(null);
      setSelectedUserIds([]);
      setFileMetadataMap(new Map());
      setClearDropzone(true);
      // Reload collections to show updated data
      await loadCollections();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to upload document",
      );
    } finally {
      setIsUploading(false);
      // Reset clearDropzone after a brief delay to allow the component to react
      setTimeout(() => setClearDropzone(false), 100);
    }
  };

  const handleConfirmUpload = async () => {
    if (!selectedCollection) {
      return;
    }

    // Validate security requirements per file
    if (confirmSecurity) {
      const filesWithoutUsers = selectedFiles.filter((file) => {
        const security = fileSecurityMap.get(file);
        return !security || security.userIds.length === 0;
      });
      if (filesWithoutUsers.length > 0) {
        alert("Please select at least one user for all documents");
        return;
      }
    }

    // If no checkboxes are checked, validate that at least one user is selected globally
    if (!confirmSecurity && !confirmMetadata) {
      const allFilesHaveUsers = selectedFiles.every((file) => {
        const security = fileSecurityMap.get(file);
        return security && security.userIds.length > 0;
      });
      if (!allFilesHaveUsers && selectedUserIds.length === 0) {
        alert("Please select at least one user");
        return;
      }
    }

    try {
      setIsUploading(true);
      setError(null);

      // Single file upload (with metadata preview)
      if (selectedFile && processedMetadata !== null) {
        await uploadDocument(selectedCollection, selectedFile, processedMetadata);
        // Reset state after successful upload
        setSelectedFile(null);
        setProcessedMetadata(null);
        setSelectedFiles([]);
        setSelectedGroupId(null);
        setSelectedUserIds([]);
        setFileMetadataMap(new Map());
        setFileSecurityMap(new Map());
        setFileProcessingStatus(new Map());
        setModalSelectedFile(null);
        setClearDropzone(true);
        // Reload collections to show updated data
        await loadCollections();
        return;
      }

      // Multiple files upload
      if (selectedFiles.length > 0) {
        // Check that all files are ready
        const allReady = selectedFiles.every(
          (file) => fileProcessingStatus.get(file)?.isReady === true,
        );
        if (!allReady) {
          alert("Please wait for all documents to finish processing");
          return;
        }

        setUploadProgress({
          current: 0,
          total: selectedFiles.length,
          fileName: "",
        });

        // Upload all files in parallel
        const uploadPromises = selectedFiles.map(async (file, index) => {
          const metadata = fileMetadataMap.get(file) || {};
          setUploadProgress({
            current: index + 1,
            total: selectedFiles.length,
            fileName: file.name,
          });
          return uploadDocument(selectedCollection, file, metadata);
        });

        await Promise.all(uploadPromises);

        // Reset state after successful upload
        setSelectedFiles([]);
        setSelectedGroupId(null);
        setSelectedUserIds([]);
        setUploadProgress(null);
        setFileMetadataMap(new Map());
        setFileSecurityMap(new Map());
        setFileProcessingStatus(new Map());
        setModalSelectedFile(null);
        setClearDropzone(true);
        // Reload collections to show updated data
        await loadCollections();
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to upload document(s)",
      );
    } finally {
      setIsUploading(false);
      // Reset clearDropzone after a brief delay to allow the component to react
      setTimeout(() => setClearDropzone(false), 100);
    }
  };

  const handleCancelPreview = () => {
    setSelectedFile(null);
    setProcessedMetadata(null);
    setSelectedFiles([]);
    setSelectedGroupId(null);
    setSelectedUserIds([]);
    setFileMetadataMap(new Map());
    setFileSecurityMap(new Map());
    setFileProcessingStatus(new Map());
    setModalSelectedFile(null);
  };

  const handleCreateGroup = () => {
    // TODO: Implement role creation API call
    alert("Role creation not yet implemented");
  };

  const handleSelectCollection = (collectionName: string) => {
    // Toggle selection: if clicking the already-selected collection, unselect it
    if (selectedCollection === collectionName) {
      setSelectedCollection(null);
    } else {
      setSelectedCollection(collectionName);
    }
  };

  const handleCreateCollection = async (data: CreateCollectionData) => {
    try {
      await createCollection({
        collection_name: data.collection_name,
        pipeline_name: data.pipeline_name,
        custom_config: data.custom_config,
        config_overrides: data.config_overrides,
        description: data.description,
        default_permissions: data.default_permissions,
        metadata_schema: data.metadata_schema,
      });
      // Reload collections after creation
      await loadCollections();
    } catch (err) {
      throw err; // Re-throw to let modal handle error display
    }
  };

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-black">
      <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100">
                Document Upload
              </h1>
              <p className="mt-2 text-zinc-600 dark:text-zinc-400">
                Select a collection and upload PDF documents
              </p>
            </div>
            <Button
              variant="outline"
              asChild
            >
              <a href={process.env.NEXT_PUBLIC_CHAT_URL || "http://localhost:3000"} target="_self">
                Chat
              </a>
            </Button>
          </div>
        </div>

        {error && (
          <div className="mb-6 rounded-lg border border-red-300 bg-red-50 p-4 dark:border-red-800 dark:bg-red-950/30">
            <div className="flex items-center gap-2">
              <svg
                className="h-5 w-5 text-red-600 dark:text-red-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <p className="text-sm font-medium text-red-800 dark:text-red-300">
                {error}
              </p>
            </div>
          </div>
        )}

        <div className="grid gap-8 lg:grid-cols-2">
          <div>
            {isLoading ? (
              <div className="rounded-lg border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
                <p className="text-center text-zinc-500 dark:text-zinc-400">
                  Loading collections...
                </p>
              </div>
            ) : (
              <CollectionsList
                collections={collections}
                selectedCollection={selectedCollection}
                onSelectCollection={handleSelectCollection}
                onRefresh={loadCollections}
                isRefreshing={isLoading}
                onCreateCollection={() => setIsCreateCollectionModalOpen(true)}
              />
            )}
          </div>

          <div>
            <h2 className="mb-4 text-xl font-semibold text-zinc-900 dark:text-zinc-100">
              Upload Document
            </h2>
            {isProcessing || isProcessingMetadata ? (
              <div className="rounded-lg border border-zinc-200 bg-white p-8 text-center dark:border-zinc-800 dark:bg-zinc-900">
                <div className="flex flex-col items-center gap-4">
                  <div className="h-8 w-8 animate-spin rounded-full border-4 border-zinc-300 border-t-blue-600 dark:border-zinc-700 dark:border-t-blue-400"></div>
                  <p className="text-zinc-600 dark:text-zinc-400">
                    {uploadProgress
                      ? `Processing ${uploadProgress.current} of ${uploadProgress.total}: ${uploadProgress.fileName}`
                      : "Processing document..."}
                  </p>
                  {uploadProgress && (
                    <div className="w-full max-w-xs">
                      <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-700">
                        <div
                          className="h-full bg-blue-600 transition-all duration-300 dark:bg-blue-500"
                          style={{
                            width: `${(uploadProgress.current / uploadProgress.total) * 100}%`,
                          }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <>
                <FileUpload
                  onFileSelect={handleFileSelect}
                  disabled={!selectedCollection}
                  clearFiles={clearDropzone}
                />
                {selectedCollection && (
                  <div className="mt-4 space-y-3 rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
                    <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                      Upload Options
                    </h3>
                    <div className="space-y-2">
                      <label className="flex cursor-pointer items-center gap-2">
                        <input
                          type="checkbox"
                          checked={confirmSecurity}
                          onChange={(e) => setConfirmSecurity(e.target.checked)}
                          className="h-4 w-4 rounded border-zinc-300 text-blue-600 focus:ring-2 focus:ring-blue-500/20 dark:border-zinc-600 dark:bg-zinc-800"
                        />
                        <span className="text-sm text-zinc-700 dark:text-zinc-300">
                          Add extra security roles
                        </span>
                      </label>
                      <label className="flex cursor-pointer items-center gap-2">
                        <input
                          type="checkbox"
                          checked={confirmMetadata}
                          onChange={(e) => setConfirmMetadata(e.target.checked)}
                          className="h-4 w-4 rounded border-zinc-300 text-blue-600 focus:ring-2 focus:ring-blue-500/20 dark:border-zinc-600 dark:bg-zinc-800"
                        />
                        <span className="text-sm text-zinc-700 dark:text-zinc-300">
                          Confirm metadata
                        </span>
                      </label>
                      <label className="flex cursor-pointer items-center gap-2">
                        <input
                          type="checkbox"
                          checked={confirmUpload}
                          onChange={(e) => setConfirmUpload(e.target.checked)}
                          className="h-4 w-4 rounded border-zinc-300 text-blue-600 focus:ring-2 focus:ring-blue-500/20 dark:border-zinc-600 dark:bg-zinc-800"
                        />
                        <span className="text-sm text-zinc-700 dark:text-zinc-300">
                          Confirm upload
                        </span>
                      </label>
                    </div>
                  </div>
                )}
                {selectedFiles.length > 0 && !isProcessingMetadata && (
                  <>
                    {!confirmSecurity && (
                      <SecurityUserForm
                        selectedGroupId={selectedGroupId}
                        selectedUserIds={selectedUserIds}
                        onGroupChange={setSelectedGroupId}
                        onUsersChange={setSelectedUserIds}
                        onCreateGroup={handleCreateGroup}
                      />
                    )}
                    {selectedFiles.length > 1 && confirmMetadata && (
                      <FileMetadataEditor
                        files={selectedFiles.map((file) => ({
                          file,
                          metadata: fileMetadataMap.get(file) || {},
                        }))}
                        onMetadataChange={(fileIndex, metadata) => {
                          const file = selectedFiles[fileIndex];
                          if (file) {
                            handleMetadataUpdate(file, metadata);
                          }
                        }}
                      />
                    )}
                    <SubmitSection
                      files={selectedFiles}
                      onSubmit={handleConfirmUpload}
                      onCancel={handleCancelPreview}
                      isUploading={isUploading}
                      disabled={
                        (!confirmSecurity &&
                          !confirmMetadata &&
                          selectedUserIds.length === 0 &&
                          selectedFiles.some(
                            (file) =>
                              !fileSecurityMap.get(file) ||
                              fileSecurityMap.get(file)?.userIds.length === 0,
                          )) ||
                        (confirmSecurity &&
                          selectedFiles.some(
                            (file) =>
                              !fileSecurityMap.get(file) ||
                              fileSecurityMap.get(file)?.userIds.length === 0,
                          )) ||
                        (confirmMetadata &&
                          selectedFiles.some(
                            (file) =>
                              fileProcessingStatus.get(file)?.isReady !== true,
                          ))
                      }
                    />
                  </>
                )}
              </>
            )}
            {selectedCollection && (
              <CollectionSchemaDisplay
                collection={
                  collections.find((c) => c.name === selectedCollection) || null
                }
              />
            )}
          </div>
        </div>

        {selectedFile &&
          !isProcessingMetadata &&
          (confirmMetadata || confirmSecurity) && (
            <MetadataPreview
              metadata={processedMetadata || {}}
              fileName={selectedFile.name}
              fileSize={selectedFile.size}
              onConfirm={handleSingleFileMetadataUpdate}
              onCancel={handleCancelPreview}
              isUploading={isUploading}
              hasSelectedUsers={
                !confirmSecurity ||
                (modalSelectedFile
                  ? (fileSecurityMap.get(modalSelectedFile)?.userIds.length || 0) > 0
                  : selectedUserIds.length > 0)
              }
              showSecurity={confirmSecurity}
              selectedGroupId={
                modalSelectedFile
                  ? fileSecurityMap.get(modalSelectedFile)?.groupId || null
                  : selectedGroupId
              }
              selectedUserIds={
                modalSelectedFile
                  ? fileSecurityMap.get(modalSelectedFile)?.userIds || []
                  : selectedUserIds
              }
              onGroupChange={(groupId) => {
                if (modalSelectedFile) {
                  const security = fileSecurityMap.get(modalSelectedFile) || {
                    groupId: null,
                    userIds: [],
                  };
                  handleSecurityUpdate(modalSelectedFile, groupId, security.userIds);
                } else {
                  setSelectedGroupId(groupId);
                }
              }}
              onUsersChange={(userIds) => {
                if (modalSelectedFile) {
                  const security = fileSecurityMap.get(modalSelectedFile) || {
                    groupId: null,
                    userIds: [],
                  };
                  handleSecurityUpdate(modalSelectedFile, security.groupId, userIds);
                } else {
                  setSelectedUserIds(userIds);
                }
              }}
              onCreateGroup={handleCreateGroup}
              files={
                selectedFiles.length > 1
                  ? selectedFiles.map((file) => ({
                    file,
                    metadata: fileMetadataMap.get(file) || {},
                    security: fileSecurityMap.get(file) || {
                      groupId: null,
                      userIds: [],
                    },
                    isProcessing:
                      fileProcessingStatus.get(file)?.isProcessing || false,
                    isReady: fileProcessingStatus.get(file)?.isReady || false,
                  }))
                  : undefined
              }
              onMetadataUpdate={handleMetadataUpdate}
              onAllDocumentsMetadataUpdate={handleAllDocumentsMetadataUpdate}
              onSecurityUpdate={handleSecurityUpdate}
              onAllDocumentsSecurityUpdate={handleAllDocumentsSecurityUpdate}
              selectedFile={modalSelectedFile}
              onSelectedFileChange={setModalSelectedFile}
            />
          )}

        <CreateCollectionModal
          isOpen={isCreateCollectionModalOpen}
          onClose={() => setIsCreateCollectionModalOpen(false)}
          onCreate={handleCreateCollection}
        />
      </main>
    </div>
  );
}
