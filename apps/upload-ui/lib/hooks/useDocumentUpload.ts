"use client";

import { useState, useRef } from "react";
import { processDocument, uploadDocument } from "@/lib/api";
import type { DocumentMetadata } from "@/lib/types";

export interface UploadConfig {
  security: boolean;
  metadata: boolean;
  upload: boolean;
}

export interface GlobalSecurity {
  groupId: string | null;
  userIds: string[];
}

export interface FileStatus {
  isProcessing: boolean;
  isReady: boolean;
}

export interface UploadProgress {
  current: number;
  total: number;
  fileName: string;
}

export interface UseDocumentUploadReturn {
  // Selection State
  selectedFiles: File[];
  setSelectedFiles: React.Dispatch<React.SetStateAction<File[]>>;
  selectedFile: File | null;
  setSelectedFile: React.Dispatch<React.SetStateAction<File | null>>;

  // Configuration State
  config: UploadConfig;
  setConfig: React.Dispatch<React.SetStateAction<UploadConfig>>;

  // Data Maps
  metadataMap: Map<File, DocumentMetadata>;
  updateMetadata: (file: File, data: DocumentMetadata) => void;
  setMetadataMap: React.Dispatch<React.SetStateAction<Map<File, DocumentMetadata>>>;
  securityMap: Map<File, GlobalSecurity>;
  updateSecurity: (file: File, groupId: string | null, userIds: string[]) => void;
  setSecurityMap: React.Dispatch<React.SetStateAction<Map<File, GlobalSecurity>>>;
  statusMap: Map<File, FileStatus>;

  // Processed State
  processedMetadata: DocumentMetadata | null;
  setProcessedMetadata: React.Dispatch<React.SetStateAction<DocumentMetadata | null>>;

  // Process State
  isProcessing: boolean;
  isUploading: boolean;
  uploadProgress: UploadProgress | null;
  error: string | null;
  clearDropzone: boolean;

  // Global Security
  globalSecurity: GlobalSecurity;
  setGlobalSecurity: React.Dispatch<React.SetStateAction<GlobalSecurity>>;

  // Handlers
  handleFileSelect: (files: File[]) => Promise<void>;
  handleUpload: () => Promise<void>;
  resetState: () => void;
}

/**
 * Handles complex upload state and logic including file selection,
 * metadata/security maps, processing, and upload.
 * @param selectedCollection - The collection to upload to (null if none selected)
 * @param onSuccess - Callback invoked after successful upload
 */
export function useDocumentUpload(
  selectedCollection: string | null,
  onSuccess: () => void
): UseDocumentUploadReturn {
  // Selection State
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null); // Currently previewed

  // Configuration State
  const [config, setConfig] = useState<UploadConfig>({ security: true, metadata: false, upload: false });
  const [globalSecurity, setGlobalSecurity] = useState<GlobalSecurity>({ groupId: null, userIds: [] });

  // Data Maps
  const [metadataMap, setMetadataMap] = useState<Map<File, DocumentMetadata>>(new Map());
  const [securityMap, setSecurityMap] = useState<Map<File, GlobalSecurity>>(new Map());
  const [statusMap, setStatusMap] = useState<Map<File, FileStatus>>(new Map());

  // Process State
  const [processedMetadata, setProcessedMetadata] = useState<DocumentMetadata | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [clearDropzone, setClearDropzone] = useState(false);

  // Helpers
  const firstFileReadyRef = useRef(false);

  const resetState = () => {
    setSelectedFiles([]);
    setSelectedFile(null);
    setMetadataMap(new Map());
    setSecurityMap(new Map());
    setStatusMap(new Map());
    setProcessedMetadata(null);
    setGlobalSecurity({ groupId: null, userIds: [] });
    setClearDropzone(true);
    setTimeout(() => setClearDropzone(false), 100);
  };

  const updateMetadata = (file: File, data: DocumentMetadata) => {
    setMetadataMap(prev => new Map(prev).set(file, JSON.parse(JSON.stringify(data))));
    if (selectedFile === file) setProcessedMetadata(data);
  };

  const updateSecurity = (file: File, groupId: string | null, userIds: string[]) => {
    setSecurityMap(prev => new Map(prev).set(file, { groupId, userIds: [...userIds] }));
  };

  const handleFileSelect = async (files: File[]) => {
    if (!selectedCollection) {
      alert("Select a collection first");
      return;
    }

    setSelectedFiles(files);
    const newStatus = new Map(files.map(f => [f, { isProcessing: false, isReady: false }]));
    setStatusMap(newStatus);
    firstFileReadyRef.current = false;

    // Initialize Maps
    const initMeta = new Map<File, DocumentMetadata>();
    const initSec = new Map<File, GlobalSecurity>();
    files.forEach(f => {
      initMeta.set(f, {});
      initSec.set(f, { groupId: null, userIds: [] });
    });

    if (config.metadata) {
      setIsProcessing(true);
      setError(null);
      try {
        await Promise.all(files.map(async (file) => {
          setStatusMap(prev => new Map(prev).set(file, { isProcessing: true, isReady: false }));
          try {
            const res = await processDocument(selectedCollection, file);
            initMeta.set(file, res.metadata);
            setStatusMap(prev => new Map(prev).set(file, { isProcessing: false, isReady: true }));

            // Auto-select first ready file
            if (!firstFileReadyRef.current) {
              firstFileReadyRef.current = true;
              setSelectedFile(file);
              setProcessedMetadata(res.metadata);
            }
          } catch (e) {
            setStatusMap(prev => new Map(prev).set(file, { isProcessing: false, isReady: false }));
            throw e;
          }
        }));
      } catch (err) {
        setError(err instanceof Error ? err.message : "Metadata processing failed");
      } finally {
        setIsProcessing(false);
      }
    } else {
      // No processing needed
      files.forEach(f => newStatus.set(f, { isProcessing: false, isReady: true }));
      setStatusMap(newStatus);
      if (config.security) {
        setSelectedFile(files[0]);
        setProcessedMetadata({});
      }
    }
    setMetadataMap(initMeta);
    setSecurityMap(initSec);
  };

  const handleUpload = async () => {
    // Validation
    const securityRequired = config.security || (!config.security && !config.metadata);
    if (securityRequired) {
      const missingAuth = selectedFiles.some(f => (securityMap.get(f)?.userIds.length || 0) === 0);
      const missingGlobal = globalSecurity.userIds.length === 0;
      // Use logic: if individual files missing, check global.
      // Simplified: If no users selected anywhere, warn.
      if (missingAuth && missingGlobal && !selectedFile) {
        // Fallback to strict check similar to original
        const allHaveUsers = selectedFiles.every(f => (securityMap.get(f)?.userIds.length || 0) > 0);
        if (!allHaveUsers && globalSecurity.userIds.length === 0) {
          alert("Please select at least one user for all documents");
          return;
        }
      }
    }

    setIsUploading(true);
    setError(null);

    try {
      // Single/Preview Upload
      if (selectedFile && processedMetadata) {
        await uploadDocument(selectedCollection!, selectedFile, processedMetadata);
      } else {
        // Bulk Upload
        const notReady = selectedFiles.some(f => !statusMap.get(f)?.isReady);
        if (notReady) {
          alert("Wait for processing to finish");
          return;
        }

        setUploadProgress({ current: 0, total: selectedFiles.length, fileName: "" });
        await Promise.all(selectedFiles.map(async (file, idx) => {
          setUploadProgress({ current: idx + 1, total: selectedFiles.length, fileName: file.name });
          const meta = metadataMap.get(file) || {};
          await uploadDocument(selectedCollection!, file, meta);
        }));
      }
      resetState();
      onSuccess();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setIsUploading(false);
      setUploadProgress(null);
    }
  };

  return {
    selectedFiles, setSelectedFiles,
    selectedFile, setSelectedFile,
    config, setConfig,
    metadataMap, updateMetadata, setMetadataMap,
    securityMap, updateSecurity, setSecurityMap,
    statusMap,
    processedMetadata, setProcessedMetadata,
    isProcessing, isUploading, uploadProgress,
    error, clearDropzone,
    handleFileSelect, handleUpload,
    globalSecurity, setGlobalSecurity,
    resetState
  };
}

