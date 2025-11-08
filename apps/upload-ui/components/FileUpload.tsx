"use client";

import { useState, useEffect } from "react";
import {
  Dropzone,
  DropzoneContent,
  DropzoneEmptyState,
} from "@/components/ui/shadcn-io/dropzone";

interface FileUploadProps {
  onFileSelect: (files: File[]) => void;
  disabled?: boolean;
  clearFiles?: boolean;
}

export function FileUpload({ onFileSelect, disabled, clearFiles }: FileUploadProps) {
  const [files, setFiles] = useState<File[] | undefined>();

  // Clear files when clearFiles prop changes to true
  useEffect(() => {
    if (clearFiles) {
      setFiles(undefined);
    }
  }, [clearFiles]);

  const handleDrop = (acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setFiles(acceptedFiles);
      onFileSelect(acceptedFiles);
    }
  };

  const handleError = (error: Error) => {
    // Show alert for disabled state error
    if (error.message.includes('select a collection')) {
      alert(error.message);
    }
  };

  return (
    <Dropzone
      accept={{ "application/pdf": [] }}
      onDrop={handleDrop}
      onError={handleError}
      disabled={disabled}
      src={files}
      maxFiles={1} // TODO: Change to 100 when we have multiple file support
    >
      <DropzoneEmptyState>
        <div className="pointer-events-none flex flex-col items-center justify-center">
          <div className="flex size-8 items-center justify-center rounded-md bg-muted text-muted-foreground">
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
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
          </div>
          <p className="my-2 w-full truncate text-wrap font-medium text-sm">
            {disabled ? "Select a collection first" : "Upload files"}
          </p>
          <p className="w-full truncate text-wrap text-muted-foreground text-xs">
            Drag and drop or click to upload
          </p>
          <p className="text-wrap text-muted-foreground text-xs">
            PDF files only
          </p>
        </div>
      </DropzoneEmptyState>
      <DropzoneContent />
    </Dropzone>
  );
}

