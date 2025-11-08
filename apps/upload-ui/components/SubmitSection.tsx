"use client";

interface SubmitSectionProps {
  files: File[];
  onSubmit: () => void;
  onCancel: () => void;
  isUploading?: boolean;
  disabled?: boolean;
}

export function SubmitSection({
  files,
  onSubmit,
  onCancel,
  isUploading = false,
  disabled = false,
}: SubmitSectionProps) {
  if (files.length === 0) {
    return null;
  }

  return (
    <div className="mt-6 rounded-lg border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
      <h3 className="mb-4 text-lg font-semibold text-zinc-900 dark:text-zinc-100">
        Ready to Submit
      </h3>
      <div className="mb-4">
        <p className="mb-2 text-sm font-medium text-zinc-700 dark:text-zinc-300">
          Files to be uploaded ({files.length}):
        </p>
        <ul className="max-h-48 space-y-1 overflow-y-auto rounded-lg border border-zinc-200 bg-zinc-50 p-3 dark:border-zinc-700 dark:bg-zinc-800">
          {files.map((file, index) => (
            <li
              key={index}
              className="flex items-center gap-2 text-sm text-zinc-700 dark:text-zinc-300"
            >
              <svg
                className="h-4 w-4 text-zinc-500 dark:text-zinc-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              <span className="truncate">{file.name}</span>
            </li>
          ))}
        </ul>
      </div>
      <div className="flex justify-end gap-3">
        <button
          onClick={onCancel}
          disabled={isUploading}
          className="rounded-lg border border-zinc-300 bg-white px-4 py-2 text-sm font-medium text-zinc-700 transition-colors hover:bg-zinc-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
        >
          Cancel
        </button>
        <button
          onClick={onSubmit}
          disabled={isUploading || disabled}
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-blue-500 dark:hover:bg-blue-600"
        >
          {isUploading ? "Uploading..." : `Submit ${files.length} file${files.length !== 1 ? "s" : ""}`}
        </button>
      </div>
    </div>
  );
}

