import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"
import type { Document, DocumentMetadata } from "@/lib/types"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Normalize metadata author to a single string (handles string or array from API).
 */
function authorToString(author: DocumentMetadata["author"]): string {
  if (author == null) return ""
  if (Array.isArray(author)) return author.filter(Boolean).join(", ")
  return String(author)
}

/**
 * Get display title and authors for a search result Document.
 * Title: metadata.title, else first line of markdown, else source, else "Untitled".
 * Authors: metadata.author normalized to string.
 */
export function getDocumentDisplayInfo(doc: Document): {
  displayTitle: string
  displayAuthors: string
} {
  const firstLineOfMarkdown =
    doc.markdown?.trim().split(/\n/)[0]?.trim() ?? ""
  const displayTitle =
    doc.metadata?.title ?? firstLineOfMarkdown ?? doc.source ?? "Untitled"
  const displayAuthors = authorToString(doc.metadata?.author)
  return { displayTitle, displayAuthors }
}
