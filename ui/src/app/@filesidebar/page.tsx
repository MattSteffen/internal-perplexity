"use client"

import { Button } from "@/components/ui/button"

const files = [
  { id: 1, title: "resume.pdf" },
  { id: 2, title: "budget.xlsx" },
  { id: 3, title: "notes.txt" },
  { id: 4, title: "report.docx" },
]

export default function FileSidebar() {
  return (
    <div>
      <h3 className="text-sm font-medium mb-2">Files</h3>
      <ul className="space-y-1">
        {files.map((file) => (
          <li key={file.id}>
            <Button
              variant="ghost"
              size="sm"
              className="w-full justify-start"
            >
              {file.title}
            </Button>
          </li>
        ))}
      </ul>
    </div>
  )
}