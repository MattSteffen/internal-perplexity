"use client"

import { Button } from "@/components/ui/button"
import Link from "next/link"
import { usePathname } from "next/navigation"

export default function SidebarNav() {
  const pathname = usePathname()

  return (
    <div className="grid grid-cols-2 grid-rows-2 gap-2 p-2">
      <Button
        asChild
        variant={pathname.startsWith("/chat") ? "default" : "outline"}
        size="sm"
      >
        <Link href="/chat">Chat</Link>
      </Button>
      <Button
        asChild
        variant={pathname.startsWith("/files") ? "default" : "outline"}
        size="sm"
      >
        <Link href="/files">Files</Link>
      </Button>
      <Button
        asChild
        variant={pathname.startsWith("/create") ? "default" : "outline"}
        size="sm"
      >
        {/* <Link href="/create">Create</Link> */}
        <div>Create</div>
      </Button>
      <Button
        asChild
        variant={pathname.startsWith("/more") ? "default" : "outline"}
        size="sm"
      >
        {/* <Link href="/more">More</Link> */}
        <div>More</div>
      </Button>
    </div>
  )
}