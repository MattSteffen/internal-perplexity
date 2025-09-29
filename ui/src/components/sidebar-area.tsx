"use client"

import { usePathname } from "next/navigation"

export function SidebarSlotArea({
  chatsidebar,
  filesidebar,
//   createsidebar,
}: {
  chatsidebar: React.ReactNode
  filesidebar: React.ReactNode
//   createsidebar: React.ReactNode
}) {
  const pathname = usePathname()
  if (pathname.startsWith("/chat")) return <>{chatsidebar}</>
  if (pathname.startsWith("/files")) return <>{filesidebar}</>
//   if (pathname.startsWith("/create")) return <>{createsidebar}</>
  return null
}