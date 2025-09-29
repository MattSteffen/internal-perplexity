"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Chat } from "@/types/chat"

// Combined chat list with pinned status and last updated time
const chats: Chat[] = [
  { id: 1, url: "/uuid_1", title: "Project Alpha", isPinned: true, updatedAt: new Date('2025-01-20T10:00:00') },
  { id: 2, url: "/uuid_2", title: "Data Science Club", isPinned: true, updatedAt: new Date('2025-01-19T15:30:00') },
  { id: 3, url: "/uuid_3", title: "Weekend Plans", isPinned: true, updatedAt: new Date('2025-01-18T09:15:00') },
  { id: 4, url: "/uuid_4", title: "Family Group", isPinned: true, updatedAt: new Date('2025-01-17T14:20:00') },
  { id: 5, url: "/uuid_5", title: "Workout Crew", isPinned: false, updatedAt: new Date('2025-01-20T16:45:00') },
  { id: 6, url: "/uuid_6", title: "AI Research", isPinned: false, updatedAt: new Date('2025-01-19T11:30:00') },
  { id: 7, url: "/uuid_7", title: "Finance Talk", isPinned: false, updatedAt: new Date('2025-01-18T20:00:00') },
  { id: 8, url: "/uuid_8", title: "Random Chat", isPinned: false, updatedAt: new Date('2025-01-17T08:10:00') },
  { id: 9, url: "/uuid_9", title: "Music Lovers", isPinned: false, updatedAt: new Date('2025-01-16T19:25:00') },
]

export default function ChatSidebar() {
  const [query, setQuery] = useState("")

  const filterChats = (chats: Chat[]) =>
    chats.filter((c) =>
      c.title.toLowerCase().includes(query.toLowerCase())
    )

  // Sort chats by pinned status first, then by updatedAt (most recent first)
  const sortedChats = chats.sort((a, b) => {
    if (a.isPinned && !b.isPinned) return -1
    if (!a.isPinned && b.isPinned) return 1
    return b.updatedAt.getTime() - a.updatedAt.getTime()
  })

  // Split into pinned and recent sections
  const pinnedChats = sortedChats.filter(chat => chat.isPinned)
  const recentChats = sortedChats.filter(chat => !chat.isPinned)

  return (
    <div className="space-y-6">
      {/* Search bar */}
      <div>
        <Input
          placeholder="Search chats..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>

      {/* Pinned Chats */}
      {pinnedChats.length > 0 && (
        <div>
          <h3 className="text-sm font-medium mb-2">Pinned</h3>
          <ul className="space-y-1">
            {filterChats(pinnedChats).map((chat) => (
              <li key={chat.id}>
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start group"
                >
                  {chat.title}
                  {/* TODO: Add link to chat */}
                </Button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Separator */}
      {pinnedChats.length > 0 && recentChats.length > 0 && (
        <div className="border-t border-gray-200 dark:border-gray-700 my-4" />
      )}

      {/* Recent Chats */}
      {recentChats.length > 0 && (
        <div>
          <h3 className="text-sm font-medium mb-2">Recent</h3>
          <ul className="space-y-1">
            {filterChats(recentChats).map((chat) => (
              <li key={chat.id}>
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start group"
                >
                  {chat.title}
                  {/* TODO: Add link to chat */}
                </Button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}