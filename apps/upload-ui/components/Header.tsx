"use client";

import { Button } from "@/components/ui/button";

interface HeaderProps {
  username: string;
}

/**
 * Page header with title and chat link button.
 */
export function Header({ username }: HeaderProps) {
  return (
    <div className="mb-8 flex justify-between items-center">
      <div>
        <h1 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100">
          Document Upload for {username}
        </h1>
        <p className="mt-2 text-zinc-600 dark:text-zinc-400">
          Select a collection and upload PDF documents
        </p>
      </div>
      <Button variant="outline" asChild>
        <a href={process.env.NEXT_PUBLIC_CHAT_URL || "/"} target="_self">
          Chat
        </a>
      </Button>
    </div>
  );
}

