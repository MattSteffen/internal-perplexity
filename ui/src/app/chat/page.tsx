"use client";

import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Paperclip, Search, Send, Settings } from "lucide-react";

export default function ChatPage() {
  return (
    <div className="h-screen flex flex-col relative">
      {/* Header floats at top, above scroll */}
      <header className="fixed top-0 left-0 right-0 z-50 p-4 border-b bg-white shadow flex justify-between items-center p-2 border-b flex-shrink-0">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline">Model Picker</Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent>
            <DropdownMenuItem>GPT-4</DropdownMenuItem>
            <DropdownMenuItem>GPT-3.5</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>

        <Button variant="ghost" size="icon">
          <Settings className="h-5 w-5" />
        </Button>
      </header>

      {/* Scrollable messages area */}
      <main className="flex-1 overflow-y-auto p-4 pt-20 pb-32">
        {Array.from({ length: 100 }).map((_, i) => (
          <p key={i}>Message {i + 1}</p>
        ))}
      </main>

      {/* Footer floats at bottom, above scroll */}
      <footer className="fixed left-0 right-0 z-50 px-4 border-t bg-background p-2 space-y-2 shadow-lg flex-shrink-0">
        <Textarea placeholder="Type your message..." />
        <div className="flex items-center justify-between">
          <div className="flex gap-2">
            <Button variant="ghost" size="icon">
              <Paperclip className="h-5 w-5" />
            </Button>
            <Button variant="ghost" size="icon">
              <Search className="h-5 w-5" />
            </Button>
          </div>
          <Button size="icon">
            <Send className="h-5 w-5" />
          </Button>
        </div>
      </footer>
    </div>
  );
}

// export default function ChatPage() {
//   return (
//     <div className="flex flex-col h-dvh">
//       {/* Header */}

//       {/* Messages */}
//       <ScrollArea className="flex-1 min-h-0 p-4">
//         <div className="mb-2 max-w-[60%] rounded-lg bg-muted p-3">
//           Hello! This is a user bubble.
//         </div>
//         <div className="mb-2 max-w-[60%] rounded-lg bg-primary p-3 text-primary-foreground">
//           Hi there, Im the AI.
//         </div>
//       </ScrollArea>

//       {/* Footer */}
//     </div>
//   );
// }