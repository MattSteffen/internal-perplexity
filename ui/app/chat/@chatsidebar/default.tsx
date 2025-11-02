import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card } from "@/components/ui/card";
import { Plus, Search, Pin, MoreHorizontal } from "lucide-react";

export default function ChatSidebarDefault() {
  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="p-4 border-b">
        <Button className="w-full justify-start gap-2">
          <Plus className="h-4 w-4" />
          New Chat
        </Button>
      </div>

      {/* Search */}
      <div className="p-4 border-b">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search chats..."
            className="pl-10"
          />
        </div>
      </div>

      {/* Pinned Chats */}
      <div className="p-4">
        <h3 className="text-sm font-medium text-muted-foreground mb-2">Pinned</h3>
        <div className="space-y-2">
          <Card className="p-3 cursor-pointer hover:bg-muted/50">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Pin className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm">Research on AI Safety</span>
              </div>
              <Button variant="ghost" size="sm">
                <MoreHorizontal className="h-4 w-4" />
              </Button>
            </div>
          </Card>
        </div>
      </div>

      {/* Recent Chats */}
      <div className="flex-1 p-4">
        <h3 className="text-sm font-medium text-muted-foreground mb-2">Recent</h3>
        <ScrollArea className="h-full">
          <div className="space-y-2">
            {Array.from({ length: 10 }).map((_, i) => (
              <Card key={i} className="p-3 cursor-pointer hover:bg-muted/50">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Chat {i + 1}</span>
                  <Button variant="ghost" size="sm">
                    <MoreHorizontal className="h-4 w-4" />
                  </Button>
                </div>
              </Card>
            ))}
          </div>
        </ScrollArea>
      </div>

      {/* Navigation */}
      <div className="p-4 border-t">
        <div className="space-y-2">
          <Button variant="ghost" className="w-full justify-start">
            Profile
          </Button>
          <Button variant="ghost" className="w-full justify-start" asChild>
            <a href="/settings">Settings</a>
          </Button>
          <Button variant="ghost" className="w-full justify-start" asChild>
            <a href="/admin">Admin</a>
          </Button>
        </div>
      </div>
    </div>
  );
}
