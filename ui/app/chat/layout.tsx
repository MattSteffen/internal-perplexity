import { ReactNode } from "react";

export default function ChatLayout({
  children,
  chatsidebar,
  filesidebar,
}: {
  children: ReactNode;
  chatsidebar: ReactNode;
  filesidebar: ReactNode;
}) {
  return (
    <div className="flex h-screen bg-background">
      {/* Left Sidebar - Chat Navigation */}
      <div className="w-80 border-r bg-muted/40">
        {chatsidebar}
      </div>
      
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {children}
      </div>
      
      {/* Right Sidebar - Files/Model Settings */}
      <div className="w-80 border-l bg-muted/40">
        {filesidebar}
      </div>
    </div>
  );
}
