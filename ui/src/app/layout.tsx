import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Meathead Mathematician LLC Internal Knowledge Base Chat Interface",
  description: "Meathead Mathematician LLC Internal Knowledge Base Chat Interface",
};
import type { ReactNode } from "react"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import SidebarNav from "@/components/sidebar-nav"
import { SidebarSlotArea } from "@/components/sidebar-area"

export default function RootLayout({ children, chatsidebar, filesidebar }: { children: ReactNode, chatsidebar: ReactNode, filesidebar: ReactNode }) {
  return (
    <html lang="en">
      <body className="h-screen flex bg-background text-foreground">
        <aside className="w-64 border-r flex flex-col">
          {/* Top quarter */}
          <div className="h-1/4 border-b flex flex-col">
            <div className="flex items-center justify-between p-2">
              <Button asChild variant="ghost" size="sm">
                <Link href="/">MM-Chat</Link>
              </Button>
              <Button variant="ghost" size="icon" className="h-6 w-6">
                ⇤
              </Button>
            </div>

            <SidebarNav />
          </div>

          <SidebarSlotArea chatsidebar={chatsidebar} filesidebar={filesidebar} />

        </aside>
        {/* Slot area gets filled by child layouts */}
        <div className="flex-1 overflow-y-auto p-2">{children}</div>
      </body>
    </html>
  )
}