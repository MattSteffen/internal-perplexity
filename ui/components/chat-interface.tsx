"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card } from "@/components/ui/card";
import { 
  Send, 
  Copy, 
  ThumbsUp, 
  ThumbsDown, 
  RotateCcw, 
  Edit, 
  Trash2, 
  Share,
  MoreHorizontal
} from "lucide-react";

interface Message {
  id: string;
  type: "user" | "assistant" | "tool";
  content: string;
  timestamp: Date;
  sources?: Source[];
}

interface Source {
  id: string;
  title: string;
  type: "website" | "internal" | "irad" | "arxiv";
  url?: string;
}

const mockMessages: Message[] = [
  {
    id: "1",
    type: "user",
    content: "What are the latest developments in AI safety research?",
    timestamp: new Date(),
  },
  {
    id: "2",
    type: "assistant",
    content: "Based on recent research, here are the key developments in AI safety:\n\n1. **Constitutional AI**: Researchers are developing methods to train AI systems to follow human values and principles through constitutional training.\n\n2. **Interpretability**: New techniques for understanding how large language models make decisions, including mechanistic interpretability and activation patching.\n\n3. **Alignment Research**: Focus on ensuring AI systems remain aligned with human intentions as they become more capable.\n\n4. **Robustness Testing**: Enhanced methods for testing AI systems' behavior in edge cases and adversarial scenarios.",
    timestamp: new Date(),
    sources: [
      { id: "1", title: "Constitutional AI: Harmlessness from AI Feedback", type: "arxiv" },
      { id: "2", title: "Mechanistic Interpretability of Transformers", type: "arxiv" },
      { id: "3", title: "AI Safety Research Overview", type: "internal" },
      { id: "4", title: "+2 more sources", type: "internal" },
    ],
  },
];

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>(mockMessages);
  const [input, setInput] = useState("");

  const handleSend = () => {
    if (!input.trim()) return;

    const newMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, newMessage]);
    setInput("");

    // Simulate assistant response
    setTimeout(() => {
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "assistant",
        content: "I understand your question. Let me search for relevant information...",
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, assistantMessage]);
    }, 1000);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages Area */}
      <ScrollArea className="flex-1 p-4">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.map((message) => (
            <MessageComponent key={message.id} message={message} />
          ))}
        </div>
      </ScrollArea>

      {/* Input Area */}
      <div className="border-t bg-background p-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex gap-2">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask anything... Use / for prompts, @ for RAG collections"
              className="min-h-[60px] max-h-[120px] resize-none"
            />
            <Button 
              onClick={handleSend}
              disabled={!input.trim()}
              size="sm"
              className="self-end"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

function MessageComponent({ message }: { message: Message }) {
  if (message.type === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] bg-primary text-primary-foreground rounded-lg p-4">
          <div className="whitespace-pre-wrap">{message.content}</div>
          <div className="flex items-center gap-2 mt-3">
            <Button variant="ghost" size="sm" className="h-8 w-8 p-0 text-primary-foreground hover:bg-primary-foreground/20">
              <Copy className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="sm" className="h-8 w-8 p-0 text-primary-foreground hover:bg-primary-foreground/20">
              <Edit className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="sm" className="h-8 w-8 p-0 text-primary-foreground hover:bg-primary-foreground/20">
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    );
  }

  if (message.type === "assistant") {
    return (
      <div className="flex justify-start">
        <div className="max-w-[80%] space-y-4">
          <Card className="p-4">
            <div className="whitespace-pre-wrap">{message.content}</div>
            
            {/* Sources */}
            {message.sources && (
              <div className="mt-4 pt-4 border-t">
                <div className="grid grid-cols-2 gap-2">
                  {message.sources.map((source) => (
                    <Card key={source.id} className="p-2 cursor-pointer hover:bg-muted/50">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-blue-500" />
                        <span className="text-sm">{source.title}</span>
                      </div>
                    </Card>
                  ))}
                </div>
              </div>
            )}
          </Card>
          
          {/* Message Actions */}
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" className="h-8">
              <Copy className="h-4 w-4 mr-2" />
              Copy
            </Button>
            <Button variant="ghost" size="sm" className="h-8">
              <ThumbsUp className="h-4 w-4 mr-2" />
              Good
            </Button>
            <Button variant="ghost" size="sm" className="h-8">
              <ThumbsDown className="h-4 w-4 mr-2" />
              Bad
            </Button>
            <Button variant="ghost" size="sm" className="h-8">
              <RotateCcw className="h-4 w-4 mr-2" />
              Regenerate
            </Button>
            <Button variant="ghost" size="sm" className="h-8">
              <Edit className="h-4 w-4 mr-2" />
              Edit
            </Button>
            <Button variant="ghost" size="sm" className="h-8">
              <Share className="h-4 w-4 mr-2" />
              Share
            </Button>
            <Button variant="ghost" size="sm" className="h-8">
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return null;
}
