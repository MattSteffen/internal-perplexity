"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { 
  Search, 
  Database, 
  Code, 
  Search as SearchIcon,
  FileText,
  Globe,
  ToggleLeft,
  ToggleRight
} from "lucide-react";

interface Tool {
  id: string;
  name: string;
  description: string;
  provider: string;
  category: string;
  enabled: boolean;
  icon: React.ReactNode;
}

const mockTools: Tool[] = [
  {
    id: "1",
    name: "Milvus Query",
    description: "Search and query the Milvus vector database",
    provider: "Milvus MCP",
    category: "RAG",
    enabled: true,
    icon: <Database className="h-4 w-4" />,
  },
  {
    id: "2",
    name: "Milvus Search",
    description: "Advanced search capabilities in Milvus",
    provider: "Milvus MCP",
    category: "RAG",
    enabled: true,
    icon: <SearchIcon className="h-4 w-4" />,
  },
  {
    id: "3",
    name: "Code Execution",
    description: "Execute Python code in a sandboxed environment",
    provider: "Code Tools",
    category: "Code",
    enabled: false,
    icon: <Code className="h-4 w-4" />,
  },
  {
    id: "4",
    name: "Web Search",
    description: "Search the web for real-time information",
    provider: "Web Tools",
    category: "Search",
    enabled: false,
    icon: <Globe className="h-4 w-4" />,
  },
  {
    id: "5",
    name: "Document Analysis",
    description: "Analyze and extract information from documents",
    provider: "Document Tools",
    category: "Analysis",
    enabled: false,
    icon: <FileText className="h-4 w-4" />,
  },
];

const categories = ["All", "RAG", "Code", "Search", "Analysis"];

export function ToolPicker() {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("All");
  const [tools, setTools] = useState<Tool[]>(mockTools);

  const filteredTools = tools.filter(tool => {
    const matchesSearch = tool.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         tool.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === "All" || tool.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const toggleTool = (toolId: string) => {
    setTools(prev => prev.map(tool => 
      tool.id === toolId ? { ...tool, enabled: !tool.enabled } : tool
    ));
  };

  return (
    <div className="space-y-4">
      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          placeholder="Search tools..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="pl-10"
        />
      </div>

      {/* Category Filter */}
      <div className="flex gap-2 flex-wrap">
        {categories.map((category) => (
          <Button
            key={category}
            variant={selectedCategory === category ? "default" : "outline"}
            size="sm"
            onClick={() => setSelectedCategory(category)}
          >
            {category}
          </Button>
        ))}
      </div>

      {/* Tools List */}
      <ScrollArea className="h-64">
        <div className="space-y-2">
          {filteredTools.map((tool) => (
            <ToolCard 
              key={tool.id} 
              tool={tool} 
              onToggle={() => toggleTool(tool.id)}
            />
          ))}
        </div>
      </ScrollArea>

      {/* RAG Tools Note */}
      <div className="text-xs text-muted-foreground bg-muted/50 p-2 rounded">
        💡 Use @ in the input bar to force toggle RAG tools
      </div>
    </div>
  );
}

function ToolCard({ 
  tool, 
  onToggle 
}: { 
  tool: Tool; 
  onToggle: () => void;
}) {
  return (
    <Card className="p-3">
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-3 flex-1">
          <div className="text-muted-foreground mt-0.5">
            {tool.icon}
          </div>
          <div className="flex-1 min-w-0">
            <div className="font-medium text-sm">{tool.name}</div>
            <div className="text-xs text-muted-foreground mt-1">
              {tool.description}
            </div>
            <div className="flex items-center gap-2 mt-2">
              <Badge variant="outline" className="text-xs">
                {tool.provider}
              </Badge>
              <Badge variant="secondary" className="text-xs">
                {tool.category}
              </Badge>
            </div>
          </div>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={onToggle}
          className="ml-2"
        >
          {tool.enabled ? (
            <ToggleRight className="h-4 w-4 text-primary" />
          ) : (
            <ToggleLeft className="h-4 w-4 text-muted-foreground" />
          )}
        </Button>
      </div>
    </Card>
  );
}
