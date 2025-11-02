"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { ScrollArea } from "@/components/ui/scroll-area";
import { 
  Search, 
  Filter, 
  Star, 
  Key, 
  Brain, 
  Eye, 
  Zap, 
  Image as ImageIcon,
  ChevronDown
} from "lucide-react";

interface Model {
  id: string;
  name: string;
  provider: string;
  type: "internal" | "external";
  capabilities: string[];
  hasApiKey: boolean;
  isPinned: boolean;
  icon: string;
}

const mockModels: Model[] = [
  {
    id: "1",
    name: "GPT-4o",
    provider: "OpenAI",
    type: "external",
    capabilities: ["reasoning", "vision", "search"],
    hasApiKey: true,
    isPinned: true,
    icon: "🤖",
  },
  {
    id: "2",
    name: "Claude 3.5 Sonnet",
    provider: "Anthropic",
    type: "external",
    capabilities: ["reasoning", "vision", "fast"],
    hasApiKey: false,
    isPinned: true,
    icon: "🧠",
  },
  {
    id: "3",
    name: "Internal Research Model",
    provider: "Internal",
    type: "internal",
    capabilities: ["reasoning", "search"],
    hasApiKey: false,
    isPinned: false,
    icon: "🏠",
  },
  {
    id: "4",
    name: "Gemini Pro",
    provider: "Google",
    type: "external",
    capabilities: ["reasoning", "vision", "generate-images"],
    hasApiKey: true,
    isPinned: false,
    icon: "💎",
  },
];

const capabilityIcons = {
  reasoning: Brain,
  vision: Eye,
  search: Zap,
  "generate-images": ImageIcon,
  fast: Zap,
};

export function ModelPicker() {
  const [searchTerm, setSearchTerm] = useState("");
  const [showAll, setShowAll] = useState(false);
  const [selectedModel, setSelectedModel] = useState<Model | null>(mockModels[0]);

  const filteredModels = mockModels.filter(model => {
    if (!showAll && !model.isPinned) return false;
    return model.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
           model.provider.toLowerCase().includes(searchTerm.toLowerCase());
  });

  const pinnedModels = mockModels.filter(model => model.isPinned);
  const otherModels = mockModels.filter(model => !model.isPinned);

  return (
    <div className="space-y-4">
      {/* Current Model */}
      {selectedModel && (
        <Card className="p-3 bg-primary/5 border-primary/20">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Avatar className="h-8 w-8">
                <AvatarFallback>{selectedModel.icon}</AvatarFallback>
              </Avatar>
              <div>
                <div className="font-medium">{selectedModel.name}</div>
                <div className="text-sm text-muted-foreground">{selectedModel.provider}</div>
              </div>
            </div>
            <div className="flex items-center gap-1">
              {selectedModel.hasApiKey ? (
                <Key className="h-4 w-4 text-green-500" />
              ) : (
                <Key className="h-4 w-4 text-red-500" />
              )}
              <Button variant="ghost" size="sm">
                <ChevronDown className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </Card>
      )}

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          placeholder="Search models..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="pl-10"
        />
      </div>

      {/* Filter Button */}
      <Button variant="outline" size="sm" className="w-full">
        <Filter className="h-4 w-4 mr-2" />
        Filter
      </Button>

      {/* Models List */}
      <ScrollArea className="h-96">
        <div className="space-y-2">
          {!showAll && pinnedModels.length > 0 && (
            <>
              <h4 className="text-sm font-medium text-muted-foreground">Pinned</h4>
              {pinnedModels.map((model) => (
                <ModelCard 
                  key={model.id} 
                  model={model} 
                  isSelected={selectedModel?.id === model.id}
                  onSelect={() => setSelectedModel(model)}
                />
              ))}
            </>
          )}

          {showAll && (
            <>
              <h4 className="text-sm font-medium text-muted-foreground">All Models</h4>
              {filteredModels.map((model) => (
                <ModelCard 
                  key={model.id} 
                  model={model} 
                  isSelected={selectedModel?.id === model.id}
                  onSelect={() => setSelectedModel(model)}
                />
              ))}
            </>
          )}
        </div>
      </ScrollArea>

      {/* Show All Button */}
      <Button 
        variant="outline" 
        className="w-full"
        onClick={() => setShowAll(!showAll)}
      >
        {showAll ? "Show Pinned Only" : "Show All Models"}
      </Button>
    </div>
  );
}

function ModelCard({ 
  model, 
  isSelected, 
  onSelect 
}: { 
  model: Model; 
  isSelected: boolean; 
  onSelect: () => void;
}) {
  return (
    <Card 
      className={`p-3 cursor-pointer transition-colors ${
        isSelected 
          ? "bg-primary/10 border-primary/30" 
          : "hover:bg-muted/50"
      }`}
      onClick={onSelect}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Avatar className="h-8 w-8">
            <AvatarFallback>{model.icon}</AvatarFallback>
          </Avatar>
          <div>
            <div className="font-medium">{model.name}</div>
            <div className="text-sm text-muted-foreground">{model.provider}</div>
            <div className="flex items-center gap-1 mt-1">
              {model.capabilities.map((capability) => {
                const Icon = capabilityIcons[capability as keyof typeof capabilityIcons];
                return Icon ? (
                  <Icon key={capability} className="h-3 w-3 text-muted-foreground" />
                ) : null;
              })}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {model.isPinned && <Star className="h-4 w-4 text-yellow-500 fill-current" />}
          {model.hasApiKey ? (
            <Key className="h-4 w-4 text-green-500" />
          ) : (
            <Key className="h-4 w-4 text-red-500" />
          )}
          <Badge variant={model.type === "internal" ? "default" : "secondary"}>
            {model.type}
          </Badge>
        </div>
      </div>
    </Card>
  );
}
