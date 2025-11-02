import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { ModelPicker } from "@/components/model-picker";
import { ToolPicker } from "@/components/tool-picker";

export default function FileSidebarDefault() {
  return (
    <div className="flex h-full flex-col">
      {/* Model Picker */}
      <div className="p-4 border-b">
        <h3 className="text-sm font-medium mb-3">Model</h3>
        <ModelPicker />
      </div>

      {/* Model Settings */}
      <div className="p-4 border-b">
        <h3 className="text-sm font-medium mb-3">Settings</h3>
        <div className="space-y-4">
          <div>
            <label className="text-xs text-muted-foreground">System Prompt</label>
            <Input 
              placeholder="Leave blank for default"
              className="mt-1"
            />
          </div>
          <div>
            <label className="text-xs text-muted-foreground">Max Tokens</label>
            <Input 
              type="number"
              placeholder="4000"
              className="mt-1"
            />
          </div>
        </div>
      </div>

      {/* Tools Section */}
      <div className="flex-1 p-4">
        <h3 className="text-sm font-medium mb-3">Tools</h3>
        <ToolPicker />
      </div>
    </div>
  );
}
