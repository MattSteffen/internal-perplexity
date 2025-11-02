import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { 
  User, 
  Key, 
  Bell, 
  Shield, 
  Database,
  Settings as SettingsIcon
} from "lucide-react";

export default function SettingsPage() {
  return (
    <div className="container mx-auto p-6 max-w-4xl">
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center gap-3">
          <SettingsIcon className="h-6 w-6" />
          <h1 className="text-2xl font-bold">Settings</h1>
        </div>

        {/* Profile Section */}
        <Card className="p-6">
          <div className="flex items-center gap-3 mb-4">
            <User className="h-5 w-5" />
            <h2 className="text-lg font-semibold">Profile</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium">Name</label>
              <Input placeholder="Your name" className="mt-1" />
            </div>
            <div>
              <label className="text-sm font-medium">Email</label>
              <Input placeholder="your@email.com" className="mt-1" />
            </div>
            <div>
              <label className="text-sm font-medium">Role</label>
              <div className="mt-1">
                <Badge variant="secondary">Researcher</Badge>
              </div>
            </div>
            <div>
              <label className="text-sm font-medium">Last Active</label>
              <div className="mt-1 text-sm text-muted-foreground">
                {new Date().toLocaleDateString()}
              </div>
            </div>
          </div>
        </Card>

        {/* API Keys Section */}
        <Card className="p-6">
          <div className="flex items-center gap-3 mb-4">
            <Key className="h-5 w-5" />
            <h2 className="text-lg font-semibold">API Keys</h2>
          </div>
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium">OpenAI API Key</label>
                <div className="flex gap-2 mt-1">
                  <Input type="password" placeholder="sk-..." className="flex-1" />
                  <Button variant="outline" size="sm">Save</Button>
                </div>
              </div>
              <div>
                <label className="text-sm font-medium">Anthropic API Key</label>
                <div className="flex gap-2 mt-1">
                  <Input type="password" placeholder="sk-ant-..." className="flex-1" />
                  <Button variant="outline" size="sm">Save</Button>
                </div>
              </div>
            </div>
            <div className="text-xs text-muted-foreground">
              API keys are encrypted and stored securely. They are only used for the services you enable.
            </div>
          </div>
        </Card>

        {/* Notifications Section */}
        <Card className="p-6">
          <div className="flex items-center gap-3 mb-4">
            <Bell className="h-5 w-5" />
            <h2 className="text-lg font-semibold">Notifications</h2>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">Email Notifications</div>
                <div className="text-sm text-muted-foreground">
                  Receive updates about your research
                </div>
              </div>
              <Button variant="outline" size="sm">Enable</Button>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">Research Alerts</div>
                <div className="text-sm text-muted-foreground">
                  Get notified about new relevant papers
                </div>
              </div>
              <Button variant="outline" size="sm">Enable</Button>
            </div>
          </div>
        </Card>

        {/* Security Section */}
        <Card className="p-6">
          <div className="flex items-center gap-3 mb-4">
            <Shield className="h-5 w-5" />
            <h2 className="text-lg font-semibold">Security</h2>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">Two-Factor Authentication</div>
                <div className="text-sm text-muted-foreground">
                  Add an extra layer of security to your account
                </div>
              </div>
              <Button variant="outline" size="sm">Setup</Button>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">Session Management</div>
                <div className="text-sm text-muted-foreground">
                  Manage your active sessions
                </div>
              </div>
              <Button variant="outline" size="sm">View</Button>
            </div>
          </div>
        </Card>

        {/* Data Section */}
        <Card className="p-6">
          <div className="flex items-center gap-3 mb-4">
            <Database className="h-5 w-5" />
            <h2 className="text-lg font-semibold">Data & Privacy</h2>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">Export Data</div>
                <div className="text-sm text-muted-foreground">
                  Download your chats and research data
                </div>
              </div>
              <Button variant="outline" size="sm">Export</Button>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">Delete Account</div>
                <div className="text-sm text-muted-foreground">
                  Permanently delete your account and data
                </div>
              </div>
              <Button variant="destructive" size="sm">Delete</Button>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
