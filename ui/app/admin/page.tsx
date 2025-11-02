import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  Users, 
  Database, 
  Settings, 
  Shield,
  Activity,
  TrendingUp
} from "lucide-react";

export default function AdminPage() {
  return (
    <div className="container mx-auto p-6 max-w-6xl">
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Admin Dashboard</h1>
            <p className="text-muted-foreground">Manage your research platform</p>
          </div>
          <Badge variant="outline" className="text-green-600 border-green-600">
            <Activity className="h-3 w-3 mr-1" />
            System Healthy
          </Badge>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="p-4">
            <div className="flex items-center gap-3">
              <Users className="h-8 w-8 text-blue-500" />
              <div>
                <div className="text-2xl font-bold">1,234</div>
                <div className="text-sm text-muted-foreground">Active Users</div>
              </div>
            </div>
          </Card>
          <Card className="p-4">
            <div className="flex items-center gap-3">
              <Database className="h-8 w-8 text-green-500" />
              <div>
                <div className="text-2xl font-bold">45.2K</div>
                <div className="text-sm text-muted-foreground">Documents</div>
              </div>
            </div>
          </Card>
          <Card className="p-4">
            <div className="flex items-center gap-3">
              <TrendingUp className="h-8 w-8 text-purple-500" />
              <div>
                <div className="text-2xl font-bold">12.5K</div>
                <div className="text-sm text-muted-foreground">Queries Today</div>
              </div>
            </div>
          </Card>
          <Card className="p-4">
            <div className="flex items-center gap-3">
              <Shield className="h-8 w-8 text-orange-500" />
              <div>
                <div className="text-2xl font-bold">99.9%</div>
                <div className="text-sm text-muted-foreground">Uptime</div>
              </div>
            </div>
          </Card>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* User Management */}
          <Card className="p-6">
            <div className="flex items-center gap-3 mb-4">
              <Users className="h-5 w-5" />
              <h2 className="text-lg font-semibold">User Management</h2>
            </div>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium">Total Users</div>
                  <div className="text-sm text-muted-foreground">1,234 active users</div>
                </div>
                <Button variant="outline" size="sm">Manage</Button>
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium">New Users (24h)</div>
                  <div className="text-sm text-muted-foreground">+23 new registrations</div>
                </div>
                <Button variant="outline" size="sm">View</Button>
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium">Permissions</div>
                  <div className="text-sm text-muted-foreground">Manage user roles and access</div>
                </div>
                <Button variant="outline" size="sm">Configure</Button>
              </div>
            </div>
          </Card>

          {/* System Status */}
          <Card className="p-6">
            <div className="flex items-center gap-3 mb-4">
              <Database className="h-5 w-5" />
              <h2 className="text-lg font-semibold">System Status</h2>
            </div>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-green-500" />
                  <span className="text-sm">Milvus Database</span>
                </div>
                <Badge variant="outline" className="text-green-600">Healthy</Badge>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-green-500" />
                  <span className="text-sm">Vector Search</span>
                </div>
                <Badge variant="outline" className="text-green-600">Healthy</Badge>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-yellow-500" />
                  <span className="text-sm">LLM Services</span>
                </div>
                <Badge variant="outline" className="text-yellow-600">Degraded</Badge>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-green-500" />
                  <span className="text-sm">API Gateway</span>
                </div>
                <Badge variant="outline" className="text-green-600">Healthy</Badge>
              </div>
            </div>
          </Card>

          {/* Content Management */}
          <Card className="p-6">
            <div className="flex items-center gap-3 mb-4">
              <Database className="h-5 w-5" />
              <h2 className="text-lg font-semibold">Content Management</h2>
            </div>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium">Document Collections</div>
                  <div className="text-sm text-muted-foreground">Manage research documents</div>
                </div>
                <Button variant="outline" size="sm">Manage</Button>
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium">Vector Indexes</div>
                  <div className="text-sm text-muted-foreground">Optimize search performance</div>
                </div>
                <Button variant="outline" size="sm">Optimize</Button>
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium">Data Sources</div>
                  <div className="text-sm text-muted-foreground">Configure external data feeds</div>
                </div>
                <Button variant="outline" size="sm">Configure</Button>
              </div>
            </div>
          </Card>

          {/* System Configuration */}
          <Card className="p-6">
            <div className="flex items-center gap-3 mb-4">
              <Settings className="h-5 w-5" />
              <h2 className="text-lg font-semibold">Configuration</h2>
            </div>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium">Model Settings</div>
                  <div className="text-sm text-muted-foreground">Configure AI models and parameters</div>
                </div>
                <Button variant="outline" size="sm">Configure</Button>
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium">Security Policies</div>
                  <div className="text-sm text-muted-foreground">Manage access controls and permissions</div>
                </div>
                <Button variant="outline" size="sm">Manage</Button>
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium">Backup & Recovery</div>
                  <div className="text-sm text-muted-foreground">Data protection and disaster recovery</div>
                </div>
                <Button variant="outline" size="sm">Configure</Button>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
