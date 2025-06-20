# outline of user interface (UI + Backend)

Requirements:

- Must have granular permissions for everything
- Must have a database for chats and other things, enable saving sensitive data like api keys
- Must have good file system navigation
- Tooltips for all buttons beneath messages

## UI

Framework: Next.js, React, Tailwind CSS, Shadcn UI
Deployments: Docker compose and k8s helm charts

### Pages

- **Layouts**

  - Nav bar on left side
    - Max 4 options: New Chat, Files, Workspace, etc
    - Search bar for chats
    - Pinned Chats
    - Sorted Chats
      - On hover: show options to delete, pin, rename, share
    - **On load**
      - To make things fast, we have 2 api calls to load the chats.
        - 1st call: Loads the chat names and metadata (pinned, date, etc)
        - 2nd call: Loads the messages
        - On click of an individual chat, recall load and refresh the page
    - Profile Button on bottom left (sticky)
      - Profile page
      - Settings page
      - Logout button

- Main Chat Page (Home page or initial page on load)

  - Main Layout + Chat Interface

  - Chat Interface
    - Messages are displayed like any other chat app, list of user and assistant messages, tool calls included.
      - Types: User, Assistant, Tool Call
    - Input bar at the bottom
    - On right side, a model sidebar
    - _Later implement Artifacts_

- Settings Page

- _File explorer_

  - _Can I use MINIO instead?_
  - Each folder or file when clicked once opens metadata in a modal
    - In that modal it has a button to pin to new chat
  - Can right click and pin to new chat

- Admin Page

### Components

- Model Picker

  - Pinned chats in list
  - Button at bottom for `show all` and on click shows like T3.chat have squares as model cards.
    - 2 Sections, Pinned or Favorites, Others
  - Next to button, a search bar, and a filter button.
  - Each entry (either list entry or card) has a model name, icon for provider, then supported types (vision, search, reasoning, generate images, fast)
  - Each entry must clearly indicate internal or external model
  - Each model that requires an api key will also have an indicator that is either red or green, if they click on a model with a red api key they will be prompted to enter their api key. that will then be saved. It will have a gray key if a key is provided by the organization.

- Assistant Message:

  - Text rendered with markdown or normal
  - Sources if present
    - Beneath text a list of 3-4 cards with icon and title, final card contains `+n` for n more sources
    - _inline sources_
  - On bottom of each message:
    - Copy button
    - Thumbs up/down
    - Regenerate (maybe with a new model)
    - Edit / Delete
    - Share
    - _Branch_

- User Message:

  - Text rendered normal or with triple tick parsing
  - On bottom of each message:
    - Copy button
    - Edit / Delete

- Source:

  - Contains Icon for location (website, internal doc, irad, etc) and a title
  - On click: Show source metadata and link to original
    - Metadata: Title, source, date, author, etc
    - Also show: button to pin, start new chat with, retry answer without source

- Input Bar:

  - Rounded rectangle with 2 rows:
  - User message input
    - Use `/` for prompts or tools, use `@` for RAG collections
    - When typing `/` or `@` show a list of options of about 3 or 4, and by typing further, it performs search and filters the list
  - Model picker, tool picker, upload new items/docs/images etc.

- Right (model) sidebar:

  - Model specific options:
  - System Prompt (leave blank for default)
  - Max tokens, etc

- Tool Picker:
  - Heirarchy of tools, a provider can have multiple tools
    - For example: Milvus MCP has query and search tools
    - Or type of tool like RAG, search, code, etc
  - Toggle for each tool
  - Each tool has name and on hover shows description as tooltip
  - By using `@` in the input bar, we can force toggle RAG tools

### **On Load API Calls**

1. User metadata
2. Chat names
3. Model data
4. All user metadata including files and such
5. Chat messages for all chats

## Database

Tables:

- Users
- Chats
- Messages
- Tools
- Models

```go
// https://github.com/open-webui/open-webui/tree/main/backend/open_webui/models
// Chat represents the 'chat' table in the database.
type Chat struct {
	ID        string                 `json:"id"`
	UserID    string                 `json:"user_id"`
	Title     string                 `json:"title"`
	Chat      map[string]interface{} `json:"chat"`
	CreatedAt int64                  `json:"created_at"`
	UpdatedAt int64                  `json:"updated_at"`
	ShareID   *string                `json:"share_id,omitempty"`
	Archived  bool                   `json:"archived"`
	Pinned    *bool                  `json:"pinned,omitempty"`
	Meta      map[string]interface{} `json:"meta"`
	FolderID  *string                `json:"folder_id,omitempty"`
}

// ChatForm represents the basic form for creating a new chat.
type ChatForm struct {
	Chat map[string]interface{} `json:"chat"`
}

// ChatImportForm represents the form for importing a chat.
type ChatImportForm struct {
	ChatForm
	Meta     map[string]interface{} `json:"meta,omitempty"`
	Pinned   *bool                  `json:"pinned,omitempty"`
	FolderID *string                `json:"folder_id,omitempty"`
}

// ChatTitleMessagesForm represents the form with title and a list of messages.
type ChatTitleMessagesForm struct {
	Title    string                   `json:"title"`
	Messages []map[string]interface{} `json:"messages"`
}

// ChatTitleForm represents a form for updating a chat's title.
type ChatTitleForm struct {
	Title string `json:"title"`
}

// ChatResponse represents the full chat data returned by the API.
type ChatResponse struct {
	ID        string                 `json:"id"`
	UserID    string                 `json:"user_id"`
	Title     string                 `json:"title"`
	Chat      map[string]interface{} `json:"chat"`
	UpdatedAt int64                  `json:"updated_at"`
	CreatedAt int64                  `json:"created_at"`
	ShareID   *string                `json:"share_id,omitempty"`
	Archived  bool                   `json:"archived"`
	Pinned    *bool                  `json:"pinned,omitempty"`
	Meta      map[string]interface{} `json:"meta"`
	FolderID  *string                `json:"folder_id,omitempty"`
}

// ChatTitleIdResponse represents a summarized chat entry.
type ChatTitleIdResponse struct {
	ID        string `json:"id"`
	Title     string `json:"title"`
	UpdatedAt int64  `json:"updated_at"`
	CreatedAt int64  `json:"created_at"`
}

// Knowledge represents the 'knowledge' table in the database.
type Knowledge struct {
	ID            string                 `json:"id"`
	UserID        string                 `json:"user_id"`
	Name          string                 `json:"name"`
	Description   string                 `json:"description"`
	Data          map[string]interface{} `json:"data,omitempty"`
	Meta          map[string]interface{} `json:"meta,omitempty"`
	AccessControl map[string]interface{} `json:"access_control,omitempty"`
	CreatedAt     int64                  `json:"created_at"`
	UpdatedAt     int64                  `json:"updated_at"`
}

// Memory represents the 'memory' table in the database.
type Memory struct {
	ID        string `json:"id"`
	UserID    string `json:"user_id"`
	Content   string `json:"content"`
	UpdatedAt int64  `json:"updated_at"`
	CreatedAt int64  `json:"created_at"`
}

// MessageReaction represents the 'message_reaction' table in the database.
type MessageReaction struct {
	ID        string `json:"id"`
	UserID    string `json:"user_id"`
	MessageID string `json:"message_id"`
	Name      string `json:"name"`
	CreatedAt int64  `json:"created_at"`
}


// Message represents the 'message' table in the database.
type Message struct {
	ID        string                 `json:"id"`
	UserID    string                 `json:"user_id"`
	ChannelID *string                `json:"channel_id,omitempty"`
	ParentID  *string                `json:"parent_id,omitempty"`
	Content   string                 `json:"content"`
	Data      map[string]interface{} `json:"data,omitempty"`
	Meta      map[string]interface{} `json:"meta,omitempty"`
	CreatedAt int64                  `json:"created_at"`
	UpdatedAt int64                  `json:"updated_at"`
}

// Model represents the 'model' table in the database.
type Model struct {
	ID            string                 `json:"id"`
	UserID        string                 `json:"user_id"`
	BaseModelID   *string                `json:"base_model_id,omitempty"`
	Name          string                 `json:"name"`
	Params        ModelParams            `json:"params"`
	Meta          ModelMeta              `json:"meta"`
	AccessControl map[string]interface{} `json:"access_control,omitempty"`
	IsActive      bool                   `json:"is_active"`
	UpdatedAt     int64                  `json:"updated_at"`
	CreatedAt     int64                  `json:"created_at"`
}

// Prompt represents the 'prompt' table in the database.
type Prompt struct {
	Command       string                 `json:"command"`
	UserID        string                 `json:"user_id"`
	Title         string                 `json:"title"`
	Content       string                 `json:"content"`
	Timestamp     int64                  `json:"timestamp"`
	AccessControl map[string]interface{} `json:"access_control,omitempty"`
}


// Tool represents the 'tool' table in the database.
type Tool struct {
	ID            string                   `json:"id"`
	UserID        string                   `json:"user_id"`
	Name          string                   `json:"name"`
	Content       string                   `json:"content"`
	Specs         []map[string]interface{} `json:"specs"`
	Meta          map[string]interface{}   `json:"meta"`
	Valves        map[string]interface{}   `json:"valves"`
	AccessControl map[string]interface{}   `json:"access_control,omitempty"`
	UpdatedAt     int64                    `json:"updated_at"`
	CreatedAt     int64                    `json:"created_at"`
}

// ToolMeta represents metadata for a tool.
type ToolMeta struct {
	Description *string                `json:"description,omitempty"`
	Manifest    map[string]interface{} `json:"manifest,omitempty"`
}

// ToolResponse represents a summary of a tool.
type ToolResponse struct {
	ID            string                 `json:"id"`
	UserID        string                 `json:"user_id"`
	Name          string                 `json:"name"`
	Meta          ToolMeta               `json:"meta"`
	AccessControl map[string]interface{} `json:"access_control,omitempty"`
	UpdatedAt     int64                  `json:"updated_at"`
	CreatedAt     int64                  `json:"created_at"`
}

// User represents the 'user' table in the database.
type User struct {
	ID              string                 `json:"id"`
	Name            string                 `json:"name"`
	Email           string                 `json:"email"`
	Role            string                 `json:"role"`
	ProfileImageURL string                 `json:"profile_image_url"`
	LastActiveAt    int64                  `json:"last_active_at"`
	UpdatedAt       int64                  `json:"updated_at"`
	CreatedAt       int64                  `json:"created_at"`
	APIKey          *string                `json:"api_key,omitempty"`
	Settings        *UserSettings          `json:"settings,omitempty"`
	Info            map[string]interface{} `json:"info,omitempty"`
	OauthSub        *string                `json:"oauth_sub,omitempty"`
}

// UserSettings represents the settings for a user. It is extensible.
type UserSettings struct {
	UI map[string]interface{} `json:"ui,omitempty"`
	// This struct is configured with extra="allow", so it can contain other fields.
	// For a statically typed language like Go, you might unmarshal into a map
	// or use a library that supports embedding for extra fields.
}
```
