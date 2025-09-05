package conversations

import (
	"fmt"
	"time"
)

// Message represents a conversation message
type Message struct {
	ID        string                 `json:"id"`
	Role      string                 `json:"role"` // user, assistant, system
	Content   string                 `json:"content"`
	Timestamp time.Time              `json:"timestamp"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// Conversation represents a conversation session
type Conversation struct {
	ID        string              `json:"id"`
	Messages  []*Message          `json:"messages"`
	CreatedAt time.Time           `json:"created_at"`
	UpdatedAt time.Time           `json:"updated_at"`
	Config    *ConversationConfig `json:"config,omitempty"`
}

// ConversationConfig holds conversation configuration
type ConversationConfig struct {
	MaxMessages   int           `json:"max_messages"`
	ContextWindow int           `json:"context_window"`
	Timeout       time.Duration `json:"timeout"`
}

// ConversationService manages conversation sessions
type ConversationService struct {
	conversations  map[string]*Conversation
	messageCounter int
}

// NewConversationService creates a new conversation service
func NewConversationService() *ConversationService {
	return &ConversationService{
		conversations:  make(map[string]*Conversation),
		messageCounter: 0,
	}
}

// CreateConversation creates a new conversation
func (cs *ConversationService) CreateConversation(config *ConversationConfig) (*Conversation, error) {
	if config == nil {
		config = &ConversationConfig{
			MaxMessages:   100,
			ContextWindow: 50,
			Timeout:       30 * time.Minute,
		}
	}

	id := fmt.Sprintf("conv_%d", time.Now().Unix())
	conversation := &Conversation{
		ID:        id,
		Messages:  []*Message{},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		Config:    config,
	}

	cs.conversations[id] = conversation
	return conversation, nil
}

// GetConversation retrieves a conversation by ID
func (cs *ConversationService) GetConversation(id string) (*Conversation, error) {
	conversation, exists := cs.conversations[id]
	if !exists {
		return nil, fmt.Errorf("conversation not found: %s", id)
	}
	return conversation, nil
}

// AddMessage adds a message to a conversation
func (cs *ConversationService) AddMessage(convID string, message *Message) error {
	conversation, exists := cs.conversations[convID]
	if !exists {
		return fmt.Errorf("conversation not found: %s", convID)
	}

	// Set message ID and timestamp
	cs.messageCounter++
	message.ID = fmt.Sprintf("msg_%d", cs.messageCounter)
	if message.Timestamp.IsZero() {
		message.Timestamp = time.Now()
	}

	conversation.Messages = append(conversation.Messages, message)
	conversation.UpdatedAt = time.Now()

	// Implement context window management
	if len(conversation.Messages) > conversation.Config.ContextWindow {
		// Keep the most recent messages within the context window
		start := len(conversation.Messages) - conversation.Config.ContextWindow
		conversation.Messages = conversation.Messages[start:]
	}

	return nil
}

// GetMessages retrieves messages from a conversation
func (cs *ConversationService) GetMessages(convID string, limit int) ([]*Message, error) {
	conversation, exists := cs.conversations[convID]
	if !exists {
		return nil, fmt.Errorf("conversation not found: %s", convID)
	}

	messages := conversation.Messages
	if limit > 0 && len(messages) > limit {
		start := len(messages) - limit
		messages = messages[start:]
	}

	return messages, nil
}

// DeleteConversation removes a conversation
func (cs *ConversationService) DeleteConversation(id string) error {
	if _, exists := cs.conversations[id]; !exists {
		return fmt.Errorf("conversation not found: %s", id)
	}
	delete(cs.conversations, id)
	return nil
}

// ListConversations returns all conversation IDs
func (cs *ConversationService) ListConversations() []string {
	ids := make([]string, 0, len(cs.conversations))
	for id := range cs.conversations {
		ids = append(ids, id)
	}
	return ids
}
