package api_test

import (
	"encoding/json"
	"fmt"
	"reflect"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"internal-perplexity/server/llm/api"
)

func strPtr(s string) *string { return &s }

var _ = Describe("Chat message model", func() {
	Describe("UserMessageContent (string vs parts) JSON (un)marshal", func() {
		It("unmarshals from a string", func() {
			input := `"Hello there"`
			var c api.MessageContent

			By("printing before")
			GinkgoWriter.Println("Before (JSON):", input)

			err := json.Unmarshal([]byte(input), &c)
			Expect(err).ToNot(HaveOccurred())
			Expect(c.String).ToNot(BeNil())
			Expect(*c.String).To(Equal("Hello there"))
			Expect(c.Parts).To(BeNil())

			out, err := json.Marshal(c)
			Expect(err).ToNot(HaveOccurred())

			By("printing after")
			GinkgoWriter.Println("After (JSON):", string(out))
			Expect(string(out)).To(Equal(`"Hello there"`))
		})

		It("unmarshals from an array of parts", func() {
			input := `
			[
				{"type":"text","text":"show image"},
				{"type":"image_url","image_url":{"url":"https://example.com/cat.png","detail":"low"}}
			]
			`

			var c api.MessageContent

			By("printing before")
			GinkgoWriter.Println("Before (JSON):", input)

			err := json.Unmarshal([]byte(input), &c)
			Expect(err).ToNot(HaveOccurred())
			Expect(c.String).To(BeNil())
			Expect(c.Parts).To(HaveLen(2))

			part1, ok := c.Parts[0].(api.ChatCompletionMessageContentPartText)
			Expect(ok).To(BeTrue())
			Expect(part1.Type).To(Equal("text"))
			Expect(part1.Text).To(Equal("show image"))

			part2, ok := c.Parts[1].(api.ChatCompletionMessageContentPartImage)
			Expect(ok).To(BeTrue())
			Expect(part2.Type).To(Equal("image_url"))
			Expect(part2.ImageURL.URL).To(Equal("https://example.com/cat.png"))
			Expect(part2.ImageURL.Detail).To(Equal("low"))

			out, err := json.Marshal(c)
			Expect(err).ToNot(HaveOccurred())

			By("printing after")
			GinkgoWriter.Println("After (JSON):", string(out))
			// Verify it's an array, not a string
			Expect(out[0]).To(Equal(byte('[')))
		})

		It("errors when user content is neither string nor array", func() {
			var c api.MessageContent
			err := json.Unmarshal([]byte(`{"foo":"bar"}`), &c)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("must be string or array"))
		})
	})

	Describe("UnmarshalChatMessages", func() {
		It("parses a mixed list of messages for all roles", func() {
			input := `
			[
				{"role":"system","content":"You are helpful."},
				{"role":"user","content":"Hello!"},
				{
					"role":"user",
					"content":[
						{"type":"text","text":"show image"},
						{"type":"image_url","image_url":{"url":"https://example.com/img.png","detail":"high"}}
					]
				},
				{"role":"assistant","content":"Hi there!"},
				{
					"role":"assistant",
					"tool_calls":[
						{
							"id":"call_1",
							"type":"function",
							"function":{"name":"search","arguments":"{\"q\":\"golang\"}"}
						}
					]
				},
				{"role":"tool","content":"{\"results\":[]}", "tool_call_id":"call_1"}
			]
			`

			By("printing before (input JSON)")
			GinkgoWriter.Println("Before (JSON):", input)

			msgs, err := api.UnmarshalChatMessages([]byte(input))
			Expect(err).ToNot(HaveOccurred())
			Expect(msgs).To(HaveLen(6))

			// system
			sys, ok := msgs[0].(api.ChatCompletionSystemMessage)
			Expect(ok).To(BeTrue())
			Expect(sys.Role).To(Equal(api.RoleSystem))
			Expect(sys.Content).To(Equal("You are helpful."))

			// user (string)
			usr1, ok := msgs[1].(api.ChatCompletionUserMessage)
			Expect(ok).To(BeTrue())
			Expect(usr1.Role).To(Equal(api.RoleUser))
			Expect(usr1.Content.String).ToNot(BeNil())
			Expect(*usr1.Content.String).To(Equal("Hello!"))
			Expect(usr1.Content.Parts).To(BeNil())

			// user (parts)
			usr2, ok := msgs[2].(api.ChatCompletionUserMessage)
			Expect(ok).To(BeTrue())
			Expect(usr2.Role).To(Equal(api.RoleUser))
			Expect(usr2.Content.String).To(BeNil())
			Expect(usr2.Content.Parts).To(HaveLen(2))

			// assistant (content)
			asst1, ok := msgs[3].(api.ChatCompletionAssistantMessage)
			Expect(ok).To(BeTrue())
			Expect(asst1.Role).To(Equal(api.RoleAssistant))
			Expect(asst1.Content).To(Equal("Hi there!"))
			Expect(asst1.ToolCalls).To(BeNil())

			// assistant (tool calls)
			asst2, ok := msgs[4].(api.ChatCompletionAssistantMessage)
			Expect(ok).To(BeTrue())
			Expect(asst2.Role).To(Equal(api.RoleAssistant))
			Expect(asst2.Content).To(Equal(""))
			Expect(asst2.ToolCalls).To(HaveLen(1))
			Expect(asst2.ToolCalls[0].ID).To(Equal("call_1"))
			Expect(asst2.ToolCalls[0].Type).To(Equal("function"))
			Expect(asst2.ToolCalls[0].Function.Name).To(Equal("search"))
			Expect(asst2.ToolCalls[0].Function.Arguments).To(Equal(`{"q":"golang"}`))

			// tool
			tool, ok := msgs[5].(api.ChatCompletionToolMessage)
			Expect(ok).To(BeTrue())
			Expect(tool.Role).To(Equal(api.RoleTool))
			Expect(tool.ToolCallID).To(Equal("call_1"))
			Expect(tool.Content).To(Equal(`{"results":[]}`))

			// Print after (normalized JSON)
			out, err := api.MarshalChatMessages(msgs)
			Expect(err).ToNot(HaveOccurred())
			By("printing after (normalized JSON)")
			GinkgoWriter.Println("After (JSON):", string(out))
		})

		It("returns error on unknown role", func() {
			input := `[{"role":"weird","content":"nope"}]`
			_, err := api.UnmarshalChatMessages([]byte(input))
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring(`unknown role`))
		})

		It("returns error on unknown content part type inside user content", func() {
			input := `
			[
				{"role":"user","content":[{"type":"unknown","text":"??"}]}
			]
			`
			_, err := api.UnmarshalChatMessages([]byte(input))
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring(`unknown content part type`))
		})
	})

	Describe("MarshalChatMessages", func() {
		It("marshals and round-trips a mixed message list", func() {
			msgs := []api.ChatCompletionMessage{
				api.ChatCompletionSystemMessage{
					Role: api.RoleSystem,
					Content: api.MessageContent{
						String: strPtr("System prompt"),
					},
				},
				api.ChatCompletionUserMessage{
					Role: api.RoleUser,
					Content: api.MessageContent{
						String: strPtr("Hi"),
					},
				},
				api.ChatCompletionUserMessage{
					Role: api.RoleUser,
					Content: api.MessageContent{
						Parts: []api.ChatCompletionMessageContentPart{
							api.ChatCompletionMessageContentPartText{
								Type: "text",
								Text: "show image",
							},
							api.ChatCompletionMessageContentPartImage{
								Type: "image_url",
								ImageURL: api.ImageURL{
									URL:    "https://example.com/a.png",
									Detail: "auto",
								},
							},
						},
					},
				},
				api.ChatCompletionAssistantMessage{
					Role: api.RoleAssistant,
					Content: api.MessageContent{
						String: strPtr("Reply"),
					},
				},
				api.ChatCompletionAssistantMessage{
					Role: api.RoleAssistant,
					ToolCalls: []api.ToolCall{
						{
							ID:   "call_2",
							Type: "function",
							Function: api.FunctionCall{
								Name:      "lookup",
								Arguments: `{"id":123}`,
							},
						},
					},
				},
				api.ChatCompletionToolMessage{
					Role:       api.RoleTool,
					ToolCallID: "call_2",
					Content:    `{"result":"ok"}`,
				},
			}

			before, err := api.MarshalChatMessages(msgs)
			Expect(err).ToNot(HaveOccurred())

			By("printing before (marshaled JSON)")
			GinkgoWriter.Println("Before (JSON):", string(before))

			rt, err := api.UnmarshalChatMessages(before)
			Expect(err).ToNot(HaveOccurred())
			Expect(rt).To(HaveLen(len(msgs)))

			// Validate a few representative fields after round-trip
			Expect(rt[0].(api.ChatCompletionSystemMessage).Content).
				To(Equal("System prompt"))
			Expect(*rt[1].(api.ChatCompletionUserMessage).Content.String).
				To(Equal("Hi"))

			u2 := rt[2].(api.ChatCompletionUserMessage)
			Expect(u2.Content.Parts).To(HaveLen(2))
			_, ok := u2.Content.Parts[0].(api.ChatCompletionMessageContentPartText)
			Expect(ok).To(BeTrue())
			img, ok := u2.Content.Parts[1].(api.ChatCompletionMessageContentPartImage)
			Expect(ok).To(BeTrue())
			Expect(img.ImageURL.URL).To(Equal("https://example.com/a.png"))

			asst2 := rt[4].(api.ChatCompletionAssistantMessage)
			Expect(asst2.ToolCalls).To(HaveLen(1))
			Expect(asst2.ToolCalls[0].Function.Name).To(Equal("lookup"))

			after, err := api.MarshalChatMessages(rt)
			Expect(err).ToNot(HaveOccurred())

			By("printing after (re-marshaled JSON)")
			GinkgoWriter.Println("After (JSON):", string(after))

			// We won't require byte-for-byte equality due to ordering nuances,
			// but we can ensure both unmarshal to equivalent structures.
			rt2, err := api.UnmarshalChatMessages(after)
			Expect(err).ToNot(HaveOccurred())

			// Compare selected fields across both round-trips
			Expect(len(rt2)).To(Equal(len(rt)))
			for i := range rt {
				Expect(reflect.TypeOf(rt2[i])).To(Equal(reflect.TypeOf(rt[i])))
			}
		})

		It("marshals UserMessageContent as string vs array appropriately", func() {
			asString := api.ChatCompletionUserMessage{
				Role: api.RoleUser,
				Content: api.MessageContent{
					String: strPtr("hello"),
				},
			}
			asArray := api.ChatCompletionUserMessage{
				Role: api.RoleUser,
				Content: api.MessageContent{
					Parts: []api.ChatCompletionMessageContentPart{
						api.ChatCompletionMessageContentPartText{
							Type: "text",
							Text: "hello",
						},
					},
				},
			}

			b1, err := json.Marshal(asString)
			Expect(err).ToNot(HaveOccurred())
			Expect(string(b1)).To(ContainSubstring(`"content":"hello"`))

			b2, err := json.Marshal(asArray)
			Expect(err).ToNot(HaveOccurred())
			Expect(string(b2)).To(ContainSubstring(`"content":[`))
			Expect(string(b2)).To(ContainSubstring(`"type":"text"`))

			By("printing both encodings for visual confirmation")
			GinkgoWriter.Println("User as string:", string(b1))
			GinkgoWriter.Println("User as array:", string(b2))
		})
	})

	Describe("Type safety helpers", func() {
		It("exposes roles via GetRole()", func() {
			var msgs []api.ChatCompletionMessage = []api.ChatCompletionMessage{
				api.ChatCompletionSystemMessage{Role: api.RoleSystem},
				api.ChatCompletionUserMessage{Role: api.RoleUser},
				api.ChatCompletionAssistantMessage{Role: api.RoleAssistant},
				api.ChatCompletionToolMessage{Role: api.RoleTool},
			}

			var roles []api.Role
			for _, m := range msgs {
				roles = append(roles, m.GetRole())
			}
			Expect(roles).To(Equal([]api.Role{
				api.RoleSystem, api.RoleUser, api.RoleAssistant, api.RoleTool,
			}))

			By("printing roles")
			GinkgoWriter.Println("Roles:", fmt.Sprint(roles))
		})
	})
})
