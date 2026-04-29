package llm

import (
	"context"
	"fmt"

	"github.com/sashabaranov/go-openai"
)

// LLMClient 封装了大模型调用的客户端
type LLMClient struct {
	client *openai.Client
	model  string
}

// NewLLMClient 是一个构造函数，用于初始化客户端
func NewLLMClient(apiKey string) *LLMClient {
	// 因为我们要用 DeepSeek（或者其他兼容 API），需要修改默认的 BaseURL
	config := openai.DefaultConfig(apiKey)
	config.BaseURL = "https://api.deepseek.com/v1" // 注意：大多数兼容接口都需要带上 /v1

	client := openai.NewClientWithConfig(config)

	return &LLMClient{
		client: client,
		model:  "deepseek-chat", // 如果用其他模型，修改这里
	}
}

// Ask 是具体向模型发请求的方法
func (c *LLMClient) Ask(sysPrompt string, userPrompt string) (string, error) {
	// Go 语言推荐使用 Context 来控制请求的超时和取消
	ctx := context.Background()

	req := openai.ChatCompletionRequest{
		Model: c.model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: sysPrompt,
			},
			{
				Role:    openai.ChatMessageRoleUser,
				Content: userPrompt,
			},
		},
		Temperature: 0.7,
	}

	resp, err := c.client.CreateChatCompletion(ctx, req)
	if err != nil {
		// Go 语言标志性的显式错误返回
		return "", fmt.Errorf("调用 LLM API 失败: %v", err)
	}

	return resp.Choices[0].Message.Content, nil
}
