package llm

import (
	"context"
	"fmt"
	"time"

	"github.com/sashabaranov/go-openai"
)

// Config holds LLM client configuration.
type Config struct {
	APIKey  string
	BaseURL string
	Model   string
	Timeout time.Duration
}

// LLMClient 封装了大模型调用的客户端
type LLMClient struct {
	client  *openai.Client
	model   string
	timeout time.Duration
}

// NewLLMClient 使用配置初始化客户端
func NewLLMClient(cfg Config) *LLMClient {
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "https://api.deepseek.com/v1"
	}
	model := cfg.Model
	if model == "" {
		model = "deepseek-v4-flash"
	}
	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}

	openaiCfg := openai.DefaultConfig(cfg.APIKey)
	openaiCfg.BaseURL = baseURL

	return &LLMClient{
		client:  openai.NewClientWithConfig(openaiCfg),
		model:   model,
		timeout: timeout,
	}
}

// Ask 向模型发送请求，带 context 支持和简单重试
func (c *LLMClient) Ask(ctx context.Context, sysPrompt string, userPrompt string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	req := openai.ChatCompletionRequest{
		Model: c.model,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: sysPrompt},
			{Role: openai.ChatMessageRoleUser, Content: userPrompt},
		},
		Temperature: 0.7,
	}

	// 简单重试：失败后重试一次
	for attempt := 0; attempt < 2; attempt++ {
		resp, err := c.client.CreateChatCompletion(ctx, req)
		if err == nil {
			return resp.Choices[0].Message.Content, nil
		}
		if attempt == 0 {
			time.Sleep(time.Second)
		} else {
			return "", fmt.Errorf("LLM API failed after retry: %v", err)
		}
	}
	return "", fmt.Errorf("LLM API call failed")
}
