package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"mcts-agent/llm"
	"mcts-agent/web"
)

func main() {
	if err := loadEnv(".env"); err != nil && !os.IsNotExist(err) {
		log.Printf("Warning: could not load .env file: %v", err)
	}

	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	if apiKey == "" {
		log.Fatal("DEEPSEEK_API_KEY environment variable is not set")
	}

	llmCfg := llm.Config{
		APIKey:  apiKey,
		BaseURL: os.Getenv("DEEPSEEK_BASE_URL"),
		Model:   os.Getenv("DEEPSEEK_MODEL"),
		Timeout: 60 * time.Second,
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	srv := web.NewServer(llmCfg)

	httpServer := &http.Server{
		Addr:    ":" + port,
		Handler: srv.Handler(),
	}

	// 可选：搜索完成后自动关闭（设置 AUTO_SHUTDOWN=true 时启用）
	if os.Getenv("AUTO_SHUTDOWN") == "true" {
		srv.SetPostSearchHook(func() {
			log.Println("搜索完成，5 秒后自动关闭...")
			time.Sleep(5 * time.Second)
			httpServer.Shutdown(context.Background())
		})
	}

	log.Printf("启动 MCTS Agent Web UI: http://localhost%s", httpServer.Addr)
	log.Println("提交任务搜索完成后需手动关闭 (Ctrl+C)，或设置 AUTO_SHUTDOWN=true 自动退出。")

	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Server failed: %v", err)
	}

	log.Println("服务已关闭。")
}

func loadEnv(filename string) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return err
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		value := strings.TrimSpace(parts[1])
		if os.Getenv(key) == "" {
			os.Setenv(key, value)
		}
	}
	return nil
}
