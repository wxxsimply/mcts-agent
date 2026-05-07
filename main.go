package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"time"

	"mcts-agent/llm"
	"mcts-agent/mcts"
)

func main() {
	if err := loadEnv(".env"); err != nil && !os.IsNotExist(err) {
		log.Printf("Warning: could not load .env file: %v", err)
	}

	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	if apiKey == "" {
		log.Fatal("DEEPSEEK_API_KEY environment variable is not set")
	}

	task := getTask()

	testCases := getTestCases()
	if len(testCases) > 0 {
		log.Printf("已输入 %d 组测试用例，将按用例评分", len(testCases))
	} else {
		log.Println("无测试用例，将按编译/运行结果评分（信号较弱）")
	}

	client := llm.NewLLMClient(llm.Config{
		APIKey:  apiKey,
		BaseURL: "https://api.deepseek.com/v1",
		Model:   "deepseek-v4-flash",
		Timeout: 30 * time.Second,
	})

	engine := mcts.NewMCTSEngine(client).
		WithMaxDepth(4).
		WithExploreC(1.414).
		WithTestCases(testCases)

	// 支持 Ctrl+C 优雅退出
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, os.Interrupt)
		<-sigCh
		cancel()
		log.Println("Received interrupt, shutting down...")
	}()

	root := engine.RunConcurrent(task, 20, 5)

	bestCode := root.GetBestCode()
	fmt.Println("\n=== MCTS Best Solution ===")
	fmt.Println(bestCode)
	_ = ctx
}

// getTask 获取用户问题：优先取命令行参数，否则交互式输入
func getTask() string {
	if len(os.Args) > 1 {
		return strings.Join(os.Args[1:], " ")
	}

	fmt.Println("请输入你的问题/任务（输入空行结束）：")
	scanner := bufio.NewScanner(os.Stdin)
	var lines []string
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			break
		}
		lines = append(lines, line)
	}
	if len(lines) == 0 {
		log.Fatal("未输入任何问题，退出。")
	}
	return strings.Join(lines, "\n")
}

// getTestCases 交互式输入测试用例
func getTestCases() []mcts.TestCase {
	fmt.Print("\n是否输入测试用例？(y/n): ")
	scanner := bufio.NewScanner(os.Stdin)
	if !scanner.Scan() {
		return nil
	}
	if strings.ToLower(strings.TrimSpace(scanner.Text())) != "y" {
		return nil
	}

	fmt.Print("测试用例数量: ")
	if !scanner.Scan() {
		return nil
	}
	n, err := strconv.Atoi(strings.TrimSpace(scanner.Text()))
	if err != nil || n <= 0 {
		log.Printf("无效数量，跳过测试用例。")
		return nil
	}

	cases := make([]mcts.TestCase, 0, n)
	fmt.Println("--- 请依次输入每个测试用例 ---")
	for i := 0; i < n; i++ {
		fmt.Printf("\n用例 %d/%d:\n", i+1, n)

		fmt.Print("  输入（标准输入，无输入则直接回车）: ")
		input := ""
		if scanner.Scan() {
			input = scanner.Text()
		}

		fmt.Print("  期望输出: ")
		var output string
		if scanner.Scan() {
			output = strings.TrimSpace(scanner.Text())
		}

		cases = append(cases, mcts.TestCase{
			Input:  input,
			Output: output,
		})
	}

	return cases
}

// loadEnv 读取 .env 文件并设置环境变量（不覆盖已存在的值）
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
