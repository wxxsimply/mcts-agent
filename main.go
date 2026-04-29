package main

import (
	"fmt"
	"mcts-agent/mcts"

	// 引入你刚刚写的内部包 (假设你的 module name 叫 mcts-agent)
	"mcts-agent/llm"
)

func main() {
	apiKey := ""
	client := llm.NewLLMClient(apiKey)
	engine := mcts.NewMCTSEngine(client)

	// 关键点：根节点的内容决定了搜索的方向
	task := ""
	//root := mcts.NewRootNode(task)
	//root.TaskType = "code" // 标记这是一个编程任务

	// 获取返回的 root 节点
	root := engine.RunConcurrent(task, 20, 5)

	// 调用 GetBestCode 输出最优结果
	bestCode := root.GetBestCode()

	fmt.Println("\n--- MCTS 搜索出的最优代码 ---")
	fmt.Println(bestCode)
}
