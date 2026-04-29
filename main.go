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
	task := "编写一个 c++ 程序，写出以下题目：# P10565 [ICPC 2024 Xi'an I] Chained Lights\n\n## 题目描述\n\n你有 $n$ 盏灯排成一行。最初，它们都是关闭的。  你将逐个按下这 $n$ 盏灯。当你按下灯 $i$ 时，灯 $i$ 将切换其状态，这意味着如果它是关闭的，它将打开；如果它是打开的，它将关闭。然后对于每个满足 $i|j, i < j \\le n$ 的 $j$，按下灯 $j$ 一次。  例如，如果 $n=4$，当你按下灯 $1$ 时，灯 $1$ 将打开，然后你将按下灯 $2,3,4$。由于你按下了灯 $2$，灯 $2$ 将打开，你将按下灯 $4$，这将导致灯 $4$ 打开。经过所有操作后，灯 $1,2,3$ 将打开，而灯 $4$ 仍然关闭。  你将逐个按下这 $n$ 盏灯并进行上述操作。经过所有操作后，你想知道灯 $k$ 是开着还是关着的。  你也可以使用以下代码来理解问题的含义： \n```cpp \nvoid press(int x) {\n    light[x]^=1;\n    for (int y=x+x; y<=n; y+=x) press(y);\n}\nfor (int i=1; i<=n; i++) press(i);\n```\n\n## 输入格式\n\n有多个测试用例。  第一行包含一个整数 $T(1 \\le T \\le 10^5)$，表示测试用例的数量。  每个测试用例包含两个整数 $n, k(1 \\le k \\le n \\le 10^6)$，在一行中。\n\n## 输出格式\n\n对于每个测试用例，如果灯 $k$ 最终是打开的，输出 `YES`，否则输出 `NO`。\n\n## 输入输出样例 #1\n\n### 输入 #1\n\n```\n2\n1 1\n3 2\n```\n\n### 输出 #1\n\n```\nYES\nNO\n```\n\n## 说明/提示\n\n（由 ChatGPT 4o 翻译）"
	//root := mcts.NewRootNode(task)
	//root.TaskType = "code" // 标记这是一个编程任务

	// 获取返回的 root 节点
	root := engine.RunConcurrent(task, 20, 5)

	// 调用 GetBestCode 输出最优结果
	bestCode := root.GetBestCode()

	fmt.Println("\n--- MCTS 搜索出的最优代码 ---")
	fmt.Println(bestCode)
}
