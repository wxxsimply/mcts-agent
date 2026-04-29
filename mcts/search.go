package mcts

import (
	"encoding/json"
	"fmt"
	"math"
	"mcts-agent/llm" // 注意替换为你真实的工程 module 名
	"mcts-agent/runner"
	"strings"
	"sync"
	"sync/atomic"
)

// ---------------- 结构化输出协议 ----------------

type ExpandResponse struct {
	Actions []string `json:"actions"`
}

type EvaluateResponse struct {
	Score float64 `json:"score"`
}

// ---------------- MCTS 引擎定义 ----------------

type MCTSEngine struct {
	llmClient *llm.LLMClient
	exploreC  float64
	maxDepth  int
}

func NewMCTSEngine(client *llm.LLMClient) *MCTSEngine {
	return &MCTSEngine{
		llmClient: client,
		exploreC:  math.Sqrt(2), // 经典探索常数
		maxDepth:  4,
	}

}

// Select: 寻找 UCB 最大的叶子节点
func (e *MCTSEngine) selectNode(node *Node) *Node {
	node.Mu.RLock()
	defer node.Mu.RUnlock()

	var bestNode *Node
	maxUCB := math.Inf(-1)

	for _, child := range node.Children {
		vLoss := float64(atomic.LoadInt32(&child.VirtualLoss))
		var ucb float64

		if child.Visits == 0 {
			ucb = 10000.0 + vLoss // 尚未访问的节点赋予极高优先级
		} else {
			// UCB 公式结合 Virtual Loss 惩罚
			exploitation := child.Value / child.Visits
			exploration := e.exploreC * math.Sqrt(math.Log(node.Visits)/(child.Visits+vLoss))
			ucb = exploitation + exploration
		}

		if ucb > maxUCB {
			maxUCB = ucb
			bestNode = child
		}
	}
	return bestNode
}

// Expand: 调用 LLM 生成下一步动作
func (e *MCTSEngine) expand(node *Node) error {
	fmt.Printf("[LLM] 正在扩展节点: %s...\n", node.Thought) // 加个日志
	// search.go 中的 expand 函数
	sysPrompt := `你是一个 acm c++算法竞赛专家。请针对任务提供一种实现，要求：
	1.如果题目给出了代码，请寻找代码的规律并以此写出代码
	2.代码能通过洛谷官网的的所有测试点
	3.代码运用到题目给的所有信息
	4.时间复杂度控制在题目要求以内。
	5.代码控制在30行以内
	请直接返回 JSON 格式: {"actions": ["完整的Go代码"]}，不要重复之前生成过的代码。`
	userPrompt := fmt.Sprintf("当前状态路径:\n%s\n请给出下一步动作。", node.GetPath())

	resp, err := e.llmClient.Ask(sysPrompt, userPrompt)
	if err != nil {
		return err
	}

	var expResp ExpandResponse
	if err := json.Unmarshal([]byte(e.cleanJSON(resp)), &expResp); err != nil {
		return err
	}

	for _, action := range expResp.Actions {
		node.AddChild(action)
	}
	return nil
}

func (e *MCTSEngine) evaluate(node *Node) (float64, error) {
	res := runner.ExecuteCode(node.Thought, "temp.go")
	// 增加负分惩罚，迫使模型避开无效路径
	if res.Err != nil {
		return -0.5, nil // 从原来的 0.1 改为负分，这样树会彻底放弃这条路
	}
	// 基础分 0.5，输出正确关键词再加 0.5
	score := 0.5
	if strings.Contains(res.Stdout, "Hello MCTS") {
		score += 0.5
	}
	return score, nil
}

// Backpropagate: 回溯更新
func (e *MCTSEngine) backpropagate(path []*Node, score float64) {
	// 4. Backpropagate (修改这里的逻辑)
	for _, p := range path {
		if p == nil { // 增加这个防守性编程检查
			continue
		}
		p.Mu.Lock()
		p.Visits++
		p.Value += score
		p.Mu.Unlock()
		atomic.AddInt32(&p.VirtualLoss, -1)
	}
}

// RunConcurrent: 并发搜索入口
func (e *MCTSEngine) RunConcurrent(initialState string, iters int, workers int) *Node {
	root := NewRootNode(initialState)
	var wg sync.WaitGroup

	fmt.Printf("开始并发 MCTS 搜索...\n")

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for j := 0; j < iters/workers; j++ {
				// 1. Selection
				curr := root
				var path []*Node
				// search.go: 139行左右
				for {
					path = append(path, curr)
					atomic.AddInt32(&curr.VirtualLoss, 1)

					curr.Mu.RLock()
					isLeaf := len(curr.Children) == 0
					curr.Mu.RUnlock()

					if isLeaf {
						break
					}

					next := e.selectNode(curr) // 先获取下一个节点
					if next == nil {           // 必须检查是否为 nil
						break // 如果找不到子节点，说明到了叶子节点
					}
					curr = next
				}

				// 2. Expand
				curr.Mu.Lock()
				shouldExpand := len(curr.Children) == 0
				curr.Mu.Unlock()

				if shouldExpand {
					err := e.expand(curr) // 此时的 curr 已经是叶子节点
					if err != nil {
						fmt.Printf("[Worker %d] Expand 失败: %v\n", workerID, err)
						// 发生错误时必须清理路径上的 VirtualLoss
						for _, p := range path {
							atomic.AddInt32(&p.VirtualLoss, -1)
						}
						continue
					}
				}

				// 3. Evaluate (必须在扩展后执行)
				// 如果刚刚扩展了，从子节点选一个评估；如果没扩展，直接评估当前
				evalTarget := curr
				curr.Mu.RLock()
				if len(curr.Children) > 0 {
					evalTarget = curr.Children[0]
				}
				curr.Mu.RUnlock()

				// 关键改动：添加 nil 检查
				if evalTarget == nil {
					continue
				}

				score, _ := e.evaluate(evalTarget)
				// 在 RunConcurrent 的循环末尾调用 backpropagate 前：
				if len(path) > 0 {
					e.backpropagate(path, score)
				}
				// 4. Backpropagate (更新所有路径节点)
				// 注意：如果选了子节点评估，需要把子节点也加进路径
				if evalTarget != curr {
					path = append(path, evalTarget)
				}
				e.backpropagate(path, score)

				fmt.Printf("[Worker %d] 迭代 %d 完成\n", workerID, j)
			}
		}(i)
	}
	wg.Wait()
	fmt.Println("\n--- 最终搜索树结果 ---")
	return root
}

// cleanJSON 处理 LLM 返回的 Markdown 代码块污染
func (e *MCTSEngine) cleanJSON(input string) string {
	start := strings.Index(input, "{")
	end := strings.LastIndex(input, "}")
	if start == -1 || end == -1 {
		return input
	}
	return input[start : end+1]
}
