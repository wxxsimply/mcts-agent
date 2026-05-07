package mcts

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"mcts-agent/llm"
	"mcts-agent/runner"
	"strings"
	"sync"
	"sync/atomic"
)

// ------------- 默认配置常量 -------------

const (
	DefaultExploreC   = 1.414 // sqrt(2)
	DefaultMaxDepth   = 4
	UnvisitedPriority = 10000.0
	ExpandPenalty     = -0.5
)

// ------------- 结构化输出协议 -------------

type ExpandResponse struct {
	Actions []string `json:"actions"`
}

// TestCase 定义评测用例
type TestCase struct {
	Input  string // 标准输入（为空表示无需输入）
	Output string // 期望的标准输出
}

// ------------- MCTS 引擎定义 -------------

type MCTSEngine struct {
	llmClient *llm.LLMClient
	exploreC  float64
	maxDepth  int
	testCases []TestCase

	// 去重：记录所有已生成的代码
	generatedMu  sync.Mutex
	generatedSet map[string]bool
}

func NewMCTSEngine(client *llm.LLMClient) *MCTSEngine {
	return &MCTSEngine{
		llmClient: client,
		exploreC:  DefaultExploreC,
		maxDepth:  DefaultMaxDepth,
	}
}

// WithTestCases 设置评测用例
func (e *MCTSEngine) WithTestCases(tcs []TestCase) *MCTSEngine {
	e.testCases = tcs
	return e
}

// WithExploreC 设置探索常数
func (e *MCTSEngine) WithExploreC(c float64) *MCTSEngine {
	e.exploreC = c
	return e
}

// WithMaxDepth 设置最大搜索深度
func (e *MCTSEngine) WithMaxDepth(d int) *MCTSEngine {
	e.maxDepth = d
	return e
}

// ------------- 核心 MCTS 方法 -------------

// selectNode: 选择 UCB 最大的子节点
func (e *MCTSEngine) selectNode(node *Node) *Node {
	node.Mu.RLock()
	defer node.Mu.RUnlock()

	var bestNode *Node
	maxUCB := math.Inf(-1)

	for _, child := range node.Children {
		vLoss := float64(atomic.LoadInt32(&child.VirtualLoss))
		var ucb float64

		if child.Visits == 0 {
			ucb = UnvisitedPriority + vLoss
		} else {
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

// buildExpandPrompt: 组装通用化的扩展提示
func (e *MCTSEngine) buildExpandPrompt(pathThought string, previousCount int) (string, string) {
	sysPrompt := `You are an expert algorithm engineer. Generate correct, efficient, and well-structured Go solutions.

Requirements:
1. The code must compile and run correctly
2. Handle edge cases properly
3. Optimize time and space complexity within reasonable limits
4. Write clean, readable Go code

Return strictly as JSON: {"actions": ["complete Go code"]}
Each action is a complete, independent solution. Generate diverse approaches or algorithmic variations when possible.
Do NOT generate duplicate or near-duplicate code.`

	userPrompt := fmt.Sprintf("Task:\n%s\n\n(Already generated %d unique solutions — please generate new variations.)",
		pathThought, previousCount)
	return sysPrompt, userPrompt
}

// expand: 调用 LLM 生成子节点（带去重）
func (e *MCTSEngine) expand(node *Node) error {
	log.Printf("[LLM] Expanding node: %s...", truncate(node.Thought, 80))

	e.generatedMu.Lock()
	prevCount := len(e.generatedSet)
	e.generatedMu.Unlock()

	sysPrompt, userPrompt := e.buildExpandPrompt(node.GetPath(), prevCount)

	resp, err := e.llmClient.Ask(context.Background(), sysPrompt, userPrompt)
	if err != nil {
		return fmt.Errorf("LLM expand failed: %v", err)
	}

	var expResp ExpandResponse
	if err := json.Unmarshal([]byte(e.cleanJSON(resp)), &expResp); err != nil {
		return fmt.Errorf("JSON parse failed: %v\nraw: %s", err, resp)
	}

	e.generatedMu.Lock()
	added := 0
	for _, action := range expResp.Actions {
		trimmed := strings.TrimSpace(action)
		if trimmed == "" {
			continue
		}
		if !e.generatedSet[trimmed] {
			e.generatedSet[trimmed] = true
			node.AddChild(trimmed)
			added++
		}
	}
	e.generatedMu.Unlock()

	log.Printf("[LLM] Generated %d actions, %d new after dedup", len(expResp.Actions), added)
	return nil
}

// evaluate: 执行代码并评分。有测试用例时按测试用例评分，否则按编译/运行结果给基础分。
func (e *MCTSEngine) evaluate(node *Node) float64 {
	if len(e.testCases) > 0 {
		return e.evaluateWithCases(node)
	}
	return e.evaluateDefault(node)
}

// evaluateWithCases: 按测试用例评分（支持标准输入）
func (e *MCTSEngine) evaluateWithCases(node *Node) float64 {
	passed := 0
	total := len(e.testCases)
	for _, tc := range e.testCases {
		var res runner.ExecutionResult
		if tc.Input != "" {
			res = runner.ExecuteCodeWithStdin(node.Thought, "temp_solution.go", tc.Input)
		} else {
			res = runner.ExecuteCode(node.Thought, "temp_solution.go")
		}

		if res.Err != nil {
			log.Printf("[Eval] 用例失败(stderr): %s", res.Stderr)
			continue
		}

		got := strings.TrimSpace(res.Stdout)
		want := strings.TrimSpace(tc.Output)
		if got == want {
			passed++
		} else {
			log.Printf("[Eval] 输出不匹配:\n  期望: %q\n  实际: %q", want, got)
		}
	}

	score := float64(passed) / float64(total)
	log.Printf("[Eval] 通过 %d/%d 测试用例 (score=%.2f)", passed, total, score)
	return score
}

// evaluateDefault: 无测试用例时的默认评分
func (e *MCTSEngine) evaluateDefault(node *Node) float64 {
	res := runner.ExecuteCode(node.Thought, "temp_solution.go")
	if res.Err != nil {
		log.Printf("[Eval] 执行失败: %s", res.Stderr)
		return ExpandPenalty
	}

	output := strings.TrimSpace(res.Stdout)
	if output == "" {
		log.Printf("[Eval] 代码运行成功但无输出 (score=0.20)")
		return 0.2
	}
	log.Printf("[Eval] 代码运行成功 (score=0.50)")
	return 0.5
}

// backpropagate: 回溯更新路径上所有节点的统计信息
func (e *MCTSEngine) backpropagate(path []*Node, score float64) {
	for _, p := range path {
		if p == nil {
			continue
		}
		p.Mu.Lock()
		p.Visits++
		p.Value += score
		p.Mu.Unlock()
		atomic.AddInt32(&p.VirtualLoss, -1)
	}
}

// RunConcurrent: 并发 MCTS 搜索入口
func (e *MCTSEngine) RunConcurrent(initialState string, iters int, workers int) *Node {
	root := NewRootNode(initialState)
	var wg sync.WaitGroup

	e.generatedMu.Lock()
	e.generatedSet = make(map[string]bool)
	e.generatedMu.Unlock()

	if len(e.testCases) > 0 {
		log.Printf("MCTS search: %d iters, %d workers, maxDepth=%d, testCases=%d",
			iters, workers, e.maxDepth, len(e.testCases))
	} else {
		log.Printf("MCTS search: %d iters, %d workers, maxDepth=%d (无测试用例，评分信号弱)",
			iters, workers, e.maxDepth)
	}

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for j := 0; j < iters/workers; j++ {
				// 1. Selection（含深度约束）
				curr := root
				var path []*Node
				depth := 0
				for {
					path = append(path, curr)
					atomic.AddInt32(&curr.VirtualLoss, 1)

					curr.Mu.RLock()
					isLeaf := len(curr.Children) == 0 || depth >= e.maxDepth
					curr.Mu.RUnlock()

					if isLeaf {
						break
					}

					next := e.selectNode(curr)
					if next == nil {
						break
					}
					curr = next
					depth++
				}

				// 2. Expand（仅对无子节点的节点扩展）
				curr.Mu.RLock()
				shouldExpand := len(curr.Children) == 0
				curr.Mu.RUnlock()

				if shouldExpand {
					if err := e.expand(curr); err != nil {
						log.Printf("[Worker %d] Expand: %v", workerID, err)
						for _, p := range path {
							atomic.AddInt32(&p.VirtualLoss, -1)
						}
						continue
					}
				}

				// 3. Evaluate — 评估选定节点自身的代码（而非它的子节点）
				//    根节点不含可执行代码，特殊处理为其第一个子节点
				evalTarget := curr
				if curr == root {
					curr.Mu.RLock()
					if len(curr.Children) > 0 {
						evalTarget = curr.Children[0]
					}
					curr.Mu.RUnlock()
				}

				if evalTarget == nil {
					continue
				}

				score := e.evaluate(evalTarget)

				// 4. Backpropagate（单次回传）
				if evalTarget != curr {
					atomic.AddInt32(&evalTarget.VirtualLoss, 1)
					path = append(path, evalTarget)
				}
				e.backpropagate(path, score)

				log.Printf("[Worker %d] Iter %d done (depth=%d, score=%.2f)", workerID, j, depth, score)
			}
		}(i)
	}
	wg.Wait()

	log.Printf("Search complete. Unique solutions generated: %d", func() int {
		e.generatedMu.Lock()
		defer e.generatedMu.Unlock()
		return len(e.generatedSet)
	}())
	root.PrintTree(0)
	return root
}

// cleanJSON: 健壮地提取最外层 JSON（处理嵌套花括号）
func (e *MCTSEngine) cleanJSON(input string) string {
	start := strings.Index(input, "{")
	if start == -1 {
		return input
	}

	depth := 0
	end := -1
	for i := start; i < len(input); i++ {
		switch input[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				end = i
				break
			}
		}
	}
	if end == -1 {
		return input
	}
	return input[start : end+1]
}
