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
	"time"
)

// SearchEvent 搜索过程中发出的实时事件
type SearchEvent struct {
	Type      string      `json:"type"`
	Timestamp time.Time   `json:"timestamp"`
	Data      interface{} `json:"data"`
}

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

	// 防止多个 worker 同时展开同一节点
	expandingMu sync.Mutex
	expanding   map[*Node]bool

	// 事件回调（前端实时更新用）
	eventCallback func(SearchEvent)
}

func NewMCTSEngine(client *llm.LLMClient) *MCTSEngine {
	return &MCTSEngine{
		llmClient: client,
		exploreC:  DefaultExploreC,
		maxDepth:  DefaultMaxDepth,
		expanding: make(map[*Node]bool),
	}
}

func (e *MCTSEngine) WithTestCases(tcs []TestCase) *MCTSEngine {
	e.testCases = tcs
	return e
}

func (e *MCTSEngine) WithExploreC(c float64) *MCTSEngine {
	e.exploreC = c
	return e
}

func (e *MCTSEngine) WithMaxDepth(d int) *MCTSEngine {
	e.maxDepth = d
	return e
}

func (e *MCTSEngine) WithEventCallback(cb func(SearchEvent)) *MCTSEngine {
	e.eventCallback = cb
	return e
}

func (e *MCTSEngine) emitEvent(evt SearchEvent) {
	if e.eventCallback != nil {
		e.eventCallback(evt)
	}
}

// ------------- 核心 MCTS 方法 -------------

func (e *MCTSEngine) selectNode(node *Node) *Node {
	node.Mu.RLock()
	defer node.Mu.RUnlock()

	var bestNode *Node
	maxUCB := math.Inf(-1)

	for _, child := range node.Children {
		child.Mu.RLock()
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
		child.Mu.RUnlock()
	}
	return bestNode
}

func (e *MCTSEngine) buildExpandPrompt(pathThought string, previousCount int) (string, string) {
	sysPrompt := `You are an expert Go algorithm engineer. You MUST return valid, compilable Go code.

CRITICAL RULES:
1. Return ONLY valid JSON: {"actions": ["package main\n...complete Go code..."]}
2. Each action MUST start with "package main" and be a complete, runnable Go program
3. Do NOT return explanations, descriptions, or the user's question text
4. Generate diverse algorithmic approaches when possible
5. Do NOT generate duplicate or near-duplicate code

Return strictly as JSON:
{"actions": ["package main\n\nimport \"fmt\"\n\nfunc main() {\n\t// your code here\n}"]}`

	userPrompt := fmt.Sprintf("Task:\n%s\n\n(Already generated %d unique solutions — please generate new variations.)",
		pathThought, previousCount)
	return sysPrompt, userPrompt
}

func (e *MCTSEngine) expand(node *Node) error {
	// 防止多个 worker 同时展开同一节点：先检查是否已有子节点或正在被展开
	e.expandingMu.Lock()
	if e.expanding[node] {
		e.expandingMu.Unlock()
		return nil
	}
	node.Mu.RLock()
	alreadyExpanded := len(node.Children) > 0
	node.Mu.RUnlock()
	if alreadyExpanded {
		e.expandingMu.Unlock()
		return nil
	}
	e.expanding[node] = true
	e.expandingMu.Unlock()

	defer func() {
		e.expandingMu.Lock()
		delete(e.expanding, node)
		e.expandingMu.Unlock()
	}()

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
		// 过滤非 Go 代码（LLM 有时返回用户提问文本）
		if !strings.Contains(trimmed, "package ") {
			log.Printf("[LLM] 过滤非代码内容 (len=%d): %s...", len(trimmed), truncate(trimmed, 40))
			continue
		}
		if !e.generatedSet[trimmed] {
			e.generatedSet[trimmed] = true
			node.AddChild(trimmed)
			added++
		}
	}
	e.generatedMu.Unlock()

	if added == 0 {
		node.Mu.Lock()
		node.DeadEnd = true
		node.Mu.Unlock()
		log.Printf("[LLM] LLM 未返回有效代码，跳过该节点")
		return fmt.Errorf("no valid Go code")
	}
	return nil
}

func (e *MCTSEngine) evaluate(node *Node) float64 {
	if len(e.testCases) > 0 {
		return e.evaluateWithCases(node)
	}
	return e.evaluateDefault(node)
}

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

func (e *MCTSEngine) evaluateDefault(node *Node) float64 {
	res := runner.ExecuteCode(node.Thought, "temp_solution.go")
	if res.Err != nil {
		log.Printf("[Eval] 执行失败: %s", res.Stderr)
		return ExpandPenalty
	}

	output := strings.TrimSpace(res.Stdout)
	if output == "" {
		return 0.2
	}

	// 请求 LLM 对代码质量做细粒度评分（0.0 ~ 1.0），增加树节点分数的区分度
	score := e.evaluateWithLLM(node, output)
	// 融合执行成功基础分和 LLM 质量分，保证最低 0.3
	finalScore := math.Max(0.3, 0.3*0.5+0.7*score)
	log.Printf("[Eval] LLM评分=%.2f, 融合后=%.2f", score, finalScore)
	return finalScore
}

// evaluateWithLLM 用 LLM 对代码质量做 0.0~1.0 评分
func (e *MCTSEngine) evaluateWithLLM(node *Node, stdout string) float64 {
	// 获取任务描述（从根节点追溯）
	root := node
	for root.Parent != nil {
		root = root.Parent
	}

	scorePrompt := `You are a strict code quality judge. Rate this Go program 0.0-1.0:

Criteria (weighted):
- Correctness (50%): Does it solve the task correctly?
- Quality (30%): Clean code, proper error handling, good structure
- Output (20%): Clear and well-formatted output

Rating guide:
0.0-0.2: Wrong or misleading output
0.2-0.4: Partially correct, missing key parts
0.4-0.6: Basic correct solution, minimal effort
0.6-0.8: Good solution, well-structured
0.8-1.0: Excellent, handles edge cases, efficient

Return ONLY valid JSON: {"score": 0.0-1.0, "reason": "brief 1-sentence reason"}`

	userPrompt := fmt.Sprintf("Task: %s\n\nCode:\n%s\n\nOutput:\n%s", root.Thought, node.Thought, stdout)

	resp, err := e.llmClient.Ask(context.Background(), scorePrompt, userPrompt)
	if err != nil {
		log.Printf("[Eval] LLM评分调用失败: %v", err)
		return 0.5
	}

	var scoreResp struct {
		Score  float64 `json:"score"`
		Reason string  `json:"reason"`
	}
	if err := json.Unmarshal([]byte(e.cleanJSON(resp)), &scoreResp); err != nil {
		log.Printf("[Eval] LLM评分JSON解析失败: %v\nraw: %s", err, resp)
		return 0.5
	}

	// 限制评分范围
	scoreResp.Score = math.Max(0.0, math.Min(1.0, scoreResp.Score))
	log.Printf("[Eval] LLM评分详情: %.2f — %s", scoreResp.Score, scoreResp.Reason)
	return scoreResp.Score
}

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
	var completedIters int32

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

	e.emitEvent(SearchEvent{Type: "start", Timestamp: time.Now(), Data: map[string]interface{}{
		"task": initialState, "iters": iters, "workers": workers, "maxDepth": e.maxDepth,
	}})

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			// 将总迭代次数均匀分配给每个 worker，余数从前向后补足
			base := iters / workers
			remainder := iters % workers
			workerIters := base
			if workerID < remainder {
				workerIters++
			}
			for j := 0; j < workerIters; j++ {
				curr := root
				var path []*Node
				depth := 0

				// 1. Selection
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

				// 2. Expand（expand 内部有双重检查防止竞态；已经展开的节点会快速返回）
				shouldExpand := func() bool {
					curr.Mu.RLock()
					defer curr.Mu.RUnlock()
					return len(curr.Children) == 0 && !curr.DeadEnd
				}()
				expanded := false
				if shouldExpand {
					if err := e.expand(curr); err != nil {
						log.Printf("[Worker %d] 展开失败: %v", workerID, err)
						for _, p := range path {
							atomic.AddInt32(&p.VirtualLoss, -1)
						}
						continue
					}
					expanded = true
				}
				if expanded {
					e.emitEvent(SearchEvent{Type: "expand", Timestamp: time.Now(), Data: map[string]interface{}{
						"worker": workerID, "thought": truncate(curr.Thought, 80),
						"children": len(curr.Children),
					}})
					// 每次展开后发送完整树供前端渲染
					e.emitEvent(SearchEvent{Type: "tree", Timestamp: time.Now(), Data: root.ToInfo()})
				}

				// 如果当前节点无法展开且无子节点（DeadEnd），跳过整轮
				curr.Mu.RLock()
				noChildren := len(curr.Children) == 0
				curr.Mu.RUnlock()
				if noChildren {
					for _, p := range path {
						atomic.AddInt32(&p.VirtualLoss, -1)
					}
					continue
				}

				// 3. Evaluate — 评估新生成的子节点（代码），而非父节点（任务描述）
				evalTarget := curr
				curr.Mu.RLock()
				if len(curr.Children) > 0 {
					evalTarget = curr.Children[0]
				}
				curr.Mu.RUnlock()

				score := e.evaluate(evalTarget)

				e.emitEvent(SearchEvent{Type: "evaluate", Timestamp: time.Now(), Data: map[string]interface{}{
					"worker": workerID, "thought": truncate(evalTarget.Thought, 80), "score": score,
				}})

				// 4. Backpropagate
				if evalTarget != curr {
					atomic.AddInt32(&evalTarget.VirtualLoss, 1)
					path = append(path, evalTarget)
				}
				e.backpropagate(path, score)

				n := atomic.AddInt32(&completedIters, 1)
				log.Printf("[Worker %d] Iter %d done (depth=%d, score=%.2f) [%d/%d]", workerID, j, depth, score, n, iters)
				if n%3 == 0 {
					e.emitEvent(SearchEvent{Type: "log", Data: map[string]string{
						"message": fmt.Sprintf("搜索进度: %d/%d 迭代完成", n, iters),
					}})
				}
			}
		}(i)
	}
	wg.Wait()

	count := func() int {
		e.generatedMu.Lock()
		defer e.generatedMu.Unlock()
		return len(e.generatedSet)
	}()

	log.Printf("Search complete. Unique solutions generated: %d", count)

	e.emitEvent(SearchEvent{Type: "complete", Timestamp: time.Now(), Data: map[string]interface{}{
		"solutions": count, "tree": root.ToInfo(), "bestCode": root.GetBestCode(),
	}})

	root.PrintTree(0)
	return root
}

// SolutionCount 返回已生成的唯一方案数量
func (e *MCTSEngine) SolutionCount() int {
	e.generatedMu.Lock()
	defer e.generatedMu.Unlock()
	return len(e.generatedSet)
}

func (e *MCTSEngine) cleanJSON(input string) string {
	// 优先匹配 {…} 对象，其次匹配 […] 数组
	start := strings.Index(input, "{")
	useBrace := true
	if start == -1 {
		start = strings.Index(input, "[")
		useBrace = false
	}
	if start == -1 {
		return input
	}

	openCh, closeCh := byte('{'), byte('}')
	if !useBrace {
		openCh, closeCh = '[', ']'
	}

	depth := 0
	end := -1
	inString := false
	escaped := false
	for i := start; i < len(input); i++ {
		ch := input[i]
		if escaped {
			escaped = false
			continue
		}
		if ch == '\\' {
			escaped = true
			continue
		}
		if ch == '"' {
			inString = !inString
			continue
		}
		if inString {
			continue
		}
		if ch == openCh {
			depth++
		} else if ch == closeCh {
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
	extracted := input[start : end+1]
	// 如果 LLM 返回的是裸数组，包装成 {"actions": [...]}
	if !useBrace {
		extracted = `{"actions": ` + extracted + `}`
	}
	return sanitizeJSON(extracted)
}

// sanitizeJSON 转义 JSON 字符串中未转义的控制字符，
// 因为 LLM 经常在 JSON string value 中直接使用字面制表符/换行符。
func sanitizeJSON(raw string) string {
	var result strings.Builder
	inString := false
	escaped := false
	for i := 0; i < len(raw); i++ {
		ch := raw[i]
		if escaped {
			escaped = false
			result.WriteByte(ch)
			continue
		}
		if ch == '\\' && inString {
			escaped = true
			result.WriteByte(ch)
			continue
		}
		if ch == '"' {
			inString = !inString
			result.WriteByte(ch)
			continue
		}
		if inString && (ch == '\t' || ch == '\n' || ch == '\r') {
			switch ch {
			case '\t':
				result.WriteString("\\t")
			case '\n':
				result.WriteString("\\n")
			case '\r':
				result.WriteString("\\r")
			}
			continue
		}
		result.WriteByte(ch)
	}
	return result.String()
}
