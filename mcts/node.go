package mcts

import (
	"fmt"
	"math"
	"strings"
	"sync"
)

// Node 代表大模型推理过程中的一个状态节点
type Node struct {
	Thought  string  // 当前步骤的文本描述
	Parent   *Node   // 父节点指针
	Children []*Node // 子节点切片

	// MCTS 统计数据
	Visits      float64
	Value       float64
	VirtualLoss int32 // 正在进行的并发探索数 (用于 UCB 惩罚)

	Mu sync.RWMutex
}

// NewRootNode 初始化思维树的根节点
func NewRootNode(initialState string) *Node {
	return &Node{
		Thought:  initialState,
		Children: make([]*Node, 0),
	}
}

// AddChild 并发安全地添加子节点
func (n *Node) AddChild(thought string) *Node {
	n.Mu.Lock()
	defer n.Mu.Unlock()
	child := &Node{
		Thought: thought,
		Parent:  n,
	}
	n.Children = append(n.Children, child)
	return child
}

// GetPath 溯源：从根到当前节点的完整思维路径
func (n *Node) GetPath() string {
	var path []string
	curr := n
	for curr != nil {
		path = append([]string{curr.Thought}, path...)
		curr = curr.Parent
	}
	return strings.Join(path, "\n -> ")
}

// PrintTree 深度优先打印整个推理树
func (n *Node) PrintTree(depth int) {
	n.Mu.RLock()
	defer n.Mu.RUnlock()

	indent := strings.Repeat("  ", depth)
	avg := 0.0
	if n.Visits > 0 {
		avg = n.Value / n.Visits
	}
	fmt.Printf("%s|-- [avg=%.2f val=%.1f visits=%.0f] %s\n",
		indent, avg, n.Value, n.Visits, truncate(n.Thought, 50))
	for _, child := range n.Children {
		child.PrintTree(depth + 1)
	}
}

// GetBestCode 按平均分（而非累计分）贪心选择最优路径
func (n *Node) GetBestCode() string {
	n.Mu.RLock()
	defer n.Mu.RUnlock()

	var bestChild *Node
	bestAvg := -1.0
	for _, child := range n.Children {
		if child == nil {
			continue
		}
		avg := child.Value / math.Max(1.0, child.Visits)
		if avg > bestAvg {
			bestAvg = avg
			bestChild = child
		}
	}

	if bestChild != nil {
		return bestChild.GetBestCode()
	}
	return n.Thought
}

// truncate 截断字符串用于日志展示
func truncate(s string, maxLen int) string {
	runes := []rune(s)
	if len(runes) <= maxLen {
		return s
	}
	return string(runes[:maxLen]) + "..."
}
