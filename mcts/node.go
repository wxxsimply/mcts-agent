package mcts

import (
	"fmt"
	"strings"
	"sync"
)

// Node 代表大模型推理过程中的一个状态节点
type Node struct {
	Thought  string  // 当前步骤的文本描述
	TaskType string  // "math" 或 "code"
	Parent   *Node   // 父节点指针
	Children []*Node // 子节点切片

	// MCTS 统计数据 (使用原子操作或锁保护)
	Visits      float64
	Value       float64
	VirtualLoss int32 // 正在进行的并发探索数 (用于 UCB 惩罚)

	Mu sync.RWMutex // 读写锁：保护 Children 的并发读写
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

// GetPath 溯源：从根到当前节点的完整思维路径（用于生成 Prompt）
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
	fmt.Printf("%s|-- [%.1f/%.0f] %s\n", indent, n.Value, n.Visits, n.Thought)
	for _, child := range n.Children {
		child.PrintTree(depth + 1)
	}
}

// node.go
func (n *Node) GetBestCode() string {
	n.Mu.RLock()
	defer n.Mu.RUnlock()

	// 寻找 Value 最高的子节点
	var bestChild *Node
	maxVal := -1.0
	for _, child := range n.Children {
		if child != nil && child.Value > maxVal { // 增加 child != nil 检查
			maxVal = child.Value
			bestChild = child
		}
	}

	// 只有找到了有效的 bestChild 才递归
	if bestChild != nil {
		return bestChild.GetBestCode()
	}
	return n.Thought
}
