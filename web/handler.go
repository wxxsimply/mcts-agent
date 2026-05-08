package web

import (
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"mcts-agent/llm"
	"mcts-agent/mcts"
)

//go:embed static
var staticFiles embed.FS

// SavedResult 保存到磁盘的搜索结果
type SavedResult struct {
	Task      string         `json:"task"`
	Timestamp time.Time      `json:"timestamp"`
	Tree      *mcts.NodeInfo `json:"tree"`
	BestCode  string         `json:"bestCode"`
	Solutions int            `json:"solutions"`
	Iters     int            `json:"iters"`
	Workers   int            `json:"workers"`
	MaxDepth  int            `json:"maxDepth"`
}

// SSE 客户端管理
type sseClient struct {
	ch   chan string
	done chan struct{}
}

type Server struct {
	llmCfg         llm.Config
	mu             sync.RWMutex
	hub            map[*sseClient]bool
	loggedIn       bool
	postSearchHook func()
	lastTreeMu     sync.RWMutex
	lastTree       *mcts.NodeInfo
	resultsDir     string
}

func NewServer(cfg llm.Config) *Server {
	resultsDir := "results"
	os.MkdirAll(resultsDir, 0755)
	return &Server{
		llmCfg:     cfg,
		hub:        make(map[*sseClient]bool),
		resultsDir: resultsDir,
	}
}

func (s *Server) SetPostSearchHook(hook func()) {
	s.postSearchHook = hook
}

func (s *Server) requireAuth(w http.ResponseWriter, r *http.Request) bool {
	s.mu.RLock()
	ok := s.loggedIn
	s.mu.RUnlock()
	if !ok {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return false
	}
	return true
}

func (s *Server) broadcastSSE(event string, data interface{}) {
	// 统一打包成 SSE 帧，所有事件一致地使用 { "data": ..., "timestamp": ... } 结构
	wrapped := map[string]interface{}{
		"data":      data,
		"timestamp": time.Now().UnixMilli(),
	}
	s.broadcast(fmt.Sprintf("event: %s\ndata: %s\n\n", event, toJSON(wrapped)))
}

func (s *Server) broadcast(data string) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for c := range s.hub {
		select {
		case c.ch <- data:
		default:
		}
	}
}

func (s *Server) addClient() *sseClient {
	c := &sseClient{ch: make(chan string, 64), done: make(chan struct{})}
	s.mu.Lock()
	s.hub[c] = true
	s.mu.Unlock()
	return c
}

func (s *Server) removeClient(c *sseClient) {
	s.mu.Lock()
	delete(s.hub, c)
	s.mu.Unlock()
	close(c.done)
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	subFS, err := fs.Sub(staticFiles, "static")
	if err != nil {
		log.Fatalf("embed static: %v", err)
	}
	mux.Handle("/", http.FileServer(http.FS(subFS)))
	mux.HandleFunc("/api/login", s.handleLogin)
	mux.HandleFunc("/api/logout", s.handleLogout)
	mux.HandleFunc("/api/me", s.handleMe)
	mux.HandleFunc("/api/search", s.handleSearch)
	mux.HandleFunc("/api/events", s.handleSSE)
	mux.HandleFunc("/api/tree", s.handleGetTree)
	mux.HandleFunc("GET /api/results", s.handleListResults)
	mux.HandleFunc("GET /api/results/{name}", s.handleGetResult)

	return mux
}

func (s *Server) handleLogin(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	// 从环境变量读取凭据，缺省 admin/admin
	adminUser := os.Getenv("ADMIN_USER")
	if adminUser == "" {
		adminUser = "admin"
	}
	adminPass := os.Getenv("ADMIN_PASS")
	if adminPass == "" {
		adminPass = "admin"
	}
	var req struct{ Username, Password string }
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Username != adminUser || req.Password != adminPass {
		http.Error(w, "invalid credentials", http.StatusUnauthorized)
		return
	}
	s.mu.Lock()
	s.loggedIn = true
	s.mu.Unlock()
	json.NewEncoder(w).Encode(map[string]bool{"ok": true})
}

func (s *Server) handleLogout(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	s.loggedIn = false
	s.mu.Unlock()
	json.NewEncoder(w).Encode(map[string]bool{"ok": true})
}

func (s *Server) handleMe(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	ok := s.loggedIn
	s.mu.RUnlock()
	json.NewEncoder(w).Encode(map[string]bool{"loggedIn": ok})
}

func (s *Server) handleSSE(w http.ResponseWriter, r *http.Request) {
	if !s.requireAuth(w, r) {
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	client := s.addClient()
	defer s.removeClient(client)

	fmt.Fprintf(w, "event: connected\ndata: {}\n\n")
	flusher.Flush()

	ctx := r.Context()
	for {
		select {
		case <-ctx.Done():
			return
		case <-client.done:
			return
		case msg := <-client.ch:
			fmt.Fprint(w, msg)
			flusher.Flush()
		}
	}
}

type searchRequest struct {
	Task      string          `json:"task"`
	TestCases []mcts.TestCase `json:"testCases"`
}

func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if !s.requireAuth(w, r) {
		return
	}

	var req searchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if req.Task == "" {
		http.Error(w, "task is required", http.StatusBadRequest)
		return
	}

	if err := json.NewEncoder(w).Encode(map[string]string{"status": "started"}); err != nil {
		log.Printf("[Search] 无法发送初始响应: %v", err)
		return
	}
	go s.runSearch(req)
}

func (s *Server) runSearch(req searchRequest) {
	client := llm.NewLLMClient(s.llmCfg)

	s.broadcastSSE("log", map[string]string{
		"message": fmt.Sprintf("开始 MCTS 搜索（maxDepth=%d, testCases=%d）", 4, len(req.TestCases)),
	})

	engine := mcts.NewMCTSEngine(client).
		WithMaxDepth(4).
		WithExploreC(1.414).
		WithTestCases(req.TestCases).
		WithEventCallback(func(evt mcts.SearchEvent) {
			// 所有引擎事件统一转发，data 字段自然保留
			s.broadcastSSE(evt.Type, evt.Data)
		})

	iters := 20
	workers := 5
	root := engine.RunConcurrent(req.Task, iters, workers)

	// 将最终树保存到服务端，供 /api/tree 查询
	treeInfo := root.ToInfo()
	bestCode := root.GetBestCode()
	s.lastTreeMu.Lock()
	s.lastTree = &treeInfo
	s.lastTreeMu.Unlock()

	// 保存到磁盘
	s.saveResult(SavedResult{
		Task:      req.Task,
		Timestamp: time.Now(),
		Tree:      &treeInfo,
		BestCode:  bestCode,
		Solutions: engine.SolutionCount(),
		Iters:     iters,
		Workers:   workers,
		MaxDepth:  4,
	})

	s.broadcastSSE("log", map[string]string{
		"message": "搜索完成，最佳方案已输出",
	})

	if s.postSearchHook != nil {
		s.postSearchHook()
	}
}

func toJSON(v interface{}) string {
	b, err := json.Marshal(v)
	if err != nil {
		log.Printf("[SSE] toJSON marshal error: %v", err)
		return "{}"
	}
	return string(b)
}

func (s *Server) saveResult(r SavedResult) {
	// 用时间戳 + 任务前缀做文件名
	taskPrefix := strings.Map(func(r rune) rune {
		if r > 0x7f {
			return -1 // 去掉非 ASCII 字符
		}
		return r
	}, r.Task)
	if len(taskPrefix) > 30 {
		taskPrefix = taskPrefix[:30]
	}
	taskPrefix = strings.TrimSpace(strings.Map(func(r rune) rune {
		if strings.ContainsRune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_", r) {
			return r
		}
		return '_'
	}, taskPrefix))

	name := fmt.Sprintf("%s_%s.json", r.Timestamp.Format("20060102_150405"), taskPrefix)
	path := filepath.Join(s.resultsDir, name)
	data, _ := json.MarshalIndent(r, "", "  ")
	if err := os.WriteFile(path, data, 0644); err != nil {
		log.Printf("[Save] 保存结果失败 %s: %v", path, err)
	} else {
		log.Printf("[Save] 结果已保存: %s", path)
	}
}

func (s *Server) handleListResults(w http.ResponseWriter, r *http.Request) {
	if !s.requireAuth(w, r) {
		return
	}
	entries, err := os.ReadDir(s.resultsDir)
	if err != nil {
		json.NewEncoder(w).Encode([]interface{}{})
		return
	}
	type ResultMeta struct {
		File      string    `json:"file"`
		Task      string    `json:"task"`
		Timestamp time.Time `json:"timestamp"`
		Solutions int       `json:"solutions"`
	}
	var list []ResultMeta
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(s.resultsDir, e.Name()))
		if err != nil {
			continue
		}
		var meta struct {
			Task      string    `json:"task"`
			Timestamp time.Time `json:"timestamp"`
			Solutions int       `json:"solutions"`
		}
		json.Unmarshal(data, &meta)
		list = append(list, ResultMeta{
			File:      e.Name(),
			Task:      meta.Task,
			Timestamp: meta.Timestamp,
			Solutions: meta.Solutions,
		})
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(list)
}

func (s *Server) handleGetResult(w http.ResponseWriter, r *http.Request) {
	if !s.requireAuth(w, r) {
		return
	}
	name := r.PathValue("name")
	if name == "" {
		http.Error(w, "missing name", http.StatusBadRequest)
		return
	}
	// 防止路径穿越
	if strings.Contains(name, "..") || strings.Contains(name, "/") || strings.Contains(name, "\\") {
		http.Error(w, "invalid name", http.StatusBadRequest)
		return
	}
	path := filepath.Join(s.resultsDir, name)
	data, err := os.ReadFile(path)
	if err != nil {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(data)
}

func (s *Server) handleGetTree(w http.ResponseWriter, r *http.Request) {
	if !s.requireAuth(w, r) {
		return
	}
	s.lastTreeMu.RLock()
	tree := s.lastTree
	s.lastTreeMu.RUnlock()
	if tree == nil {
		http.Error(w, "no tree available", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(tree)
}
