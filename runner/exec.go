package runner

import (
	"context"
	"os"
	"os/exec"
	"time"
)

// ExecutionResult 存储运行结果
type ExecutionResult struct {
	Stdout string
	Stderr string
	Err    error
}

// ExecuteCode 运行一段 Go 代码，带 5 秒超时保护
func ExecuteCode(code string, filename string) ExecutionResult {
	if filename == "" {
		filename = "temp_solution.go"
	}

	err := os.WriteFile(filename, []byte(code), 0644)
	if err != nil {
		return ExecutionResult{Stderr: "写入文件失败", Err: err}
	}
	defer os.Remove(filename)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "go", "run", filename)

	// CombinedOutput 获取所有流并返回字节数组
	output, err := cmd.CombinedOutput()

	return ExecutionResult{
		Stdout: string(output),
		Stderr: "",
		Err:    err,
	}
}
