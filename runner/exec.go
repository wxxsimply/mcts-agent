package runner

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
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
	return ExecuteCodeWithStdin(code, filename, "")
}

// ExecuteCodeWithStdin 运行 Go 代码并传入标准输入，带 5 秒超时保护
func ExecuteCodeWithStdin(code string, filename string, stdin string) ExecutionResult {
	if filename == "" {
		filename = "temp_solution.go"
	}

	filePath := filepath.Join(os.TempDir(), filename)

	if err := os.WriteFile(filePath, []byte(code), 0644); err != nil {
		return ExecutionResult{Stderr: "write file failed", Err: err}
	}
	defer os.Remove(filePath)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "go", "run", filePath)

	// 传入标准输入
	if stdin != "" {
		cmd.Stdin = bytes.NewBufferString(stdin)
	}

	output, err := cmd.CombinedOutput()

	result := ExecutionResult{
		Stdout: string(output),
		Stderr: "",
		Err:    err,
	}

	if err != nil {
		if ctx.Err() != nil {
			result.Stderr = "execution timed out (5s)"
		} else {
			result.Stderr = string(output)
		}
	}

	return result
}
