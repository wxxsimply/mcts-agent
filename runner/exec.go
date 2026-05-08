package runner

import (
	"bytes"
	"context"
	"fmt"
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

// DefaultExecTimeout 默认代码执行超时
var DefaultExecTimeout = 2 * time.Second

// ExecuteCode 运行一段 Go 代码，带超时保护
func ExecuteCode(code string, filename string) ExecutionResult {
	return ExecuteCodeWithStdin(code, filename, "")
}

// ExecuteCodeWithStdin 运行 Go 代码并传入标准输入，带超时保护
func ExecuteCodeWithStdin(code string, filename string, stdin string) ExecutionResult {
	return ExecuteCodeWithConfig(code, filename, stdin, DefaultExecTimeout)
}

// ExecuteCodeWithConfig 使用完整配置运行 Go 代码，stdout/stderr 分离
func ExecuteCodeWithConfig(code string, filename string, stdin string, timeout time.Duration) ExecutionResult {
	if filename == "" {
		filename = "temp_solution"
	}

	// 使用 os.CreateTemp 创建唯一临时文件，避免并发 worker 之间的竞态
	f, err := os.CreateTemp(os.TempDir(), fmt.Sprintf("%s_*.go", filename))
	if err != nil {
		return ExecutionResult{Stderr: "create temp file failed", Err: err}
	}
	filePath := f.Name()
	if _, err := f.Write([]byte(code)); err != nil {
		f.Close()
		os.Remove(filePath)
		return ExecutionResult{Stderr: "write file failed", Err: err}
	}
	f.Close()
	defer os.Remove(filePath)

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "go", "run", filePath)

	// 传入标准输入
	if stdin != "" {
		cmd.Stdin = bytes.NewBufferString(stdin)
	}

	// 分离 stdout 和 stderr，避免测试用例比较时因 stderr 内容导致误判
	var stdoutBuf, stderrBuf bytes.Buffer
	cmd.Stdout = &stdoutBuf
	cmd.Stderr = &stderrBuf

	err = cmd.Run()

	result := ExecutionResult{
		Stdout: stdoutBuf.String(),
		Stderr: stderrBuf.String(),
		Err:    err,
	}

	if err != nil {
		if ctx.Err() != nil {
			result.Stderr = fmt.Sprintf("execution timed out (%v)\n%s", timeout, stderrBuf.String())
		}
	}

	return result
}
