// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"syscall"
	"time"
)

// sshOptions keep every remote session non-interactive and detect a dead
// peer within about a minute instead of hanging on a half-open connection.
var sshOptions = []string{
	"-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
	"-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
}

func RemoteArgv(host string, argv []string) []string {
	result := []string{"ssh"}
	result = append(result, sshOptions...)
	result = append(result, host, shellJoin(argv))
	return result
}

// commandRunner executes one command to completion and reports its combined
// output and exit status. A non-nil error means the command could not be run
// or was cut short; a non-zero status means it ran and failed.
type commandRunner func(ctx context.Context, timeout time.Duration, directory string, environment []string, argv []string) ([]byte, int, error)

// runCommand is the single seam through which every local and remote
// command passes, so tests can script command outcomes.
var runCommand commandRunner = execCommand

func execCommand(ctx context.Context, timeout time.Duration, directory string, environment []string, argv []string) ([]byte, int, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	command := exec.CommandContext(ctx, argv[0], argv[1:]...)
	command.Dir = directory
	if environment != nil {
		command.Env = environment
	}
	output, err := command.CombinedOutput()
	if ctx.Err() == context.DeadlineExceeded {
		return output, -1, fmt.Errorf("command timed out after %s", timeout)
	}
	if ctx.Err() != nil {
		return output, -1, ctx.Err()
	}
	if err == nil {
		return output, 0, nil
	}
	if exit, ok := err.(*exec.ExitError); ok {
		return output, exit.ExitCode(), nil
	}
	return output, -1, err
}

func remoteRun(ctx context.Context, host string, argv []string, timeout time.Duration) ([]byte, int, error) {
	return runCommand(ctx, timeout, "", nil, RemoteArgv(host, argv))
}

// streamStarter starts a long-running command whose output goes straight to
// the terminal and returns a wait function.
type streamStarter func(ctx context.Context, argv []string) (wait func() error, err error)

var startStream streamStarter = execStream

func execStream(ctx context.Context, argv []string) (func() error, error) {
	command := exec.CommandContext(ctx, argv[0], argv[1:]...)
	command.Stdout, command.Stderr = os.Stdout, os.Stderr
	command.Cancel = func() error { return command.Process.Signal(syscall.SIGTERM) }
	command.WaitDelay = 5 * time.Second
	if err := command.Start(); err != nil {
		return nil, err
	}
	return command.Wait, nil
}

// runVisible runs a command attached to the terminal; an interrupt reaches
// the child as SIGINT before the launcher gives up on it.
func runVisible(ctx context.Context, argv []string, environment []string) error {
	command := exec.CommandContext(ctx, argv[0], argv[1:]...)
	command.Stdin, command.Stdout, command.Stderr = os.Stdin, os.Stdout, os.Stderr
	command.Cancel = func() error { return command.Process.Signal(os.Interrupt) }
	command.WaitDelay = 10 * time.Second
	if environment != nil {
		command.Env = environment
	}
	return command.Run()
}

func lastOutputLine(output []byte, fallback string) string {
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	if len(lines) == 0 || lines[0] == "" {
		return fallback
	}
	return lines[len(lines)-1]
}
