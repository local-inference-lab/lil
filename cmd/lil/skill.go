// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package main

import (
	_ "embed"
	"fmt"
	"os"

	"github.com/spf13/pflag"
)

//go:embed skill.txt
var modelDefinitionSkill string

func skillCommand(args []string) error {
	fs := pflag.NewFlagSet("skill", pflag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, `Usage: lil skill

Print plaintext instructions for agents adding lil model definitions to stdout.
The guide is embedded in the executable and requires no network or configuration.`)
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("skill takes no positional arguments")
	}
	_, err := fmt.Fprint(os.Stdout, modelDefinitionSkill)
	return err
}
