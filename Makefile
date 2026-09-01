SHELL := /bin/sh

GO ?= go
PREFIX ?= $(HOME)/.local
BINDIR ?= $(PREFIX)/bin
DESTDIR ?=
VERSION ?= $(shell git describe --always --dirty 2>/dev/null || printf dev)

BINARY := bin/lil
PACKAGE := ./cmd/lil
GO_SOURCES := $(shell find cmd configs internal -type f -name '*.go' ! -name '*_test.go')
GO_FILES := $(shell find . -type f -name '*.go' -not -path './bin/*')
CONFIG_SOURCES := $(shell find configs -type f -name '*.yaml')
LDFLAGS := -s -w -X main.version=$(VERSION)

.PHONY: all build install fmt fmt-check test test-race vet check clean

all: build

build: $(BINARY)

$(BINARY): $(GO_SOURCES) $(CONFIG_SOURCES) go.mod go.sum
	@mkdir -p $(@D)
	CGO_ENABLED=0 $(GO) build -trimpath -ldflags '$(LDFLAGS)' -o $@ $(PACKAGE)

install: build
	install -d $(DESTDIR)$(BINDIR)
	install -m 0755 $(BINARY) $(DESTDIR)$(BINDIR)/lil

fmt:
	gofmt -w $(GO_FILES)

fmt-check:
	@test -z "$$(gofmt -l $(GO_FILES))"

test:
	$(GO) test -count=1 ./...

test-race:
	$(GO) test -race -count=1 ./...

vet:
	$(GO) vet ./...

check: fmt-check test vet

clean:
	rm -f $(BINARY)
