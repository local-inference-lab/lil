SHELL := /bin/sh

GO ?= go
PREFIX ?= $(HOME)/.local
BINDIR ?= $(PREFIX)/bin
DESTDIR ?=
VERSION ?= $(shell git describe --always --dirty 2>/dev/null || printf dev)

BINARY := bin/lil
PACKAGE := ./cmd/lil
GO_FILES := $(shell find . -type f -name '*.go' -not -path './bin/*')
LDFLAGS := -s -w -X main.version=$(VERSION)

.PHONY: all build install fmt fmt-check test test-race vet check clean

all: build

build:
	@mkdir -p $(dir $(BINARY))
	CGO_ENABLED=0 $(GO) build -trimpath -ldflags '$(LDFLAGS)' -o $(BINARY) $(PACKAGE)

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
