.PHONY: all build build-arm64 test clean run validate benchmark install check-go deps resolve-deps

BINARY_NAME=sekha-knowledge-store
VALIDATE_NAME=sekha-validate
NODE1_HOST=192.168.8.213
NODE1_USER=admin

all: check-go build

check-go:
	@which go > /dev/null 2>&1 || (echo "ERROR: 'go' is not installed or not in PATH. Ensure Go 1.22+ is installed and in your PATH." && exit 1)

resolve-deps: check-go
	@if [ ! -f go.sum ] || ! grep -q "modernc.org/sqlite" go.sum 2>/dev/null; then \
		echo "Resolving Go module dependencies (downloading modernc.org/sqlite pure-Go driver)..."; \
		go mod tidy; \
		go mod download; \
		echo "Dependencies successfully resolved and verified."; \
	fi

deps: resolve-deps

build: check-go resolve-deps
	@mkdir -p bin
	go build -ldflags="-s -w" -o bin/$(BINARY_NAME) ./cmd/server
	go build -ldflags="-s -w" -o bin/$(VALIDATE_NAME) ./cmd/validate
	@echo "Build complete: bin/$(BINARY_NAME) and bin/$(VALIDATE_NAME)"

build-arm64: check-go resolve-deps
	@mkdir -p bin
	CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -ldflags="-s -w" -o bin/$(BINARY_NAME)-linux-arm64 ./cmd/server
	CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -ldflags="-s -w" -o bin/$(VALIDATE_NAME)-linux-arm64 ./cmd/validate
	@echo "Cross-compiled ARM64 binaries in bin/"

install: build
	sudo systemctl stop sekha-knowledge-store.service 2>/dev/null || true
	sudo cp bin/$(BINARY_NAME) /usr/local/bin/
	sudo cp bin/$(VALIDATE_NAME) /usr/local/bin/
	sudo cp systemd/sekha-knowledge-store.service /etc/systemd/system/
	sudo mkdir -p /var/lib/sekha
	sudo chown -R $(NODE1_USER):$(NODE1_USER) /var/lib/sekha 2>/dev/null || true
	sudo systemctl daemon-reload
	sudo systemctl restart sekha-knowledge-store.service
	@echo "Daemon updated and restarted: sekha-knowledge-store.service"

test: check-go resolve-deps
	go test -v ./...

validate: check-go build
	./bin/$(VALIDATE_NAME) -nodes 10000 -edges 25000 -queries 100

benchmark: validate

run: check-go resolve-deps
	go run ./cmd/server -port 8084 -db ./data/knowledge.db

clean:
	rm -rf bin/ *.db *.db-wal *.db-shm test_benchmark.db*
