.PHONY: all build build-arm64 test clean run validate validate-consolidation benchmark install install-consolidation check-go deps resolve-deps

BINARY_NAME=sekha-knowledge-store
CONSOLIDATION_BINARY=sekha-consolidation
VALIDATE_NAME=sekha-validate
VALIDATE_CONSOLIDATION_NAME=sekha-validate-consolidation
NODE1_HOST=192.168.8.213
NODE1_USER=admin

all: check-go build

check-go:
	@which go > /dev/null 2>&1 || (echo "ERROR: 'go' is not installed or not in PATH. Ensure Go 1.22+ is installed and in your PATH." && exit 1)

resolve-deps: check-go
	@go mod tidy
	@go mod download

deps: resolve-deps

build: check-go resolve-deps
	@mkdir -p bin
	go build -ldflags="-s -w" -o bin/$(BINARY_NAME) ./cmd/server
	go build -ldflags="-s -w" -o bin/$(CONSOLIDATION_BINARY) ./cmd/consolidation
	go build -ldflags="-s -w" -o bin/$(VALIDATE_NAME) ./cmd/validate
	go build -ldflags="-s -w" -o bin/$(VALIDATE_CONSOLIDATION_NAME) ./cmd/validate-consolidation
	@echo "Build complete: binaries in bin/"

build-arm64: check-go resolve-deps
	@mkdir -p bin
	CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -ldflags="-s -w" -o bin/$(BINARY_NAME)-linux-arm64 ./cmd/server
	CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -ldflags="-s -w" -o bin/$(CONSOLIDATION_BINARY)-linux-arm64 ./cmd/consolidation
	CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -ldflags="-s -w" -o bin/$(VALIDATE_NAME)-linux-arm64 ./cmd/validate
	CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -ldflags="-s -w" -o bin/$(VALIDATE_CONSOLIDATION_NAME)-linux-arm64 ./cmd/validate-consolidation
	@echo "Cross-compiled ARM64 binaries in bin/"

install: build
	sudo systemctl stop sekha-knowledge-store.service 2>/dev/null || true
	sudo systemctl stop sekha-consolidation.service 2>/dev/null || true
	sudo cp bin/$(BINARY_NAME) /usr/local/bin/
	sudo cp bin/$(CONSOLIDATION_BINARY) /usr/local/bin/
	sudo cp bin/$(VALIDATE_NAME) /usr/local/bin/
	sudo cp bin/$(VALIDATE_CONSOLIDATION_NAME) /usr/local/bin/
	sudo cp systemd/sekha-knowledge-store.service /etc/systemd/system/
	sudo cp systemd/sekha-consolidation.service /etc/systemd/system/
	sudo mkdir -p /var/lib/sekha
	sudo chown -R $(NODE1_USER):$(NODE1_USER) /var/lib/sekha 2>/dev/null || true
	sudo systemctl daemon-reload
	sudo systemctl enable sekha-knowledge-store.service sekha-consolidation.service
	sudo systemctl restart sekha-knowledge-store.service sekha-consolidation.service
	@echo "Services updated and restarted: sekha-knowledge-store.service and sekha-consolidation.service"

install-consolidation: build
	sudo systemctl stop sekha-consolidation.service 2>/dev/null || true
	sudo cp bin/$(CONSOLIDATION_BINARY) /usr/local/bin/
	sudo cp bin/$(VALIDATE_CONSOLIDATION_NAME) /usr/local/bin/
	sudo cp systemd/sekha-consolidation.service /etc/systemd/system/
	sudo systemctl daemon-reload
	sudo systemctl enable sekha-consolidation.service
	sudo systemctl restart sekha-consolidation.service
	@echo "Consolidation daemon updated and restarted: sekha-consolidation.service"

test: check-go resolve-deps
	go test -v ./...

validate: check-go build
	./bin/$(VALIDATE_NAME) -nodes 10000 -edges 25000 -queries 100

validate-consolidation: check-go build
	./bin/$(VALIDATE_CONSOLIDATION_NAME) -events 1000 -db test_consolidation_benchmark.db

benchmark: validate validate-consolidation

run: check-go resolve-deps
	go run ./cmd/server -port 8084 -db ./data/knowledge.db

clean:
	rm -rf bin/ *.db *.db-wal *.db-shm test_benchmark.db* test_consolidation_benchmark.db* test_*.db*
