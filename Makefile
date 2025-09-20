# Makefile for calibration library development

.PHONY: build test clean format lint coverage help

BUILD_DIR := build
BUILD_TYPE := Release

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

build: ## Build the project
	cmake -S . -B $(BUILD_DIR) -G Ninja -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	cmake --build $(BUILD_DIR) -j4

test: build ## Build and run tests
	cd $(BUILD_DIR) && ctest --output-on-failure

clean: ## Clean build directory
	rm -rf $(BUILD_DIR)

format: ## Format source code with clang-format
	find src include examples -name "*.cpp" -o -name "*.h" | xargs clang-format -i

format-check: ## Check code formatting
	find src include examples -name "*.cpp" -o -name "*.h" | xargs clang-format --dry-run --Werror

lint: build ## Run static analysis
	@echo "Running clang-tidy..."
	clang-tidy -p $(BUILD_DIR) src/*.cpp --config-file=.clang-tidy --header-filter="^$(shell pwd)/(include|src)/.*"
	@echo "Running cppcheck..."
	cppcheck --enable=all --std=c++20 --suppress=missingIncludeSystem --suppress=unusedFunction \
		--suppress=unmatchedSuppression --suppress=unreadVariable \
		-I include src/ include/calib/

coverage: ## Generate test coverage report
	cmake -S . -B $(BUILD_DIR) -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="--coverage -g -O0"
	cmake --build $(BUILD_DIR) -j4
	cd $(BUILD_DIR) && ctest --output-on-failure
	lcov --capture --directory $(BUILD_DIR) --output-file coverage.info
	lcov --remove coverage.info '/usr/*' '*/test/*' '*/build/_deps/*' --output-file coverage.info
	lcov --list coverage.info
	genhtml coverage.info --output-directory coverage_report
	@echo "Coverage report generated in coverage_report/index.html"

install-deps-ubuntu: ## Install dependencies on Ubuntu
	sudo apt update
	sudo apt install -y cmake ninja-build libeigen3-dev libceres-dev nlohmann-json3-dev \
		libgtest-dev libgmock-dev libboost-dev libcli11-dev clang-tidy cppcheck clang-format

install-hooks: ## Install pre-commit hooks
	pip install pre-commit
	pre-commit install

all: clean build test lint ## Clean, build, test, and lint

release: BUILD_TYPE=Release
release: clean build test ## Build release version

debug: BUILD_TYPE=Debug  
debug: clean build test ## Build debug version
