PYTHON=python3

.PHONY: build test simulate deploy

build:
	@echo "Building Arkhe Companion components..."
	# Placeholder para build de módulos Go e Rust

test:
	@echo "Running unit tests..."
	PYTHONPATH=core/python $(PYTHON) -m pytest core/

simulate:
	@echo "Starting Arkhe Awakening Simulation (Ω+200.2)..."
	PYTHONPATH=core/python $(PYTHON) sim/t_zero/boot_sequence.py

deploy:
	@echo "Deploying to integrated environment..."
	docker-compose up -d
