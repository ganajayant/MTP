.PHONY: activate run-script run

activate:
	@echo "Activating virtual environment..."
	. venv/bin/activate

run-script:
	@echo "Running run.sh..."
	bash run.sh

run:
	@echo "Running main.py..."
	uv run main.py

