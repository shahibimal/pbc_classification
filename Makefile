# Makefile for downloading pbc classification dataset

# Default target: Help Message
help:
	@echo "Usage: make <target>"
	@echo "Targets:"
	@echo "dataset    : Download the pbc dataset"

# Download data
dataset:
	@echo "Downloading the pbc dataset..."
	@python scripts/download_datasets.py

# Run the main script
run:
	@python main.py
