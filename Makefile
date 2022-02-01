################################################################
# GLOBALS													   #
################################################################

PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME := hearai
PYTHON ?= python3
VENV = $(PROJECT_DIR)/.venv
PIP = $(VENV)/bin/pip
VIRTUALENV = $(PYTHON) -m venv
SHELL=/bin/bash

################################################################
# VIRTUAL ENVIRONMENT AND DEPENDENCIES						   #
################################################################

.PHONY: venv
## create virtual environment
venv: ./.venv/.requirements
        
.venv:
		$(VIRTUALENV) $(VENV)
		$(PIP) install -U pip setuptools wheel

.venv/.requirements: .venv
		$(PIP) install -r $(PROJECT_DIR)/requirements.txt -r $(PROJECT_DIR)/requirements-dev.txt
		touch $(VENV)/.requirements

.PHONY: venv-clean
## clean virtual environment
venv-clean:
		rm -rf $(VENV)

################################################################
# style														   #
################################################################

.PHONY: format-check
## check the code style
format-check: .venv/.requirements
		$(VENV)/bin/black --check $(PROJECT_DIR)/

.PHONY: format-apply
## reformat the code style
format-apply: venv
		$(VENV)/bin/black $(PROJECT_DIR)/

.PHONY: lint
lint: venv
		@PYTHONPATH=$(PYTHONPATH):$(PROJECT_DIR) $(VENV)/bin/pylint --rcfile=setup.cfg train.py $(PROJECT_DIR)/models/