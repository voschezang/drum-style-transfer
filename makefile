PROJECT := pattern-recognition
LOG_DIR := /tmp/pattern-recognition_ml_models
# SHELL=/bin/bash
# SOURCE=$(shell bash -c ./setenv.sh)

RUN_PYTHON := (cd src && exec | pythonw .py)

.PHONY: logs test


#	  source activate envs/default
# conda install -c conda-forge ggplot

start:
	jupyter notebook src/

# run:
# 	anaconda-project run


activate:
	bash -c "source setenv.sh; env | sed 's/=/:=/' | sed 's/^/export /' > makeenv"
	include makeenv
# bash setenv.sh

deps:
	conda install anaconda-project

test:
	make run-python script=test_import.py
	# make run-python script=test_midi2.py
# pythonw src/test_midi2.py

run-python:
	(cd src && exec | pythonw $(script))


archive:
	anaconda-project archive $(PROJECT).zip

train:
	make clear-logs
	python3 src/main.py

logs:
	open http:localhost:6006
	tensorboard --logdir=$(LOG_DIR)

refresh-logs:
	make clear-logs
	make logs

save-logs:
	rm -rf logs/
	mkdir logs
	cp $(LOG_DIR)/* logs/

clear-logs:
	rm -rf $(LOG_DIR)/*

load-logs:
	cp logs/* $(LOG_DIR)/

ls:
	ls $(LOG_DIR)/

rm-env:
	conda env remove -n pattern_recognition

dialyzer:
	mypy src/test_midi.py

predict:
	python3 src/main.py $(book)

clean:
	find . -name \*.pyc -delete

build-docs:
	mkdocs build --clean

deploy:
	mkdocs gh-deploy
ldropout=0.1
