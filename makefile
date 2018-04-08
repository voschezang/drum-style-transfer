LOG_DIR := /tmp/pattern-recognition_ml_models

start:
	jupyter notebook src/

default:
	python3 src/main.py

logs:
	rm -Rf $(LOG_DIR)/*
	tensorboard --logdir=$(LOG_DIR)

clear:
	rm -r $(LOG_DIR)/*

ls:
	ls $(LOG_DIR)/

deps:
	pip3 install -r requirements.txt
	pip2 install -r requirements-python2.txt

deps2:
	pip install -r requirements.txt
	pip2 install -r requirements-python2.txt

predict:
	python3 src/main.py $(book)

clean:
	find . -name \*.pyc -delete
