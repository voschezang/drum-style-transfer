LOG_DIR := /tmp/pattern-recognition_ml_models

.PHONY: logs test

start:
	jupyter notebook src/

default:
	python3 src/main.py

test_midi:
	python3 src/test_midi2.py

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

mk-env:
	conda env create --file environment.txt

activate:
	source activate pattern_recognition_env

deps:
	conda install theano pygpu
	pip3 install -r requirements.txt
	pip2 install -r requirements-python2.txt

deps2:
	pip install -r requirements.txt
	pip2 install -r requirements-python2.txt

dialyzer:
	mypy src/test_midi.py

predict:
	python3 src/main.py $(book)

clean:
	find . -name \*.pyc -delete
