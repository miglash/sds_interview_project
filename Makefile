install:
	python3 -m pip install --upgrade pip
	pip install -r requirements.txt

test:
	python3 -m pytest -vv --cov=model tests/*.py

format:
	black src/*.py --line-length 80

lint:
	flake8 --max-line-length=80 --extend-ignore=E203,E501

all: install format lint test