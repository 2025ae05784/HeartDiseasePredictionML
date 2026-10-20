install:
	python -m pip install -r requirements.txt

train:
	python -m src.train

evaluate:
	python -m src.evaluate

test:
	python -m pytest -q

run:
	uvicorn app.main:app --reload
