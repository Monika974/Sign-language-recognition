.PHONY: install test models manifest extract train compare demo
install:
	pip install -r requirements.txt && pip install -e .
test:
	pytest -q
models:        # one-time download of the MediaPipe landmarker files
	python -m signlang.download_models
extract:       # videos -> cached landmarks
	python -m signlang.extract_landmarks --config config.yaml
train:
	python -m signlang.train --model gru
compare:
	python -m signlang.train --model all
demo:
	python -m signlang.realtime --model-dir outputs/gru
