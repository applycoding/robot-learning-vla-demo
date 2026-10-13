.PHONY: test benchmark demo

test:
	python -m pytest

benchmark:
	python -m simreach benchmark --out results

demo: benchmark
