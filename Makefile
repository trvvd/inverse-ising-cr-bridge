.PHONY: setup A B bridge clean

setup:
	python -m pip install -r requirements.txt

A:
	python -u src/bridge/inverse_ising.py

B:
	python -u src/bridge/consumer_resource.py

bridge: A B

clean:
	rm -rf results/ cr_results/
