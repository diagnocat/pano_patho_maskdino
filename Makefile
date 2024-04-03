GPUS ?= all

export GPUS

COMPOSE := docker-compose -p "${USER}" -f docker-compose.yaml

main-jupyter:
	$(COMPOSE) up --build

clean:
	$(COMPOSE) rm -sf

logs:
	$(COMPOSE) logs -t -f --tail=100

sync-submodules:
	git submodule update --init --checkout --recursive
