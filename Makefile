USER_NAME := $(shell whoami)
IMAGE_NAME := bevradar
TAG_NAME := v1.0.0
GPU_ID := 0
CONTAINER_NAME := $(IMAGE_NAME)_container

UID := $(shell id -u)
GID := $(shell id -g)

PATH_TO_NUSCENES := /home/$(USER_NAME)/Datasets/nuscenes

define run_docker
	@docker run -it --rm \
		--net host \
		--gpus '"device=$(GPU_ID)"' \
		--ipc host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		--name=$(CONTAINER_NAME) \
		-u $(USER_NAME) \
		-v ./:/workspace \
		-v $(PATH_TO_NUSCENES):/data/nuscenes \
		-e WANDB_API_KEY=$(WANDB_API_KEY) \
		-e TERM=xterm-256color \
		$(IMAGE_NAME):$(TAG_NAME) \
		/bin/bash -c $(1)
endef

.PHONY: build run attach jupyter train clear compile
build:
	docker build . -t $(IMAGE_NAME):$(TAG_NAME) --build-arg USER=$(USER_NAME) --build-arg UID=$(UID) --build-arg GID=$(GID)
	$(call run_docker, "sudo python setup.py develop")

run:
	$(call run_docker, "source entrypoint.sh && bash")

attach:
	docker exec -it $(CONTAINER_NAME) /bin/bash -c bash

jupyter:
	$(call run_docker, "jupyter notebook")

train:
	$(call run_docker, "sh entrypoint.sh && python tools/train.py configs/base.yaml")

DIRS_TO_CLEAN = "__pycache__" ".pytest_cache" "build" "$(IMAGE_NAME).egg-info"
clear:
	@for str in $(DIRS_TO_CLEAN); do \
		find . -name $$str | sudo xargs rm -rf; \
	done
	@sudo find . -type f -name "*.so" -exec rm {} +
	@echo "Cleaned up cache and compiled files!"

compile:
	$(call run_docker, "sudo python setup.py develop")