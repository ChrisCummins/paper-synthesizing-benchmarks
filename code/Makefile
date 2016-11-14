.DEFAULT_GOAL = install

.PHONY: install run

ifeq ($(VIRTUAL_ENV),)
#
# System-wide install:
#
PYTHON ?= python
PIP ?= pip

install: .ipython/kernels/CLgen/kernel.json
	$(MAKE) -C ../data
	mkdir -p ~/.ipython/kernels
	test -d ~/.ipython/kernels/CLgen || cp -R .ipython/kernels/CLgen ~/.ipython/kernels
	$(PIP) install -r requirements.txt

.ipython/kernels/CLgen/kernel.json: .ipython/kernels/CLgen/kernel.json.template
	cp $< $@
	sed "s,@PYTHON@,$(PYTHON)," -i $@

run:
	jupyter notebook
else
#
# Virtualenv install:
#
install: .ipython/kernels/CLgen/kernel.json
	$(MAKE) -C ../data
	mkdir -p ~/.ipython/kernels
	test -d ~/.ipython/kernels/CLgen || cp -R .ipython/kernels/CLgen ~/.ipython/kernels
	$(VIRTUAL_ENV)/bin/pip install -r requirements.txt

.ipython/kernels/CLgen/kernel.json: .ipython/kernels/CLgen/kernel.json.template
	cp $< $@
	sed "s,@PYTHON@,$(VIRTUAL_ENV)/bin/python," -i $@

run:
	$(VIRTUAL_ENV)/bin/jupyter notebook
endif
