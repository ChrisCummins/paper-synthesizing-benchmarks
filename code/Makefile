.DEFAULT_GOAL = install

.PHONY: install run

ifeq ($(VIRTUAL_ENV),)
$(error "not in a virtualenv")
endif

PYTHON = $(VIRTUAL_ENV)/bin/python
PIP = $(VIRTUAL_ENV)/bin/pip

data_files := benchmarks/amd-app-sdk-3.0

%: %.tar.bz2
	cd $(dir $<) && tar xf $(notdir $<)
	touch $@

install: .ipython/kernels/CLgen/kernel.json $(data_files)
	$(MAKE) -C ../data
	mkdir -p ~/.ipython/kernels
	test -d ~/.ipython/kernels/CLgen || cp -R .ipython/kernels/CLgen ~/.ipython/kernels
	$(PIP) install -r requirements.txt

.ipython/kernels/CLgen/kernel.json: .ipython/kernels/CLgen/kernel.json.template
	cp $< $@
	sed "s,@PYTHON@,$(PYTHON)," -i $@

run:
	jupyter notebook
