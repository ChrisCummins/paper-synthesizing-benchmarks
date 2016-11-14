.DEFAULT_GOAL = all

unpacked_files := \
	github.db \
	tiny-corpus

clgen-github-model-2016-nov-2048x3.tar.bz2:
	cat clgen-github-model-2016-nov-2048x3-* > $@

%: %.tar.bz2
	tar xf $<
	touch $@

clean:
	rm -rf $(unpacked_files)

all: $(unpacked_files) clgen-github-model-2016-nov-2048x3.tar.bz2
