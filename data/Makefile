.DEFAULT_GOAL = all

data_files := \
	clgen-github-model-2016-nov-2048x3.tar.bz2 \
	clgen-1000 \
	github.db \
	tiny-corpus

clgen-github-model-2016-nov-2048x3.tar.bz2:
	cat clgen-github-model-2016-nov-2048x3-* > $@

%: %.tar.bz2
	tar xf $<
	touch $@

clean:
	rm -rf $(data_files)

all: $(data_files)
