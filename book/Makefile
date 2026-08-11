# Book build targets. Invoke from the repository root:
#
#     make -C book html      # build to book/_build/html
#     make -C book serve      # build, then serve at http://localhost:8000
#     make -C book check      # report lesson cells that were never executed
#     make -C book clean      # remove build artifacts
#
# The Sphinx *source* directory is the repository root, not this directory, so that
# `notebooks/` and `course_utils/` keep the same import paths they have when a lesson is
# run directly. Only the book's own files (config, toc, pages, static assets) live here,
# and the output is redirected back under book/ via --path-output.

ROOT   := $(realpath $(CURDIR)/..)
JB     := $(ROOT)/.venv-docs/bin/jupyter-book
PYTHON := $(ROOT)/.venv-docs/bin/python
HTML   := $(CURDIR)/_build/html
PORT   ?= 8000

.PHONY: help html serve check clean

help:
	@echo "html   build the book to book/_build/html"
	@echo "serve  build, then serve at http://localhost:$(PORT)"
	@echo "check  report lesson code cells with no execution_count"
	@echo "clean  remove the build output"

html:
	cd $(ROOT) && $(JB) build . \
		--path-output book \
		--config book/_config.yml \
		--toc book/_toc.yml

serve: html
	cd $(HTML) && $(PYTHON) -m http.server $(PORT)

check:
	cd $(ROOT) && $(PYTHON) book/scripts/check_outputs.py

clean:
	rm -rf $(CURDIR)/_build
