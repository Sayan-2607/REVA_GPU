.PHONY: install test test-fast example analyze check reproduce paper clean
install:      ; pip install -e ".[test]"
test:         ; pytest -q
test-fast:    ; pytest -q -m "not slow"
example:      ; rava example
analyze:      ; rava analyze
check:        ; rava check
reproduce:    ; bash scripts/run_all.sh
paper:        ; cd paper && latexmk -pdf -interaction=nonstopmode main.tex
clean:        ; cd paper && latexmk -C; find . -name __pycache__ -prune -exec rm -rf {} +
