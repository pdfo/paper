## Makefile
## Copyright 2023 Tom M. Ragonneau and Zaikun Zhang
LC := latexmk
LCFLAGS := -file-line-error -halt-on-error -interaction=nonstopmode

latex: $(basename $(wildcard *.tex))

%: %.tex
	$(LC) $(LCFLAGS) $^

.PHONY: clean
clean:
	$(LC) -c
