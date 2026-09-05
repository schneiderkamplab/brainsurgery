PDFLATEX ?= pdflatex

.PHONY: all clean

all: main.pdf

main.pdf: main.tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex

clean:
	rm -f main.aux main.log main.out main.pdf
