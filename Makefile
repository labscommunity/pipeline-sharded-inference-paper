.PHONY: all clean

PAPER = main

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib arxiv.sty
	pdflatex $(PAPER)
	bibtex $(PAPER)
	pdflatex $(PAPER)
	pdflatex $(PAPER)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz $(PAPER).pdf
