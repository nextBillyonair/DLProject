PDFLATEX:=pdflatex
BIBTEX:=bibtex


TEX_SRC=interim.tex

${TEX_SRC:.tex=.pdf}:%.pdf: %.tex %.bib
	$(PDFLATEX) $<
	$(BIBTEX) ${<:.tex=}
	$(PDFLATEX) $<
	$(PDFLATEX) $<

clean:
	rm -f *.out *.aux *.log *.bbl *.blg


.PHONY: clean
