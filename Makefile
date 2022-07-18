pdf:
	latexmk -pdf

clean:
	rm -f main.aux  main.bbl  main.bcf  main.blg  main.log  main.out  main.pdf  main.xdv
