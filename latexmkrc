## latexmkrc
## Copyright 2023 Tom M. Ragonneau and Zaikun Zhang

# Generate pdf using pdflatex.
$pdf_mode = 1;
$postscript_mode = 0;
$dvi_mode = 0;

# Run bibtex or biber as needed to regenerate the bbl files
$bibtex_use = 2;

# Remove extra extensions on clean
$clean_ext = "run.xml synctex.gz";
