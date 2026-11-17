#!/bin/bash
# Build the manuscript scaffold and the experimental briefing.
#   bash papers/manuscript/build.sh            # both PDFs
#   bash papers/manuscript/build.sh briefing   # one of: main, briefing
# The baselines equations are a single source in docs/methods-equations/methods-equations.tex;
# the body between the BEGIN BODY / END BODY markers is extracted here so both documents \input it.
set -e
cd "$(dirname "$0")"
sed -n '/^% -* BEGIN BODY/,/^% -* END BODY/p' ../../docs/methods-equations/methods-equations.tex | sed '1d;$d' > methods-baselines-body.tex
# BibTeX reads a stray "@" inside a comment as an entry, so the data-source bib is copied without its comment lines.
grep -v "^%" ../../docs/data-sources.bib > data-sources-plain.bib
for doc in ${1:-main briefing}; do
  pdflatex -interaction=nonstopmode -halt-on-error "$doc.tex" > /dev/null
  bibtex "$doc" > /dev/null || true
  pdflatex -interaction=nonstopmode -halt-on-error "$doc.tex" > /dev/null
  pdflatex -interaction=nonstopmode -halt-on-error "$doc.tex" > /dev/null
  echo "built $doc.pdf"
done
rm -f *.aux *.bbl *.blg *.out *.toc
