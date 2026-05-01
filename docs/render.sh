#!/bin/bash
# Render every docs/tex/*.tex into docs/figures/*.svg via pdflatex + dvisvgm.
# Run from the project root or from this directory.

set -e
cd "$(dirname "$0")"

mkdir -p figures

for tex in tex/*.tex; do
    name=$(basename "$tex" .tex)
    tmpdir=$(mktemp -d)
    cp "$tex" "$tmpdir/$name.tex"
    (cd "$tmpdir" && pdflatex -interaction=nonstopmode "$name.tex" >/dev/null)
    dvisvgm --pdf --no-fonts --output="figures/$name.svg" "$tmpdir/$name.pdf" >/dev/null
    rm -rf "$tmpdir"
    echo "rendered figures/$name.svg"
done
