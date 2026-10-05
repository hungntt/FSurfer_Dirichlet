#!/bin/bash
# Final analyses, tables, figures, and LaTeX build for main-acm.tex.
# Run after all artifacts exist (neural backbones: seeds 42-45, classical: seeds 42-43).
set -e
cd "$(dirname "$0")"
P=~/venvs/fsd/bin/python
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
DEEP=42,43,44,45
CLASSIC=42,43

for s in 44 45; do
  $P analyze_cv_refinement.py --artifacts-root ../outputs/cv_artifacts/seed$s --output-root ../outputs/cv_analysis/seed$s --backbones tabm,resnet --light
done
for s in; do
  $P analyze_cv_refinement.py --artifacts-root ../outputs/cv_artifacts/seed$s --output-root ../outputs/cv_analysis/seed$s --backbones logreg,svm,rf --light
done
$P analyze_cv_reliability.py --seeds $DEEP --backbones tabm,resnet
$P analyze_cv_reliability.py --seeds $CLASSIC --backbones logreg,svm,rf
$P analyze_cv_extras.py --seeds $DEEP --backbones tabm,resnet,logreg,svm,rf
$P make_acm_figures.py --seeds $DEEP --reliability-dir ../outputs/cv_analysis/reliability_seeds42-43-44-45
$P make_acm_tables.py --deep-seeds $DEEP --classic-seeds $CLASSIC

cd ../6a4e845051244b4623abab2e
grep -ohE "\\\\N[A-Z][A-Za-z]+" main-acm.tex sec_acm/*.tex | sort -u | sed -E 's/(.*)/\\providecommand{\1}{??}/' > acm_fallback.tex
export PATH=~/.TinyTeX/bin/aarch64-linux:$PATH
B=../outputs/acm_build
rm -rf $B && mkdir -p $B
cp -r main-acm.tex sec_acm img tables acm_numbers.tex acm_fallback.tex ref-acm.bib $B/
cd $B
pdflatex -interaction=nonstopmode main-acm.tex > b1.log 2>&1 || true
bibtex main-acm > bib.log 2>&1 || true
pdflatex -interaction=nonstopmode main-acm.tex > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main-acm.tex > build.log 2>&1 || true
grep -E "Output written|^! " build.log | sort | uniq -c
grep -c "??" <(pdftotext main-acm.pdf - 2>/dev/null) 2>/dev/null || true
