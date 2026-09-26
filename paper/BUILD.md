# BUILD

## Style
Official ICLR 2027 style files, **unmodified**, from
https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip (linked from
https://iclr.cc/Conferences/2027/AuthorGuidelines), fetched 2026-09-26; zip sha256
`0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e`; per-file sha256 in
`iclr_style_official/SHA256SUMS`. Used via TEXINPUTS/BSTINPUTS (not copied or edited).

## TeX engine (repo-local, not installed system-wide)
No TeX engine existed. Official Tectonic binaries were unreachable (GitHub release download blocked for
this session, HTTP 403). Instead: Ubuntu 24.04 signed packages fetched with `apt-get download` (apt checks
hashes against the GPG-signed index; nothing installed), unpacked with `dpkg-deb -x` into
`paper/.texlocal/` (git-ignored). Package list and sha256: `paper/.texlocal/DEBS.SHA256`
(texlive-binaries, libkpathsea6, libptexenc1, libsynctex2, tex-common, texlive-base, texlive-latex-base,
texlive-latex-recommended, texlive-pictures, texlive-fonts-recommended 2023.20240207; poppler-utils and
libpoppler134 24.02.0-1ubuntu9 for page rendering only). Download: 62.2 MB in ~5 s (+ poppler, small).
Local fixes, all inside `paper/.texlocal`: dangling `ls-R`/`updmap.cfg` symlinks (pointing to
`/var/lib/texmf`) replaced by local files; a local `updmap.cfg` enabling the AMS CM and URW maps.
Nothing was written to `/var/lib/texmf` or other system paths.

## Commands
```bash
python3 paper/tools/cpu_run.py analysis -- python3 analysis/counter_equivalence.py   # optional analysis
python3 paper/tools/cpu_run.py analysis -- python3 paper/tools/export_numbers.py     # numbers/tables/curves
python3 paper/tools/cpu_run.py build    -- python3 paper/tools/build_pdf.py          # pdflatex, bibtex, pdflatex x<=4
```
Output: `paper/main.pdf`, log `paper/build_main.log`. CPU per call: `paper/cpu_ledger.tsv`
(caps: analysis 900 s, build 600 s; RLIMIT_AS 3 GiB; 1 thread).

## Status
BUILT, 10 pages (main text through p. 7). Engine: pdfTeX 3.141592653-2.6-1.40.25 (TeX Live 2023/Debian).
