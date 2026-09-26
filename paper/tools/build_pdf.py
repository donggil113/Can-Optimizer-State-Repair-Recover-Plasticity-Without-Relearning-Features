"""Build paper/main.pdf with a repo-local TeX Live 2023 extracted from signed Ubuntu 24.04 debs.

No system installation: paper/.texlocal/root holds `dpkg-deb -x` output of the packages listed in
paper/.texlocal/DEBS.SHA256; all kpathsea paths are redirected there by environment variables.
Official ICLR style files are used unmodified from paper/iclr_style_official/iclr2027 via TEXINPUTS/BSTINPUTS.

    python3 paper/tools/cpu_run.py build -- python3 paper/tools/build_pdf.py
"""

import os
import shutil
import subprocess
import sys

PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L = os.path.join(PAPER, ".texlocal")
R = os.path.join(L, "root")
STYLE = os.path.join(PAPER, "iclr_style_official", "iclr2027")
BIN = os.path.join(R, "usr", "bin")


def env():
    e = dict(os.environ)
    dist = os.path.join(R, "usr/share/texlive/texmf-dist")
    deb = os.path.join(R, "usr/share/texmf")
    var = os.path.join(L, "var")
    e.update({
        "PATH": BIN + os.pathsep + e["PATH"],
        "LD_LIBRARY_PATH": os.path.join(R, "usr/lib/x86_64-linux-gnu"),
        "TEXMFCNF": os.path.join(R, "usr/share/texmf/web2c"),
        "TEXMFROOT": os.path.join(R, "usr/share/texlive"),
        "TEXMFDIST": dist, "TEXMFMAIN": dist, "TEXMFDEBIAN": deb,
        "TEXMFLOCAL": os.path.join(L, "local"), "TEXMFSYSVAR": var, "TEXMFVAR": var,
        "TEXMFSYSCONFIG": os.path.join(R, "etc/texmf"), "TEXMFCONFIG": os.path.join(L, "config"),
        "TEXMFHOME": os.path.join(L, "home"),
        "TEXFORMATS": os.path.join(var, "web2c") + "//:",
        "TEXINPUTS": ".:" + STYLE + "//:",
        "BSTINPUTS": ".:" + STYLE + "//:",
        "BIBINPUTS": ".:",
        "SOURCE_DATE_EPOCH": "0", "FORCE_SOURCE_DATE": "1",
        "PERL5LIB": os.path.join(R, "usr/share/texlive/tlpkg"),
    })
    return e


def run(cmd, cwd=PAPER, check=True):
    print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=cwd, env=env())
    if check and p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd)}")
    return p.returncode


def setup():
    var = os.path.join(L, "var")
    fmt = os.path.join(var, "web2c", "pdftex", "pdflatex.fmt")
    if os.path.exists(fmt):
        return
    for d in ("local", "config", "home", "var/web2c/pdftex"):
        os.makedirs(os.path.join(L, d), exist_ok=True)
    e = env()
    # The debs ship ls-R as symlinks into /var/lib/texmf (system path, absent here). Replace them in the
    # local extracted tree with real files so nothing is written outside paper/.texlocal.
    for tree in (e["TEXMFDIST"], e["TEXMFDEBIAN"]):
        lsr = os.path.join(tree, "ls-R")
        if os.path.islink(lsr):
            os.remove(lsr)
    # Debian generates updmap.cfg at install time (dangling symlink in the debs); write a local one
    # enabling only map files present in the extracted tree.
    cfg_dir = os.path.join(L, "local", "web2c")
    os.makedirs(cfg_dir, exist_ok=True)
    maps = ["cm.map", "cmextra.map", "symbols.map", "euler.map", "latxfont.map", "utm.map", "uhv.map",
            "ucr.map", "usy.map", "uzd.map", "psnfss.map", "eurosym.map", "rsfs.map", "wasy.map", "marvosym.map"]
    with open(os.path.join(cfg_dir, "updmap.cfg"), "w") as f:
        f.write("LW35 URWkb\npdftexDownloadBase14 true\ndvipsDownloadBase35 false\n")
        f.write("".join(f"Map {m}\n" for m in maps))
    for tree in (e["TEXMFDIST"], e["TEXMFDEBIAN"]):
        for rel in ("web2c/updmap.cfg",):
            pth = os.path.join(tree, rel)
            if os.path.islink(pth) and not os.path.exists(pth):
                os.remove(pth)
    run(["sh", os.path.join(BIN, "mktexlsr"), e["TEXMFDIST"], e["TEXMFDEBIAN"], var, os.path.join(L, "local")])
    run([os.path.join(BIN, "updmap"), "--sys", "--nohash", "--quiet",
         "--cnffile", os.path.join(cfg_dir, "updmap.cfg")])
    run([os.path.join(BIN, "mktexlsr"), var])
    tmp = os.path.join(var, "web2c", "pdftex")
    run([os.path.join(BIN, "pdftex"), "-ini", "-interaction=nonstopmode", "-jobname=pdflatex",
         "-progname=pdflatex", "-etex", "-translate-file=cp227.tcx", "*pdflatex.ini"], cwd=tmp)


def main():
    setup()
    for ext in ("aux", "bbl", "blg", "out"):
        p = os.path.join(PAPER, "main." + ext)
        if os.path.exists(p):
            os.remove(p)
    pdflatex = [os.path.join(BIN, "pdflatex"), "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    run(pdflatex)
    run([os.path.join(BIN, "bibtex.original"), "main"])
    for _ in range(4):
        run(pdflatex)
        with open(os.path.join(PAPER, "main.log"), errors="replace") as f:
            if "Rerun to get cross-references right" not in f.read():
                break
    shutil.copy(os.path.join(PAPER, "main.log"), os.path.join(PAPER, "build_main.log"))
    print("built", os.path.join(PAPER, "main.pdf"))


if __name__ == "__main__":
    sys.exit(main())
