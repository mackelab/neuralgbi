from invoke import task
from pathlib import Path

basepath = "."
open_cmd = "open"

fig_names = {
    "1": "paper/fig1",
    "2": "paper/fig2",
    "3": "paper/fig3",
    "4": "paper/fig4",
    "5": "paper/fig5",
}


@task
def convert(c, fig):
    _convertsvg2pdf(c, fig)
    _convertpdf2png(c, fig)


@task
def _convertsvg2pdf(c, fig):
    if fig is None:
        for f in range(len(fig_names)):
            _convert_svg2pdf(c, str(f + 1))
        return
    pathlist = Path("{bp}/{fn}/fig/".format(bp=basepath, fn=fig_names[fig])).glob(
        "*.svg"
    )
    for path in pathlist:
        c.run("inkscape {} --export-pdf={}.pdf".format(str(path), str(path)[:-4]))


@task
def _convertpdf2png(c, fig):
    if fig is None:
        for f in range(len(fig_names)):
            _convert_pdf2png(c, str(f + 1))
        return
    pathlist = Path("{bp}/{fn}/fig/".format(bp=basepath, fn=fig_names[fig])).glob(
        "*.pdf"
    )
    for path in pathlist:
        c.run(
            'inkscape {} --export-png={}.png -b "white" --export-dpi=250'.format(
                str(path), str(path)[:-4]
            )
        )
