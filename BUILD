# Synthesizing Benchmarks for Predictive Modeling (CGO'17).

licenses(["restricted"])  # GPL v3

exports_files([
    "LICENSE",
    "README.md",
])

genrule(
    name = "2017_02_cgo",
    srcs = glob([
        "tex/**/*.tex",
        "tex/img/*",
        "tex/refs.bib",
        "tex/sigplanconf.cls",
    ]),
    outs = ["2017_02_cgo.pdf"],
    cmd = "$(location //tools:autotex) docs/2017_02_cgo/tex/paper.tex $@",
    tools = ["//tools:autotex"],
)
