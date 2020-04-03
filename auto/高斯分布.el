(TeX-add-style-hook
 "高斯分布"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("ulem" "normalem")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art11"
    "inputenc"
    "fontenc"
    "graphicx"
    "grffile"
    "longtable"
    "wrapfig"
    "rotating"
    "ulem"
    "amsmath"
    "textcomp"
    "amssymb"
    "capt-of"
    "hyperref"
    "ctex")
   (LaTeX-add-labels
    "sec:org5bffd59"
    "sec:orgb47ac9b"
    "sec:org9dbe9ec"
    "sec:orgb6f6467"
    "eq:3"
    "sec:org8bd9231"
    "sec:org5a615a4"
    "eq:4"
    "sec:org9000bbb"
    "eq:5"
    "eq:6"
    "sec:orga465625"
    "eq:2"
    "eq:8"
    "eq:9"
    "sec:org6c0a7f3"
    "sec:org21eda2c"))
 :latex)

