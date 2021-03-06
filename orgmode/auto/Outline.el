(TeX-add-style-hook
 "Outline"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("ulem" "normalem")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
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
    "sec:orga260cf5"
    "sec:org3491100"
    "sec:orgb166ea1"
    "sec:org2a5f9b5"
    "sec:org260a4a1"
    "sec:orgf822865"
    "sec:org5732098"
    "sec:orga29245b"
    "sec:org0017be6"
    "sec:org0f4a600"
    "sec:orgafff404"
    "sec:org9869aeb"
    "sec:org8c580a1"
    "sec:orgcf7865b"
    "sec:org80401fc"
    "sec:orgd568c6b"
    "sec:orgf4a788b"
    "sec:org51846b2"
    "sec:org184b605"
    "sec:orga995de2"
    "sec:org6a9f713"
    "sec:org856a12b"
    "sec:orgc8dd216"
    "sec:orgdb341de"
    "sec:org3863831"
    "sec:org9e4236e"
    "sec:org38528ff"
    "sec:orgf7d7146"
    "sec:org8d2f5d5"
    "sec:org501a54e"
    "sec:org9243dd5"
    "sec:orgf2cce2d"
    "sec:org4caddef"
    "sec:org841087f"
    "sec:orgd49e1a0"
    "sec:orge127ef9"
    "sec:org204462d"
    "sec:orgd0f0ebd"
    "sec:orgbf1eebc"
    "sec:orgdf49d52"
    "sec:org3be65da"
    "sec:orgae9b8a6"
    "sec:org7ea4338"
    "sec:orgf548733"
    "sec:org963cfde"
    "sec:orgf097cc7"
    "sec:org522c237"
    "sec:org82a0d7a"
    "sec:orgf23585c"
    "sec:org21f07f2"
    "sec:org0f2a5ae"
    "sec:orge69c7bc"
    "sec:orgaa35ce9"
    "sec:org3740017"
    "sec:orgdffa043"
    "sec:org2ed51b4"
    "sec:org89d006d"
    "sec:orgad70185"
    "sec:org082fabb"
    "sec:org9fef3c6"
    "sec:org4c8e872"
    "sec:org364a2a3"
    "sec:orga2648fe"
    "sec:orgafe4052"
    "sec:orgab1316a"
    "sec:org8f5edbd"
    "sec:org85edb7d"
    "sec:org7d2c0b0"
    "sec:orga5992b0"
    "sec:org1c46dd6"
    "sec:org409c8cf"
    "sec:org6e6ade7"
    "sec:org97e26fd"
    "sec:org7c1694e"
    "sec:org6671a39"
    "sec:orgfdbecac"
    "sec:org51aebab"))
 :latex)

