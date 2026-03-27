%%%%%%% Before use %%%%%%%
% DO NOT modify this file unless you know what you're doing !

% Identification
\NeedsTeXFormat{LaTeX2e}

% We base ourselves on the preexisting "report" class
\LoadClass[a4paper,12pt]{report}

%%%%%%%%%%%%%%%%
%%% PACKAGES %%%
%%%%%%%%%%%%%%%%

%%%%%%% IMPORTANT %%%%%%%
\usepackage[french]{babel} % Package for french language
\renewcommand\frenchtablename{Tableau}
\renewcommand\frenchfigurename{Figure}
\usepackage[utf8]{inputenc} % Encoding package
\usepackage{parskip} % Package for paragraph skip size
\usepackage{xcolor}
\usepackage{xcolor-solarized}
\usepackage[
    colorlinks=true,
    hyperfootnotes=false,
    linkcolor=solarized-base03,
    citecolor=solarized-red,
    urlcolor=solarized-blue
]{hyperref} % PDF information

%%%%%%% LOOKS %%%%%%%
\usepackage[T1]{fontenc} % Font
\usepackage[
    includeheadfoot,
    top=2cm, bottom=2cm, left=2cm, right=2cm,
    headsep=.8cm,
    headheight=15pt,
    footskip=1.5cm
]{geometry}% Margins
\usepackage{helvet} % Helvetica
\renewcommand{\familydefault}{\sfdefault} % Arial look-alike
\usepackage[explicit,compact]{titlesec} % Title modifications
\usepackage{fancyhdr} % Page
\usepackage{etoolbox} % For footnotes and quotes
\usepackage[
    font=small,
    labelfont=bf,
    tableposition=top
]{caption} % Better captions

%%%%%%% USEFUL %%%%%%%
\usepackage{graphicx} % Package adding figures
\usepackage{float} % Forces float environment for figures
\usepackage{emptypage} % Emptypage disables numbering of empty pages
\usepackage[nottoc,notlof,notlot,notbib]{tocbibind} % ToC modifications
\usepackage[nodisplayskipstretch]{setspace} % Spacing in reports
\usepackage{textcomp} % ASCII characters loader
\usepackage{amsmath} % Maths tools
\usepackage{amssymb} % Maths symbols loader
\usepackage{enumerate} % List insertion
\usepackage{enumitem} % List customisation
\usepackage{calc} % Natural syntax
\usepackage{nicefrac} % Better fractions in text
\usepackage{xurl} % Breaks up long urls

%%%%%%% BIBLIOGRAPHY %%%%%%%
\usepackage[nonumberlist]{glossaries} % Glossary making
\usepackage[french=guillemets]{csquotes} % Typographical rules for bib
\usepackage{notoccite} % Bib formatting
\usepackage[
    backend=biber,
    style=apa,
    sorting=nyt,
    uniquelist=false, % https://tex.stackexchange.com/questions/637149/incorrect-biblatex-apa-7-citation-when-having-multiple-same-author-year-entries
    doi=false,
    isbn=false,
    url=true,
    defernumbers,
    uniquename=false,
    date=year,
    labeldate=year,
    natbib
]{biblatex}
\DefineBibliographyStrings{french}{
  retrieved = {Récupéré}
}
\addbibresource{pages/bibliography/refs.bib}

%%%%%%% TABLES %%%%%%%
\usepackage{hhline} % Better lines in tables
\usepackage{tabularx} % Automatic table width
\usepackage{longtable} % Allows multipage tables

%%%%%%% MISC. %%%%%%%
\usepackage{lipsum} % Used in placeholder and writing

%%%%%%%%%%%%%%%%
%%% Metadata %%%
%%%%%%%%%%%%%%%%

% Directory where images can be found
\graphicspath{{figures/}}
\SetSinglespace{1}
% Line spacing 
\newlength{\singlebaselineskip} % Captions
\newlength{\textbaselineskip} % Main text
% Default line spacing
\setlength{\textbaselineskip}{1.15\baselineskip plus2pt}
% Block title style for chapters
\titleformat{\chapter}[block]
    {\fontfamily{qcs}\bfseries}{\filright\LARGE\thechapter.}{1ex}{\LARGE\filright #1 \pagestyle{fancyplain} \markboth{#1}{}}
\titlespacing{\chapter}{0px}{0px}{1px}

%%%%%%%%%%%%%%%%
%%% Commands %%%
%%%%%%%%%%%%%%%%

% For unnumbered chapters that should show up in toc
\newcommand\unnumberedchap[1]{
  \chapter*{#1}
  \addcontentsline{toc}{chapter}{#1}
  \pagestyle{fancyplain}
  \markboth{#1}{}
}

% For unnumbered chapters that should show up in toc
\newcommand\unnumberedchapnoToc[1]{
  \chapter*{#1}
  \pagestyle{fancyplain}
  \markboth{#1}{}
}

% Citations :
\AtBeginEnvironment{quote}{\small{\vspace{1.3\baselineskip}}}% Stuff before {quote}
\AtEndEnvironment{quote}{\vspace{1.3\baselineskip}}% Stuff after {quote}

% Paramètres de sortie du fichier pdf.
\pdfminorversion=5
\pdfcompresslevel=9
\pdfobjcompresslevel=3

%%%%%%%%%%%%%%%%%%%
%%% Page styles %%%
%%%%%%%%%%%%%%%%%%%

% Turn on the style
\pagestyle{fancyplain}
% Clear the header and footer
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

% Set the right side of the footer to be the page number
\fancypagestyle{fancyplain}{
    % Space between text and before text
    \setlength{\parskip}{0pt plus 1pt}
    \setlength{\parindent}{15pt}

    \fancyhf{}
    \fancyfoot[L]{
        \small \insertauthor~\textbar~Mémoire~\textbar~Université de Strasbourg
    }
    \fancyfoot[R]{
        \small \nouppercase{\leftmark}~\textbar~\thepage
    }
}

% Better figure placement
\renewcommand{\topfraction}{0.85}
\renewcommand{\textfraction}{0.1}
\renewcommand{\floatpagefraction}{0.75}

% Flexible space after figure
\setlength{\textfloatsep}{20pt plus15pt minus4pt}

% Bibliography
\nocite{*}

% Remove fancy style, leaving only the bibliography without header or footer
\renewcommand{\bibsetup}{\thispagestyle{empty}\pagestyle{empty}}
\makeatletter
\patchcmd{\blx@endenv@bibliography}{\endlist}{\endlist\clearpage}{}{}
\makeatother
\hyphenation{Fill-ing-ham}

% Remove fancy style, leaving only the glossary without header or footer
\renewcommand{\glsnamefont}[1]{\textbf{\textup{#1}}\thispagestyle{empty}}

% Footnote
\usepackage{remreset}
\makeatletter
\@removefromreset{footnote}{chapter}
\newcommand\footnoteref[1]{\protected@xdef\@thefnmark{\ref{#1}}\@footnotemark}
\makeatother
\interfootnotelinepenalty=10000
\raggedbottom