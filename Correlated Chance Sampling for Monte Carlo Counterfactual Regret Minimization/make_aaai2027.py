#!/usr/bin/env python3
"""Generate the AAAI-2027 anonymous submission (main + supplementary) from the
single-column arXiv source paper.  Idempotent: re-running overwrites the outputs.

AAAI2027/main.tex and AAAI2027/supplementary.tex are DERIVED FILES.  Do not edit
them by hand: the next run of this script overwrites both.  Content changes go
into ../main.tex (the arXiv source); submission-only rewrites go into the
substitution tables below.  Every substitution is anchored with an
``assert count == 1`` so that an edit upstream fails loudly instead of silently
dropping a rewrite.

Usage:
    python make_aaai2027.py
    cd AAAI2027 && latexmk -pdf main.tex supplementary.tex

The submission body must occupy exactly 7 pages (AAAI allows 7 pages of content
plus up to 2 pages of references).  Check with:
    pdftotext -f 8 -l 8 main.pdf - | head -1     # must print "References"
"""

import re
from pathlib import Path

# Resolve relative to this file so the directory can be moved or renamed.
SRC = Path(__file__).resolve().parent / "main.tex"
OUT = SRC.parent / "AAAI2027"

lines = SRC.read_text().splitlines(keepends=True)


def _line_of(prefix: str) -> int:
    """0-based index of the unique line starting with prefix."""
    hits = [i for i, ln in enumerate(lines) if ln.startswith(prefix)]
    assert len(hits) == 1, f"anchor not unique: {prefix!r} -> {hits}"
    return hits[0]


# Anchored on structure rather than hard-coded line numbers, so that editing the
# arXiv source (e.g. adding a Related Work paragraph) does not silently shift the
# slice boundaries.
_ABS = _line_of(r"\begin{abstract}")
_ACK = _line_of(r"\section*{Acknowledgments}")
_APX = _line_of(r"\appendix")
_END = _line_of(r"\end{document}")

body = "".join(lines[_ABS:_ACK - 1])   # abstract .. end of Conclusion
appx = "".join(lines[_APX:_END - 1])   # \appendix .. before \end{document}

# ---------------------------------------------------------------- preambles --
COMMON = r"""\documentclass[letterpaper]{article} % DO NOT CHANGE THIS
\usepackage[submission]{aaai2027}  % DO NOT CHANGE THIS
\usepackage[hyphens]{url}  % DO NOT CHANGE THIS
\usepackage{graphicx} % DO NOT CHANGE THIS
\urlstyle{rm} % DO NOT CHANGE THIS
\def\UrlFont{\rm}  % DO NOT CHANGE THIS
\usepackage{natbib}  % DO NOT CHANGE THIS AND DO NOT ADD ANY OPTIONS TO IT
\usepackage{caption} % DO NOT CHANGE THIS AND DO NOT ADD ANY OPTIONS TO IT
\frenchspacing  % DO NOT CHANGE THIS
\setlength{\pdfpagewidth}{8.5in}  % DO NOT CHANGE THIS
\setlength{\pdfpageheight}{11in}  % DO NOT CHANGE THIS

\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage{amsthm}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algorithmic}

\pdfinfo{
/TemplateVersion (2027.1)
}

\setcounter{secnumdepth}{2}

%% Float placement: the paper is float-heavy, so allow denser float pages.
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{5}
\setcounter{dbltopnumber}{3}
\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.7}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.85}
\renewcommand{\dbltopfraction}{0.95}
\renewcommand{\dblfloatpagefraction}{0.85}

\newtheorem{proposition}{Proposition}
\newtheorem{theorem}{Theorem}
\newtheorem{observation}{Observation}
"""

MAIN_HEAD = COMMON + r"""
%% Section letters of the supplementary material (verified against
%% supplementary.aux by check_numbers.py).
\newcommand{\appproofs}{A}
\newcommand{\appscope}{B}
\newcommand{\appfuture}{C}
\newcommand{\appprotocol}{D}
\newcommand{\appreset}{E}
\newcommand{\apprevisit}{F}
\newcommand{\appcomposition}{G}
\newcommand{\apphunl}{H}
\newcommand{\appextra}{I}
\newcommand{\appcurves}{J}

\title{Faster Game Solving via Correlated Chance Sampling}

\author{Anonymous submission}
\affiliations{Anonymous submission}

\begin{document}
\maketitle

"""

SUPP_HEAD = COMMON + r"""
%% Results restated from the main text keep the main-text numbering; floats and
%% the new appendix theorem get an S prefix.
\newtheorem{mainthm}{Theorem}

\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\thetheorem}{S\arabic{theorem}}

\title{Supplementary Material:\\Faster Game Solving via Correlated Chance Sampling}

\author{Anonymous submission}
\affiliations{Anonymous submission}

\begin{document}
\maketitle

"""

TAIL = "\n\\bibliography{references}\n\n\\end{document}\n"

# ------------------------------------------------------- main-text rewrites --
APP = {
    "scope": r"\appscope{}", "future": r"\appfuture{}",
    "experiment-protocol": r"\appprotocol{}", "reset-proof": r"\appreset{}",
    "revisit-protocol": r"\apprevisit{}", "composition-details": r"\appcomposition{}",
    "es-control-variate": r"\appcomposition{}", "hunl-details": r"\apphunl{}",
    "extra": r"\appextra{}", "adaptive-frequency": r"\appextra{}",
}

# Cross-file references to supplementary floats/theorems, rewritten by hand.
MAIN_EXACT = [
    (r"and Theorem~\ref{thm:reset} gives a per-traversal reset construction that "
     r"retains the standard External Sampling guarantee.",
     r"and a per-traversal reset construction (Theorem~S1, Appendix~\appreset{}) "
     r"retains the standard External Sampling guarantee."),
    (r"and shows no gain (Table~\ref{tab:revisit})",
     r"and shows no gain (Table~S2, Appendix~\apprevisit{})"),
    (r"as the HUNL reduction figures (Figure~\ref{fig:hunl}, Figure~\ref{fig:hunl-river}) and is",
     r"as the HUNL reduction figures in Appendix~\apphunl{} and is"),
    (r"Table~\ref{tab:revisit} reports this revisit diagnostic; "
     r"Appendix~\ref{app:revisit-protocol} gives its measurement procedure.",
     r"Table~S2 in Appendix~\apprevisit{} reports this revisit diagnostic, and that "
     r"appendix gives its measurement procedure."),
    (r"Table~\ref{tab:revisit} and Figure~\ref{fig:revisit} add the corresponding "
     r"concrete-node exposure diagnostics",
     r"Table~S2 and Figure~S1 add the corresponding concrete-node exposure diagnostics"),
]

for old, new in MAIN_EXACT:
    assert body.count(old) == 1, f"main exact-replacement not unique: {old[:60]!r}"
    body = body.replace(old, new)

assert not re.search(r"\\ref\{(tab|fig|thm):(revisit|hunl|hunl-river|reset)\}", body)
body = re.sub(r"\\ref\{app:([a-z-]+)\}", lambda m: APP[m.group(1)], body)

# First mention of the supplementary material spells the pointer out once.
body = body.replace(
    r"Appendix~\appscope{} states the scope of these claims.",
    r"Appendix~\appscope{} of the supplementary material states the scope of these claims.",
    1,
)

# ------------------------------------------------- Related Work compression --
# Same content, same citations, fewer words: merged repetitions only.
RELATED = [
    (r"Outcome Sampling samples a single trajectory and uses importance "
     r"corrections. External Sampling samples opponent actions and chance outcomes "
     r"while enumerating the traversing player's actions, and is the external "
     r"sampling regime used throughout this paper.",
     r"Outcome Sampling samples a single trajectory with importance corrections; "
     r"External Sampling samples opponent actions and chance outcomes while "
     r"enumerating the traversing player's actions, and is the regime used "
     r"throughout this paper."),
    (r"These methods choose what is sampled and how the sampled estimator is "
     r"weighted. CCS-MCCFR keeps the chance distribution and the external sampling "
     r"estimator fixed, but changes the temporal structure of the chance draws by "
     r"attaching a persistent low discrepancy stream to each concrete chance node "
     r"across iterations.",
     r"These methods choose what is sampled and how the estimate is weighted. "
     r"CCS-MCCFR keeps the chance distribution and the external sampling estimator "
     r"fixed and changes the temporal structure of the draws, attaching a persistent "
     r"low discrepancy stream to each concrete chance node across iterations."),
    (r"ESCHER is a model-free deep CFR-style algorithm that combines a learned "
     r"history-value estimator with a fixed update-player sampling policy, removing "
     r"the OS-MCCFR-derived reach and continuation importance corrections from its "
     r"own estimator \citep{mcaleer2023escher}.",
     r"ESCHER combines a learned history-value estimator with a fixed update-player "
     r"sampling policy, removing the OS-MCCFR-derived reach and continuation "
     r"importance corrections from its own estimator \citep{mcaleer2023escher}."),
    (r"Our intervention instead leaves the External-Sampling estimator algebra "
     r"unchanged and alters the temporal allocation of repeated chance outcomes at "
     r"each concrete node. Thus VR-MCCFR and CCS-MCCFR act on "
     r"distinct algorithmic axes: the former corrects sampled value estimates, "
     r"whereas the latter changes the cross-visit schedule of chance outcomes while "
     r"preserving the chance law. Because they act on different axes, they can be "
     r"applied together, and we measure that composition rather than assume it.",
     r"Our intervention leaves the External-Sampling estimator algebra unchanged and "
     r"instead reallocates repeated chance outcomes across visits to each concrete "
     r"node, so the two act on distinct axes and can be applied together; we measure "
     r"that composition rather than assume it."),
    (r"These are modular implementation axes, but an update rule changes the "
     r"downstream values paired with the chance stream. Section~\ref{sec:orthogonality} "
     r"therefore measures their empirical composition directly, and "
     r"Appendix~\appcomposition{} reports a restricted External-Sampling "
     r"control-variate composition test motivated by VR-MCCFR.",
     r"An update rule changes the downstream values paired with the chance stream, so "
     r"Section~\ref{sec:orthogonality} measures the composition directly and "
     r"Appendix~\appcomposition{} reports a restricted External-Sampling "
     r"control-variate test motivated by VR-MCCFR."),
    (r"CCS-MCCFR uses the simplest one dimensional randomized construction, a "
     r"shifted Weyl rotation, and binds one stream to each concrete chance node. The "
     r"primitive is classical; the placement inside adaptive repeated MCCFR traversal "
     r"is the point that creates both the benefit and the phase selection issue "
     r"analyzed in Section~\ref{sec:method}.",
     r"CCS-MCCFR binds the simplest one dimensional randomized construction, a "
     r"shifted Weyl rotation, to each concrete chance node. The primitive is "
     r"classical; placing it inside adaptive repeated MCCFR traversal creates both "
     r"the benefit and the phase selection issue analyzed in "
     r"Section~\ref{sec:method}."),
    (r"Their target is online sampling at search time, whereas our target is repeated "
     r"tabular MCCFR regret estimation with persistent per-chance-node streams. Common "
     r"random numbers in reinforcement learning \citep{peshkin2002learning} and "
     r"antithetic sampling in random search \citep{mania2018simple} also use "
     r"correlation to reduce estimator noise, but they do not address chance sampling "
     r"inside extensive form game regret minimization. We contribute a new placement "
     r"of randomized low discrepancy streams within MCCFR, an explicit characterization "
     r"of adaptive phase selection, and paired experiments that identify both favorable "
     r"configurations and empirical regime boundaries.",
     r"Their target is online sampling at search time, whereas ours is repeated tabular "
     r"MCCFR regret estimation with persistent per-chance-node streams. Common random "
     r"numbers in reinforcement learning \citep{peshkin2002learning} and antithetic "
     r"sampling in random search \citep{mania2018simple} also exploit correlation, but "
     r"not inside extensive form game regret minimization."),
]
RELATED += [
    (r"and four HUNL transfer stress tests place the endpoint at vanilla with no "
     r"statistically detectable final-endpoint difference, and revisit exposure, "
     r"symmetry, and private-information coupling organize the full pattern as "
     r"empirical moderators.",
     r"and four HUNL transfer stress tests show no statistically detectable "
     r"final-endpoint difference, with revisit exposure, symmetry, and "
     r"private-information coupling as the empirical moderators."),
    (r"Its first $N$ draws at one concrete chance node, where $N$ is the node's "
     r"consumed-draw count rather than global iterations, satisfy deterministic "
     r"$O(\!\log(N+1)/N)$ frequency discrepancy, and fixed-index marginal correctness "
     r"together with fixed-trajectory unbiasedness supply complementary local "
     r"guarantees. The adaptive phase-selection bound then names the single dependence "
     r"term that a global analysis must control, and per-traversal resetting already "
     r"retains the standard $O(1/\sqrt{T})$ guarantee. The paper thus establishes "
     r"concrete local sampling properties, demonstrates substantial value in tabular "
     r"regimes, and isolates a precise next question for the theory of adaptive "
     r"correlated sampling.",
     r"Its first $N$ draws at one concrete chance node, where $N$ counts that node's "
     r"consumed draws rather than global iterations, satisfy deterministic "
     r"$O(\!\log(N+1)/N)$ frequency discrepancy, and fixed-index marginal correctness "
     r"together with fixed-trajectory unbiasedness supply complementary local "
     r"guarantees. The adaptive phase-selection bound names the single dependence term "
     r"a global analysis must control, and per-traversal resetting already retains the "
     r"standard $O(1/\sqrt{T})$ guarantee, which isolates a precise next question for "
     r"the theory of adaptive correlated sampling."),
]

RELATED += [
    (r"The result is stable across the tested 6/10/12-card expansion, with the "
     r"10-card configuration producing the largest measured reduction. This "
     r"controlled family complements the standard Leduc result by showing that the "
     r"benefit persists when the chance-outcome space is enlarged while the game "
     r"structure is held fixed.",
     r"The result is stable across the tested 6/10/12-card expansion, with the "
     r"10-card configuration producing the largest measured reduction, so the benefit "
     r"persists when the chance-outcome space is enlarged with the game structure "
     r"held fixed."),
    (r"The complete theorem, proof, implementation distinction, ablation table, and "
     r"diagnostics appear in Appendix~\appreset{}.",
     r"Appendix~\appreset{} gives the theorem, proof, implementation distinction, "
     r"ablation table, and diagnostics."),
]

for old, new in RELATED:
    assert body.count(old) == 1, f"related-work rewrite not unique: {old[:60]!r}"
    body = body.replace(old, new)

# ------------------------------- Leduc-family curves moved to supplementary --
# Table 2 already carries every endpoint, reduction, and p-value of this
# experiment; the trajectory plot is only legible at full page width, so it is
# reproduced there instead of being shrunk into one column.
_fig_re = re.compile(r"\\begin\{figure\}\[t\]\n(?:(?!\\end\{figure\}).)*?"
                     r"\\label\{fig:bigleduc\}\n\\end\{figure\}\n\n", re.S)
_m = _fig_re.search(body)
assert _m, "fig:bigleduc block not found"
bigleduc_fig = (_m.group(0).replace(r"\begin{figure}[t]", r"\begin{figure*}[t]")
                .replace(r"\end{figure}", r"\end{figure*}")
                .replace(r"using the same convention as Figure~\ref{fig:convergence}; "
                         r"since this figure uses 50 paired seeds whereas "
                         r"Figure~\ref{fig:convergence} uses 200",
                         r"using the same convention as Figure~1 of the main text; "
                         r"since this figure uses 50 paired seeds whereas that "
                         r"figure uses 200"))
assert r"\ref{fig:convergence}" not in bigleduc_fig
body = body[:_m.start()] + body[_m.end():]

_curve_ref = (r"(Table~\ref{tab:bigleduc}; convergence curves in "
              r"Figure~\ref{fig:bigleduc})")
assert body.count(_curve_ref) == 1
body = body.replace(_curve_ref,
                    r"(Table~\ref{tab:bigleduc}; convergence trajectories in "
                    r"Appendix~\appcurves{})")
assert "fig:bigleduc" not in body

# ---------------------------------------- proofs moved to the supplementary --
# The main text keeps every statement in full and adds a one-sentence sketch;
# the four proofs are reproduced verbatim in the supplementary.
PROOF_ORDER = ["prop:unbiased", "prop:traj", "prop:biasbound", "thm:freq"]
MAIN_NUMBER = {"prop:unbiased": "Proposition 1", "prop:traj": "Proposition 2",
               "prop:biasbound": "Proposition 3", "thm:freq": "Theorem 1"}

SKETCH = {
    "prop:unbiased": (
        r"Shifting by a constant modulo one is measure preserving on the circle, so "
        r"$u_{c,n}$ is uniform for every fixed $n$ and the quantile map pushes it "
        r"forward to $f_c$; the same holds for any random index independent of "
        r"$\phi_c$. Appendix~\appproofs{} gives the proof."),
    "prop:traj": (
        r"In an acyclic tree a concrete chance node is consumed at most once per "
        r"traversal, and under a fixed strategy sequence the prior visit count "
        r"$N_c(t)$ is independent of $\phi_c$, so conditioning on it reduces the draw "
        r"to the fixed-index case of Proposition~\ref{prop:unbiased}. Distinct nodes "
        r"carry independent phases, so the within-traversal chance law is the product "
        r"law of independent External Sampling. Appendix~\appproofs{} gives the proof."),
    "prop:biasbound": (
        r"Conditioning on $g_t$ and on the reach event turns $B_c(t)$ into the gap "
        r"between the expectations of one bounded test function under the conditional "
        r"phase law and under the uniform law, which total variation bounds by "
        r"$2G\,\delta_{c,t}$ after averaging over $g_t$. Appendix~\appproofs{} gives "
        r"the proof."),
    "thm:freq": (
        r"Each outcome is selected on a half-open interval, so the frequency error is "
        r"bounded by the extreme discrepancy of the point set; that discrepancy is "
        r"invariant under the rotation by $\phi_c$ up to a constant factor and "
        r"satisfies $ND_N^*=O(\log(N+1))$ for the golden ratio. "
        r"Appendix~\appproofs{} gives the proof."),
}

STMT_RE = re.compile(
    r"\\begin\{(proposition|theorem)\}\[([^\]]*)\]\s*\n\\label\{([^}]+)\}\s*\n(.*?)"
    r"\\end\{\1\}", re.S)
stmts = [s for s in STMT_RE.findall(body) if s[2] in MAIN_NUMBER]
assert [s[2] for s in stmts] == PROOF_ORDER, [s[2] for s in stmts]

PROOF_RE = re.compile(r"\\begin\{proof\}\s*\n(.*?)\\end\{proof\}\n", re.S)
proofs = PROOF_RE.findall(body)
assert len(proofs) == 4, len(proofs)

_seq = iter(PROOF_ORDER)
body = PROOF_RE.sub(lambda m: SKETCH[next(_seq)] + "\n", body)
assert r"\begin{proof}" not in body

# Notation the restated Proposition 3 needs to stand on its own.
LEADIN = {
    "prop:biasbound": (
        "Here $g_t$ is the bounded downstream scalar contribution at node $c$ and "
        "iteration $t$, $R_{c,t}$ is the event that the traversal reaches $c$, "
        "$\\bar g_t=\\mathbb{E}_{o\\sim f_c}[g_t(o)]$, and\n"
        "\\[\nB_c(t):=\\mathbb{E}\\!\\left[g_t\\!\\left(\\Phi_c^{-1}(u_{c,N_c(t)})"
        "\\right)\\,\\middle|\\,R_{c,t}\\right]-\\mathbb{E}[\\bar g_t\\mid R_{c,t}].\n"
        "\\]\n\n"),
}

proof_sec = ["\\section{Omitted Proofs}\n\\label{app:proofs}\n\n"
             "This appendix proves the results stated in the main text. Each result "
             "is restated with its main-text number, and the notation is that of "
             "Sections~4 and~5 of the main text.\n\n"]
for (env, title, label, stmt), pf in zip(stmts, proofs):
    supp_env = "proposition" if env == "proposition" else "mainthm"
    proof_sec.append(LEADIN.get(label, ""))
    proof_sec.append("\\begin{%s}[%s]\n%s\\end{%s}\n\n\\begin{proof}\n%s\\end{proof}\n\n"
                     % (supp_env, title, stmt, supp_env, pf))

assert appx.startswith("\\appendix\n")
appx = "\\appendix\n\n" + "".join(proof_sec) + appx[len("\\appendix\n"):]

# ------------------------------------------------------------- Algorithm 1 --
ALGO = r"""
\begin{algorithm}[t]
\caption{CCS-MCCFR: one draw at a concrete chance node.}
\label{alg:weyl}
\begin{algorithmic}[1]
\STATE \textbf{Constant:} $g=(\sqrt{5}-1)/2=0.6180339887498949$
\STATE \textbf{Persistent state:} phase table $\phi[\cdot]$, empty at the start of a run
\STATE \textbf{Input:} chance-node key $c$; outcome probabilities $(p_1,\ldots,p_m)$
\IF{$c\notin\phi$}
\STATE $\phi[c]\leftarrow$ fresh draw from $\mathrm{Uniform}[0,1)$
\ENDIF
\STATE $u\leftarrow\phi[c]$
\STATE $\phi[c]\leftarrow(u+g)\bmod 1$ \quad\COMMENT{advance the persistent stream}
\STATE $F_k\leftarrow\sum_{j\le k}p_j$ for $k=1,\ldots,m$, then set $F_m\leftarrow 1$
\STATE \textbf{return} $\min\{k: u<F_k\}$ \quad\COMMENT{quantile map $\Phi_c^{-1}(u)$}
\end{algorithmic}
\end{algorithm}

"""

_algo_anchor = (r"the quantile rule selects outcome $k$ when "
                r"$\sum_{j<k}p_j\le u_{c,N_c}<\sum_{j\le k}p_j$." + "\n")
assert body.count(_algo_anchor) == 1
body = body.replace(
    _algo_anchor,
    _algo_anchor.rstrip("\n") + " Algorithm~\\ref{alg:weyl} summarizes the "
    "per-node procedure.\n" + ALGO)

# Float widths for the two-column layout: the two 2x2 panel plots stay inside
# one column, and only the wide 1x3 scaling strip spans both columns.
body = body.replace(r"\includegraphics[width=0.8\textwidth]",
                    r"\includegraphics[width=0.95\columnwidth]")


def fix_figure(match: "re.Match") -> str:
    inner = match.group(1)
    return r"\begin{figure}[tb]" + inner + r"\end{figure}"


body = re.sub(r"\\begin\{figure\}\[t\](.*?)\\end\{figure\}", fix_figure, body, flags=re.S)


SPAN_TABLES = (r"\label{tab:main}", r"\label{tab:bigleduc}")


def fix_table(match: "re.Match") -> str:
    """Set every table at the body font size, spanning both columns when wide.

    The single-column source scaled two of these to the 20cm text block, which
    in a 9pt two-column layout blows the type up well past the surrounding
    text. Dropping the \\resizebox and the \\small keeps the tabular at its
    natural size; the two wide ones (six-column endpoints, five-column deck
    scan) get a \\table* so they still fit at that size.
    """
    inner = match.group(1)
    inner = inner.replace("\\resizebox{\\textwidth}{!}{%\n", "")
    inner = inner.replace("\\end{tabular}}", "\\end{tabular}")
    if any(lbl in inner for lbl in SPAN_TABLES):
        inner = inner.replace("\\small\n", "")
        return r"\begin{table*}[t]" + inner + r"\end{table*}"
    # The composition grid stays in one column (a table* could only float to
    # the top of the following page, past the references). It keeps \small and
    # tightens the column separation so it fits without any rescaling.
    inner = inner.replace("\\label{tab:orthogonal}\n",
                          "\\label{tab:orthogonal}\n"
                          "\\setlength{\\tabcolsep}{2pt}\n")
    return r"\begin{table}[tb]" + inner + r"\end{table}"


body = re.sub(r"\\begin\{table\}\[h\](.*?)\\end\{table\}", fix_table, body, flags=re.S)

# ------------------------------------------------- spanning-caption trimming --
# A line of a table* caption costs two lines of body text. These two captions
# lose only wording that the surrounding text already carries; every number,
# sample size, and significance criterion stays.
CAPTION_TRIM = [
    (r"Exploitability is reported as mean $\pm 1.96$\,SEM over seeds. The "
     r"CCS-MCCFR and Antithetic reduction columns are both relative to the same "
     r"vanilla baseline. Significance is assessed using 10{,}000 resample paired "
     r"bootstrap 95\% intervals; a result is significant iff its interval clears "
     r"zero. In configurations with a detected CCS-MCCFR gain, the simple antithetic "
     r"construction does not reproduce its magnitude.",
     r"Exploitability is mean $\pm 1.96$\,SEM over seeds; both reduction columns "
     r"are relative to the same vanilla baseline. Significance is assessed using "
     r"10{,}000 resample paired bootstrap 95\% intervals; a result is significant "
     r"iff its interval clears zero."),
    (r"reduction is the relative drop of the mean (positive $=$ improvement), "
     r"reported as a point estimate. Significance is assessed using the paired "
     r"bootstrap 95\% interval; all reductions are significant (interval clears "
     r"zero; paired Cohen's $d=0.39$ to $0.60$).",
     r"reduction is the relative drop of the mean (positive $=$ improvement). All "
     r"reductions are significant under the paired bootstrap 95\% interval "
     r"(interval clears zero; paired Cohen's $d=0.39$ to $0.60$)."),
]
CAPTION_TRIM += [
    (r"This uses the same statistic and convention as the HUNL reduction figures "
     r"in Appendix~\apphunl{} and is the paired-reduction view of the raw "
     r"exploitability ribbons in Figure~\ref{fig:convergence}.",
     r"It is the paired-reduction view of Figure~\ref{fig:convergence} and uses "
     r"the same statistic as the HUNL figures in Appendix~\apphunl{}."),
    (r"Each chance cell reports exploitability (mean $\pm 1.96$\,SEM); the "
     r"rightmost column is the paired CCS-MCCFR versus i.i.d.\ reduction point "
     r"estimate. Significance is assessed using paired-bootstrap 95\% intervals.",
     r"Cells are exploitability (mean $\pm 1.96$\,SEM); the last column is the "
     r"paired CCS-MCCFR versus i.i.d.\ reduction, significant under "
     r"paired-bootstrap 95\% intervals."),
]

CAPTION_TRIM += [
    # One line of the closing sentence, same claim.
    (r"already retains the standard $O(1/\sqrt{T})$ guarantee, which isolates a "
     r"precise next question for the theory of adaptive correlated sampling.",
     r"already retains the standard $O(1/\sqrt{T})$ guarantee, isolating the next "
     r"question for adaptive correlated sampling."),
    # Spelling out both acronyms on first use costs one line; these two clauses
    # give it back without dropping a number, a claim, or a citation.
    (r"The method achieves this as a minimal sampler change that leaves the chance "
     r"law, the External-Sampling estimator, and the MCCFR regret updates intact.",
     r"It is a minimal sampler change that leaves the chance law, the "
     r"External-Sampling estimator, and the MCCFR regret updates intact."),
    (r"and 27.65\%, 34.01\%, and 19.05\% across a controlled 6/10/12-card Leduc "
     r"family, with all paired-bootstrap intervals above zero.",
     r"and 27.65\%, 34.01\%, and 19.05\% across a 6/10/12-card Leduc family, with "
     r"all paired-bootstrap intervals above zero."),
    (r"and fixed-index marginal correctness together with fixed-trajectory "
     r"unbiasedness supply complementary local guarantees.",
     r"while fixed-index marginal correctness and fixed-trajectory unbiasedness "
     r"supply complementary local guarantees."),
    (r"The Leduc benefit remains 24.5\% at 3M node touches, and combining "
     r"CCS-MCCFR with Linear CFR produces the lowest measured composition cell",
     r"The Leduc benefit remains 24.5\% at 3M node touches, and CCS-MCCFR with "
     r"Linear CFR gives the lowest measured composition cell"),
    (r"stress tests show no statistically detectable final-endpoint difference, with "
     r"revisit exposure, symmetry, and private-information coupling as the empirical "
     r"moderators.",
     r"stress tests show no statistically detectable endpoint difference, with "
     r"revisit exposure, symmetry, and private-information coupling as the "
     r"moderators."),
    (r"Its first $N$ draws at one concrete chance node, where $N$ counts that node's "
     r"consumed draws rather than global iterations, satisfy deterministic",
     r"Its first $N$ draws at one concrete chance node, where $N$ counts that node's "
     r"draws rather than global iterations, satisfy deterministic"),
]

for _old, _new in CAPTION_TRIM:
    assert body.count(_old) == 1, f"caption anchor not unique: {_old[:50]!r}"
    body = body.replace(_old, _new)


# --------------------------------------------------------- prose de-itemizing --
# Two bulleted lists read as note-form in a two-column paper. Both become running
# prose that carries the same content in the same order; no number, interval, or
# significance criterion changes.
DEITEMIZE = [
    (r"""The observed effect may depend on:
\begin{itemize}
\item \textbf{Chance decision coupling:} When chance is interleaved with decisions and reveals private or public information, its temporally correlated contributions enter many downstream regret differences. Root-only chance also carries this coupling: Kuhn deals private cards once and still yields one of the largest measured gains.
\item \textbf{Outcome symmetry:} Symmetric payoff structure can cancel chance contributions in regret differences, leaving less variation for temporal balancing to affect.
\end{itemize}
""",
     r"""Two structural properties govern how much of the effect survives. The first is
the coupling between chance and decisions: when chance is interleaved with
decisions and reveals private or public information, its temporally correlated
contributions enter many downstream regret differences. Root-only chance also
carries this coupling, since Kuhn deals private cards once and still yields one
of the largest measured gains. The second is outcome symmetry, which works in
the opposite direction: a symmetric payoff structure cancels chance
contributions inside regret differences, leaving less variation for temporal
balancing to act on.
"""),
    (r"""\textbf{Key findings:}
\begin{itemize}
\item \textbf{Poker games (Kuhn, Leduc):} CCS-MCCFR produces the largest main-benchmark reductions, 27.64\% and 24.59\%, with paired-bootstrap CIs well above zero. Both games repeatedly revisit private-information chance allocations under the node-touch budget; Leduc also interleaves chance and decisions.
\item \textbf{Goofspiel:} The 4.27\% reduction is smaller but significant ($4.3\%\,{\pm}\,1.1\%$, paired-bootstrap 95\% CI), extending the detected effect beyond poker while illustrating attenuation in a symmetric chance structure (Observation~\ref{obs:coupling}).
\item \textbf{Liar's Dice:} The 200-seed experiment precisely localizes an endpoint near zero: CCS-MCCFR and antithetic reductions are $-0.19\%$ and $0.03\%$, with both paired intervals crossing zero. This high-power boundary contrasts with Kuhn, which also places private chance at the root, and therefore rules out root-only placement as a sufficient explanation of the cross-game pattern.
\end{itemize}
""",
     r"""The two poker games carry the largest main-benchmark reductions, 27.64\% on
Kuhn and 24.59\% on Leduc, with paired-bootstrap CIs well above zero; both
repeatedly revisit private-information chance allocations under the node-touch
budget, and Leduc additionally interleaves chance and decisions. Goofspiel
attenuates the effect without removing it: the 4.27\% reduction is smaller but
still significant ($4.3\%\,{\pm}\,1.1\%$, paired-bootstrap 95\% CI), which
extends the detected effect beyond poker in exactly the symmetric chance
structure anticipated by Observation~\ref{obs:coupling}. Liar's Dice then
localizes an endpoint near zero at high power: over 200 seeds the CCS-MCCFR and
antithetic reductions are $-0.19\%$ and $0.03\%$, with both paired intervals
crossing zero. That boundary is informative because Kuhn also places private
chance at the root, so root-only placement cannot by itself explain the
cross-game pattern.
"""),
]
for _old, _new in DEITEMIZE:
    assert body.count(_old) == 1, f"de-itemize anchor not unique: {_old[:50]!r}"
    body = body.replace(_old, _new)


# --------------------------------------------------- narrow display formulas --
# Two displays from the theory section are set for a 20cm single-column line and
# overflow a 9pt AAAI column. Both appear verbatim in the main text and again in
# the restated statements of the supplementary proof appendix.
NARROW_MATH = [
    # delta_{c,t} definition: drop the padding thin spaces and shrink the fences.
    (r"\delta_{c,t}:=\mathbb{E}_{g_t}\!\Big[\,\mathrm{TV}\!"
     r"\big(\mathrm{Law}(u_{c,N_c(t)}\mid g_t,R_{c,t}),\,\mathrm{Unif}[0,1)\big)"
     r"\,\Big|\,R_{c,t}\Big]",
     r"\delta_{c,t}:=\mathbb{E}_{g_t}\!\big[\mathrm{TV}\!"
     r"\big(\mathrm{Law}(u_{c,N_c(t)}\mid g_t,R_{c,t}),\mathrm{Unif}[0,1)\big)"
     r"\,\big|\,R_{c,t}\big]"),
    # Theorem 1: the two frequency bounds are stacked instead of set side by side.
    ("\\max_{1\\le k\\le m}\\big|\\hat p_k^{(N)}-p_k\\big| \\;\\le\\; "
     "\\frac{C\\log(N+1)}{N},\n\\qquad\n"
     "\\sum_{k=1}^m\\big|\\hat p_k^{(N)}-p_k\\big| \\;\\le\\; "
     "\\frac{C\\,m\\log(N+1)}{N},",
     "\\begin{gathered}\n"
     "\\max_{1\\le k\\le m}\\big|\\hat p_k^{(N)}-p_k\\big| \\;\\le\\; "
     "C\\log(N+1)/N,\\\\[-1pt]\n"
     "\\textstyle\\sum_{k=1}^m\\big|\\hat p_k^{(N)}-p_k\\big| \\;\\le\\; "
     "C\\,m\\log(N+1)/N,\n"
     "\\end{gathered}"),
]
for _old, _new in NARROW_MATH:
    assert body.count(_old) == 1, f"narrow-math anchor not unique in body: {_old[:50]!r}"
    assert appx.count(_old) == 1, f"narrow-math anchor not unique in appx: {_old[:50]!r}"
    body = body.replace(_old, _new)
    appx = appx.replace(_old, _new)

# ------------------------------------------------------- two-column tightening --
# The arXiv version has room to restate results; the seven-page submission does
# not. Each rewrite below removes a restatement of a number that a table or an
# earlier sentence already gives, or a duplicated appendix pointer. No number,
# interval, sample size, or significance verdict is changed or dropped.
TIGHTEN = [
    # Main-results discussion repeats the two headline numbers that the preceding
    # paragraph and Table 1 both state.
    ("The two poker games carry the largest main-benchmark reductions, 27.64\\% on\n"
     "Kuhn and 24.59\\% on Leduc, with paired-bootstrap CIs well above zero; both\n"
     "repeatedly revisit private-information chance allocations under the node-touch\n"
     "budget, and Leduc additionally interleaves chance and decisions. Goofspiel\n"
     "attenuates the effect without removing it: the 4.27\\% reduction is smaller but\n"
     "still significant ($4.3\\%\\,{\\pm}\\,1.1\\%$, paired-bootstrap 95\\% CI), which\n"
     "extends the detected effect beyond poker in exactly the symmetric chance\n"
     "structure anticipated by Observation~\\ref{obs:coupling}. Liar's Dice then\n"
     "localizes an endpoint near zero at high power: over 200 seeds the CCS-MCCFR and\n"
     "antithetic reductions are $-0.19\\%$ and $0.03\\%$, with both paired intervals\n"
     "crossing zero. That boundary is informative because Kuhn also places private\n"
     "chance at the root, so root-only placement cannot by itself explain the\n"
     "cross-game pattern.",
     r"Both poker games repeatedly revisit private-information chance allocations "
     r"under the node-touch budget, and Leduc additionally interleaves chance and "
     r"decisions. Goofspiel attenuates the effect without removing it: its reduction "
     r"is smaller but still significant ($4.3\%\,{\pm}\,1.1\%$, paired-bootstrap 95\% "
     r"CI), extending the detected effect beyond poker in exactly the symmetric chance "
     r"structure anticipated by Observation~\ref{obs:coupling}. Liar's Dice localizes "
     r"an endpoint near zero at high power, with both paired intervals crossing zero. "
     r"That boundary is informative because Kuhn also places private chance at the "
     r"root, so root-only placement cannot by itself explain the cross-game pattern."),
    # The antithetic column is already labelled in Table 1 and its numbers are in it.
    (r"\textbf{Antithetic control.} The antithetic column of Table~\ref{tab:main} "
     r"evaluates a simpler paired correlation scheme, not CCS-MCCFR, across all four "
     r"small games. It does not reproduce the magnitude of the CCS-MCCFR gains in the "
     r"configurations where a gain is detected (e.g.\ Kuhn $-0.5\%$ vs.\ CCS-MCCFR "
     r"$27.6\%$). On Liar's Dice, both constructions remain centered near the same "
     r"vanilla endpoint.",
     r"\textbf{Antithetic control.} The antithetic column of Table~\ref{tab:main} "
     r"evaluates a simpler paired correlation scheme, not CCS-MCCFR. It does not "
     r"reproduce the magnitude of the CCS-MCCFR gains wherever a gain is detected, "
     r"and on Liar's Dice both constructions stay near the same vanilla endpoint."),
    # Conclusion restates the full results table; the pointer-free summary suffices.
    (r"CCS-MCCFR substantially reduces exploitability across tabular poker: 27.64\% "
     r"on Kuhn, 24.59\% on standard Leduc, and 27.65\%, 34.01\%, and 19.05\% across a "
     r"6/10/12-card Leduc family, with all paired-bootstrap intervals above zero. The "
     r"Leduc benefit remains 24.5\% at 3M node touches, and CCS-MCCFR with Linear CFR "
     r"gives the lowest measured composition cell, 42.59\% below vanilla updates with "
     r"i.i.d.\ chance. Goofspiel-4 adds a smaller significant 4.27\% gain. Liar's "
     r"Dice, the reduced Flop budget scan, Goofspiel-5, and four HUNL transfer stress "
     r"tests show no statistically detectable endpoint difference, with revisit "
     r"exposure, symmetry, and private-information coupling as the moderators.",
     r"CCS-MCCFR substantially reduces exploitability across tabular poker: 27.64\% "
     r"on Kuhn, 24.59\% on standard Leduc, and 19.05\% to 34.01\% across a 6/10/12-card "
     r"Leduc family, with all paired-bootstrap intervals above zero. The Leduc benefit "
     r"remains 24.5\% at 3M node touches, and CCS-MCCFR with Linear CFR gives the "
     r"lowest measured composition cell. Goofspiel-4 adds a smaller significant gain, "
     r"while Liar's Dice, reduced Flop Hold'em, Goofspiel-5, and four HUNL transfer "
     r"stress tests show no detectable endpoint difference, with revisit exposure, "
     r"symmetry, and private-information coupling as the moderators."),
    # The composition paragraph names the same appendix twice.
    (r"reaching 28.69\% on Kuhn and 25.64\% on Leduc; Appendix~\appcomposition{} "
     r"reports the design, the endpoints, and what the comparison establishes.",
     r"reaching 28.69\% on Kuhn and 25.64\% on Leduc."),
    # Conclusion's second paragraph restates the three theory contributions that the
    # Intro bullet and Section 5 already state; one sentence carries the takeaway.
    (r"It is a minimal sampler change that leaves the chance law, the "
     r"External-Sampling estimator, and the MCCFR regret updates intact. Its first "
     r"$N$ draws at one concrete chance node, where $N$ counts that node's draws "
     r"rather than global iterations, satisfy deterministic $O(\!\log(N+1)/N)$ "
     r"frequency discrepancy, while fixed-index marginal correctness and "
     r"fixed-trajectory unbiasedness supply complementary local guarantees. The "
     r"adaptive phase-selection bound names the single dependence term a global "
     r"analysis must control, and per-traversal resetting already retains the "
     r"standard $O(1/\sqrt{T})$ guarantee, isolating the next question for adaptive "
     r"correlated sampling.",
     r"It achieves this as a minimal sampler change that leaves the chance law, the "
     r"External-Sampling estimator, and the MCCFR regret updates intact, with "
     r"guarantees stated per concrete node rather than per global iteration, and its "
     r"adaptive phase-selection bound names the single dependence term that a global "
     r"analysis must control."),
    # The revisit paragraph opens with a rhetorical question answered two lines later.
    (r"What distinguishes the strongest gain cases from the small effect and no gain "
     r"cases? Our hypothesis suggests that repeated visits expose more cross-visit "
     r"temporal structure for possible cancellation. Table~S2 in "
     r"Appendix~\apprevisit{} reports this revisit diagnostic, and that appendix "
     r"gives its measurement procedure.",
     r"Repeated visits should expose more cross-visit temporal structure for "
     r"cancellation, which suggests a diagnostic that separates the strongest gains "
     r"from the null cases. Table~S2 in Appendix~\apprevisit{} reports it together "
     r"with its measurement procedure."),
    # The HUNL paragraph points at the same appendix twice in one sentence.
    (r"Appendix~\apphunl{} reports the full protocol, solver-revision scope, endpoint "
     r"tables, and Turn/River trajectories; Table~S2 and Figure~S1 add the "
     r"corresponding concrete-node exposure diagnostics.",
     r"Appendix~\apphunl{} reports the full protocol, solver-revision scope, endpoint "
     r"tables, and Turn/River trajectories."),
    # "N is not T" is stated in Motivation, restated after Theorem 1, and restated
    # again in the Conclusion. Keep the first two, drop the third clause here.
    (r"\textbf{What the theorem controls.} The guarantee is per-node and per-stream: "
     r"it bounds the unweighted outcome counts in the first $N$ draws consumed by one "
     r"concrete node, where $N$ is that node's visit count rather than the global "
     r"iteration count $T$. Because the bound holds uniformly over phases, pauses "
     r"between visits leave the consumed prefix and its discrepancy unchanged, which "
     r"is what makes a stream persistent across iterations well behaved.",
     r"\textbf{What the theorem controls.} The guarantee is per-node and per-stream, "
     r"bounding unweighted outcome counts over one node's own visits. Because it holds "
     r"uniformly over phases, pauses between visits leave the consumed prefix and its "
     r"discrepancy unchanged, which is what makes a persistent stream well behaved."),
    # The golden-ratio rationale appears once in Section 4 and again verbatim in the
    # discrepancy paragraph; the second occurrence keeps the citation and the bound.
    (r"Kronecker sequences $\{ng \bmod 1\}$ with a badly approximable rotation number "
     r"satisfy $N D_N^* = O(\log N)$, and the golden ratio, whose continued fraction "
     r"is $[0;1,1,1,\ldots]$, attains the most favorable constant "
     r"\citep{niederreiter1992random}. This yields the following deterministic frequency "
     r"guarantee for our per node streams.",
     r"Kronecker sequences $\{ng \bmod 1\}$ with a badly approximable rotation number "
     r"satisfy $N D_N^* = O(\log N)$, and the golden ratio attains the most favorable "
     r"constant \citep{niederreiter1992random}, which yields the following deterministic "
     r"per-node frequency guarantee."),
    # Motivation's closing sentence duplicates the italicized idea box that follows.
    (r"When downstream per-outcome values are fixed or effectively oblivious to the "
     r"sampled phase, this frequency error propagates linearly into the corresponding "
     r"chance-node value estimates. The fully adaptive case, where those values "
     r"themselves evolve with the sampled outcomes, is analyzed separately below.",
     r"When downstream per-outcome values are oblivious to the sampled phase, this "
     r"frequency error propagates linearly into the chance-node value estimates; the "
     r"fully adaptive case is analyzed below."),
    # The Leduc-family paragraph restates the paired protocol that Experimental Setup
    # already fixes for every experiment, including its n=50 seed count.
    ("This preserves the chance/decision and private-information structure while "
     "expanding the outcome space.\n\nWe use a paired design in which the same seed "
     "drives vanilla and CCS-MCCFR runs, with $n=50$ seeds and a 10{,}000 resample "
     "bootstrap confidence interval on the relative reduction. A result is positive "
     "when this interval clears zero.",
     r"This preserves the chance/decision and private-information structure while "
     r"expanding the outcome space, under the same paired protocol as above."),
    # The ablation's interpretive hedge is already the point of the sentence before it.
    (r"This supports across-iteration phase coupling as a contributor to the measured "
     r"gains, while leaving the exact covariance pathway for future identification. A "
     r"distinct per-traversal reset variant has the standard $O(1/\sqrt{T})$ External "
     r"Sampling guarantee.",
     r"This supports across-iteration phase coupling as a contributor to the measured "
     r"gains. A distinct per-traversal reset variant has the standard "
     r"$O(1/\sqrt{T})$ External Sampling guarantee."),
    # Section 5 previews the revisit diagnostic with the same three cases that
    # Section 6 then reports from the measurement; the preview keeps the mechanism
    # and the pointer, and the numbers stay where they are measured.
    (r"\textbf{The revisit diagnostic.} CCS-MCCFR spreads a node's outcomes across "
     r"its repeated visits, so the length of a node's stream measures how much "
     r"cross-visit structure the sampler has to work with. A node visited once offers "
     r"none. In the tested set, Kuhn and Leduc have tens to hundreds of visits and "
     r"show substantial gains, whereas a reduced Flop Hold'em has 75\% of chance nodes "
     r"visited once and shows no gain (Table~S2, Appendix~\apprevisit{}). Symmetric "
     r"Goofspiel is revisited often and gains less, which is where the second feature "
     r"enters.",
     r"\textbf{The revisit diagnostic.} The length of a node's stream measures how much "
     r"cross-visit structure the sampler has to work with, and a node visited once "
     r"offers none. Section~\ref{sec:revisit-exp} measures this exposure across games."),
    # Observations 1 and 2 state the same three moderators. Section 5 keeps the one
    # the experiments cite (obs:coupling) and folds the other into running prose, so
    # no moderator and no appendix pointer is lost.
    ("\\begin{observation}[Scope of Effectiveness]\n"
     "\\label{obs:scope}\n"
     "In our tested tabular games, favorable effects are associated with material "
     "chance contributions to regret differences, repeated per-node visits, and "
     "private-information coupling. These are empirical correlates of the proposed "
     "temporal-cancellation mechanism (Appendix~\\appscope{}).\n"
     "\\end{observation}",
     "These are empirical correlates of the proposed temporal-cancellation "
     "mechanism rather than a characterization of it (Appendix~\\appscope{})."),
    # Intro already lists the deck-family range in the contribution bullet.
    (r"CCS-MCCFR lowers final exploitability by 27.64\% on Kuhn, 24.59\% on standard "
     r"Leduc, and 27.65\%, 34.01\%, and 19.05\% in a controlled 6/10/12-card Leduc "
     r"deck expansion, with all corresponding paired-bootstrap confidence intervals "
     r"above zero.",
     r"CCS-MCCFR lowers final exploitability by 27.64\% on Kuhn, 24.59\% on standard "
     r"Leduc, and by 19.05\% to 34.01\% in a controlled 6/10/12-card Leduc deck "
     r"expansion, with all corresponding paired-bootstrap confidence intervals above "
     r"zero."),
]
# Second tightening pass, needed after the related-work citation additions.
TIGHTEN += [
    # The Conclusion restates every headline number that Table 1 and the Main
    # Results paragraph already give.
    (r"CCS-MCCFR substantially reduces exploitability across tabular poker: 27.64\% "
     r"on Kuhn, 24.59\% on standard Leduc, and 19.05\% to 34.01\% across a "
     r"6/10/12-card Leduc family, with all paired-bootstrap intervals above zero. The "
     r"Leduc benefit remains 24.5\% at 3M node touches, and CCS-MCCFR with Linear CFR "
     r"gives the lowest measured composition cell. Goofspiel-4 adds a smaller "
     r"significant gain, while Liar's Dice, reduced Flop Hold'em, Goofspiel-5, and "
     r"four HUNL transfer stress tests show no detectable endpoint difference, with "
     r"revisit exposure, symmetry, and private-information coupling as the moderators.",
     r"CCS-MCCFR substantially reduces exploitability across tabular poker, holds that "
     r"benefit at 3M node touches, and composes with Linear CFR. Goofspiel-4 adds a "
     r"smaller significant gain, while Liar's Dice, reduced Flop Hold'em, Goofspiel-5, "
     r"and four HUNL transfer stress tests show no detectable endpoint difference, with "
     r"revisit exposure, symmetry, and private-information coupling as the moderators."),
    # Four number-restatement trims that used to live here (Table 3 composition
    # cells, the three per-deck reductions, the four HUNL endgame point
    # estimates, and the Goofspiel-4 endpoint) were reverted once dropping two
    # related-work citations freed roughly six lines on page 7.  The body now
    # carries those numbers again; only the two trims below remain, because the
    # Conclusion restating every headline number and the deck-family lead-in
    # repeating the preceding sentence are redundant regardless of page budget.
    # The preceding sentence already says the deck grows while structure is fixed.
    (r"The main experiments establish strong gains on standard Kuhn and Leduc. We next "
     r"isolate deck size in a controlled Leduc-style limit-poker family implemented "
     r"with OpenSpiel's \texttt{universal\_poker}: the structure remains fixed (one "
     r"private hole card, one public board card, two betting rounds, fold/call "
     r"betting), while the deck grows from 6 to 10 and 12 cards. This preserves the "
     r"chance/decision and private-information structure while expanding the outcome "
     r"space, under the same paired protocol as above.",
     r"We next isolate deck size in a controlled Leduc-style limit-poker family "
     r"implemented with OpenSpiel's \texttt{universal\_poker}: the structure remains "
     r"fixed (one private hole card, one public board card, two betting rounds, "
     r"fold/call betting), while the deck grows from 6 to 10 and 12 cards. This "
     r"expands the outcome space while preserving the chance/decision and "
     r"private-information structure, under the paired protocol above."),
]
for _old, _new in TIGHTEN:
    assert body.count(_old) == 1, f"tighten anchor not unique: {_old[:60]!r}"
    body = body.replace(_old, _new)

(OUT / "main.tex").write_text(MAIN_HEAD + body + TAIL)

# ---------------------------------------------------- supplementary rewrites --
SUPP_EXACT = [
    (r"and Observations~\ref{obs:scope} and~\ref{obs:coupling} are descriptive summaries",
     r"and Observations~1 and~2 of the main text are descriptive summaries"),
    (r"in Section~\ref{sec:theory} is an interpretive approximation",
     r"in Section~5 of the main text is an interpretive approximation"),
    (r"This quantity is distinct from the frequency discrepancy of Theorem~\ref{thm:freq}",
     r"This quantity is distinct from the frequency discrepancy of Theorem~1 of the main text"),
    (r"the persistent randomized Weyl construction in Section~\ref{sec:method}",
     r"the persistent randomized Weyl construction in Section~4 of the main text"),
    (r"Table~\ref{tab:orthogonal} gives the $2\times3$ Leduc grid.",
     r"Table~3 of the main text gives the $2\times3$ Leduc grid."),
]
for old, new in SUPP_EXACT:
    assert appx.count(old) == 1, f"supp exact-replacement not unique: {old[:60]!r}"
    appx = appx.replace(old, new)

BACK = {"prop:unbiased": "1", "prop:traj": "2", "prop:biasbound": "3",
        "thm:freq": "1", "obs:scope": "1", "obs:coupling": "2",
        "sec:method": "4", "sec:theory": "5"}
appx = re.sub(r"\\ref\{(" + "|".join(BACK) + r")\}", lambda m: BACK[m.group(1)], appx)

appx += (
    "\n\\section{Leduc-Family Convergence Trajectories}\n\\label{app:curves}\n\n"
    "Table~2 of the main text reports the endpoints, reductions, and paired tests "
    "for the controlled 6/10/12-card Leduc family. "
    "Figure~\\ref{fig:bigleduc} plots the corresponding convergence "
    "trajectories at full page width.\n\n" + bigleduc_fig)

# ------------------------------------------- supplementary two-column layout --
# The appendix floats were sized for a single 20cm column. Spanning them keeps
# the panel plots and the wide endpoint tables readable and inside the text
# block; the supplementary has no page budget, so nothing needs to shrink.
_SUPP_DISPLAY = (
    "\\big|\\,\\mathbb{E}[g_t(\\Phi_c^{-1}(u_{c,N_c(t)}))-\\bar g_t"
    "\\mid g_t=\\hat g,\\,R_{c,t}]\\,\\big|\n"
    "=\\big|\\mathbb{E}_{P_{\\hat g}}[\\psi]-\\mathbb{E}_{\\mathrm{Unif}}[\\psi]\\big|\n"
    "\\le 2\\|\\psi\\|_\\infty\\,\\mathrm{TV}(P_{\\hat g},\\mathrm{Unif}).")
assert appx.count(_SUPP_DISPLAY) == 1, "phase-selection display not found"
appx = appx.replace(_SUPP_DISPLAY,
                    "\\begin{gathered}\n"
                    "\\big|\\,\\mathbb{E}[g_t(\\Phi_c^{-1}(u_{c,N_c(t)}))-\\bar g_t"
                    "\\mid g_t=\\hat g,\\,R_{c,t}]\\,\\big|\\\\\n"
                    "=\\big|\\mathbb{E}_{P_{\\hat g}}[\\psi]"
                    "-\\mathbb{E}_{\\mathrm{Unif}}[\\psi]\\big|\n"
                    "\\le 2\\|\\psi\\|_\\infty\\,\\mathrm{TV}(P_{\\hat g},\\mathrm{Unif}).\n"
                    "\\end{gathered}")

for _env in ("figure", "table"):
    for _pos in ("[t]", "[h]"):
        appx = appx.replace("\\begin{%s}%s" % (_env, _pos), "\\begin{%s*}[t]" % _env)
    appx = appx.replace("\\end{%s}" % _env, "\\end{%s*}" % _env)
appx = appx.replace("\\begin{figure**}[t]", "\\begin{figure*}[t]")
appx = appx.replace("\\end{figure**}", "\\end{figure*}")
assert "\\begin{figure}" not in appx and "\\begin{table}" not in appx

leftover = set(re.findall(r"\\ref\{([^}]+)\}", appx))
allowed = {"app:scope", "app:future", "app:experiment-protocol", "app:reset-proof",
           "app:revisit-protocol", "app:composition-details", "app:es-control-variate",
           "app:hunl-details", "app:extra", "app:adaptive-frequency",
           "thm:reset", "tab:persistence", "tab:revisit", "fig:revisit",
           "tab:es-control-variate", "tab:hunl-all", "fig:hunl", "fig:hunl-river",
           "fig:dcfr-accel", "fig:stack-grid", "fig:bigleduc"}
assert leftover <= allowed, f"unresolved supplementary refs: {leftover - allowed}"

(OUT / "supplementary.tex").write_text(SUPP_HEAD + appx + TAIL)
print("wrote", OUT / "main.tex", "and", OUT / "supplementary.tex")
