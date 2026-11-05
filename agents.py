#agents.py
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import web_search, scrape_url
from paper_tools import arxiv_search, semantic_scholar_search
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# MODEL SETUP
# ============================================================
# num_ctx=8192 added: Ollama defaults to a 2048-token context window
# regardless of what the model supports, unless told otherwise. That
# was silently truncating the combined research context passed into
# the Writer/Literature/Gap chains. Raise further if your RAM allows.

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    num_ctx=8192
)


# ============================================================
# 1. RESEARCH AGENT
# ============================================================

def build_research_agent():
    return create_agent(
        model=llm,
        tools=[
            arxiv_search,
            semantic_scholar_search,
            web_search
        ],
    )


# ============================================================
# 2. READER AGENT
# ============================================================

def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[
            scrape_url
        ],
    )


# ============================================================
# 3. WRITER CHAIN
# ============================================================

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert scientific research writer.

STRICT SOURCE AND EVIDENCE RULES:

1. Use ONLY information provided in the research context.

2. Never invent:
   - papers
   - authors
   - datasets
   - experiments
   - findings
   - statistics
   - URLs
   - citations

3. Never create or guess a URL.

4. Never modify a URL.

5. If information is not available in the research context,
   write:

   "Not specified in the provided evidence."

6. Do not treat a hypothesis as an established research finding.

7. Clearly distinguish:
   - Existing evidence
   - Research limitations
   - Potential research gaps
   - Proposed hypotheses

8. Sources must contain ONLY URLs explicitly present in the
   research context.

9. Do not use fictional sources.

10. Do not strengthen a claim beyond the evidence provided.

11. Preserve hypothesis IDs exactly as given (H1, H2, H3, etc.).
    Never substitute the content of one hypothesis for another.
    Never merge H1, H2, and H3 into a single hypothesis.
    Every hypothesis number provided in the input must appear as its
    own clearly labeled subsection in the output.

12. Every claim in the report must be traceable to the corresponding
    experiment design, contradiction analysis, or verified evidence
    supplied — do not blend content across hypotheses.

13. Do not present a predicted/expected outcome as an actual finding.
"""
    ),
    (
        "human",
        """
Write a research report on:

Topic:
{topic}

Research Gathered:
{research}

SStructure:

1. Title
2. Abstract
3. Introduction
4. Literature Review
5. Research Gaps
6. Research Objectives
7. Original Hypotheses (H1, H2, H3 — each as its own labeled subsection)
8. Contradiction Analysis (per hypothesis)
9. Evolved Hypotheses (per hypothesis)
10. Proposed Experimental Designs (per hypothesis)
11. Experiment Critiques and Required Corrections (per hypothesis)
12. Limitations
13. Conclusion
14. Verified Sources

IMPORTANT:

The Verified Sources section must contain ONLY URLs that
actually appear in the provided research context.

Do not invent or guess any source.

Do not present proposed hypotheses as established facts.

Be factual, evidence-based, and professional.
"""
    )
])

writer_chain = (
    writer_prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# 4. CRITIC CHAIN
# ============================================================
# UPDATED: the critic now receives the programmatically-verified URL
# list directly, so it stops guessing whether a source "looks real"
# and incorrectly flagging genuine, already-verified arXiv URLs as
# fictional/hallucinated (this happened in a real run — the critic
# has no way to check a URL itself, it only sees report text).

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a strict scientific research critic.

Review the report for:

- factual consistency
- evidence grounding
- unsupported claims
- hallucinated sources
- logical consistency
- quality of research gaps
- quality of hypotheses
- contradiction analysis
- clarity
- completeness

Do not assume that a statement is correct simply because
it appears in the report.

If the report makes a claim that is not supported by the
provided research, identify it.

IMPORTANT - SOURCE VERIFICATION RULE:

A list of already-verified URLs is provided to you separately below.
These URLs were confirmed directly against live academic search APIs
(arXiv / Semantic Scholar) BEFORE the report was written - they are
guaranteed real. Do NOT flag any URL that appears in that verified
list as fictional, fabricated, or hallucinated, even if the URL
format looks unfamiliar to you or you do not personally recognize
the paper. Only flag a URL as a hallucination issue if it does NOT
appear in the verified list below.
"""
    ),
    (
        "human",
        """
Review the research report below.

Report:
{report}

Already-Verified Source URLs (confirmed real - do NOT flag these as
fictional or hallucinated):
{verified_urls}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

Evidence / Hallucination Issues:
- ...
- ...

One line verdict:
...
"""
    ),
])

critic_chain = (
    critic_prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# 5. LITERATURE ANALYSIS AGENT
# ============================================================

literature_analysis_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert scientific literature analysis agent.

Your job is to analyze retrieved research literature and identify:

- research methods
- findings
- limitations
- datasets
- experimental setups
- trends
- disagreements
- potential research opportunities

STRICT EVIDENCE RULES:

1. Use ONLY the research context provided.

2. Do NOT invent information.

3. Do NOT assume a dataset, experiment, method, or result
   exists if it is not explicitly mentioned.

4. If a requested piece of information is not available,
   write:

   "Not specified in the provided evidence."

5. Absence of information is NOT evidence that something
   does not exist.

6. Do not treat one study's limitation as a limitation of
   the entire research field.

7. Clearly distinguish between:
   - what the literature explicitly reports
   - what can reasonably be inferred
   - what remains uncertain

8. Do not claim that studies disagree unless their findings
   actually support different conclusions.

Be precise and evidence-based.
"""
    ),
    (
        "human",
        """
Analyze the following research literature.

Research Topic:
{topic}

Retrieved Literature:
{research}

Provide the analysis in this structure:

1. Research Methods
- Identify the main methods, models, or approaches explicitly
  mentioned in the literature.

2. Key Findings
- Summarize major findings explicitly supported by the
  provided literature.

3. Common Limitations
- Identify limitations explicitly reported in the literature.
- Do not invent limitations.

4. Datasets / Experimental Setup
- Identify datasets, experiments, benchmarks, or evaluation
  methods explicitly mentioned.
- If not specified, say:
  "Not specified in the provided evidence."

5. Research Trends
- Identify patterns that can reasonably be observed across
  the provided studies.

6. Areas of Disagreement
- Identify genuine conflicting findings.
- A difference in method or research context is NOT automatically
  a disagreement.
- If no genuine disagreement is present, say so.

7. Potential Research Opportunities
- Identify areas that appear underexplored based on the evidence.
- Do NOT call something a confirmed research gap yet.
- Clearly state when evidence is insufficient.

Keep the analysis factual and concise.
"""
    )
])

literature_analysis_chain = (
    literature_analysis_prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# 6. RESEARCH GAP AGENT
# ============================================================

research_gap_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert scientific Research Gap Discovery Agent.

Your task is to identify meaningful and evidence-supported
research gaps from existing literature.

STRICT EVIDENCE RULES:

1. Use ONLY the literature and analysis provided.

2. Do NOT invent research findings.

3. Do NOT invent limitations.

4. Do NOT assume that something is a research gap simply
   because one study did not mention it.

IMPORTANT:

NOT MENTIONED != RESEARCH GAP.

A missing topic is NOT automatically a research gap.

A potential research gap should be identified only when:

1. The provided literature indicates that the area is
   insufficiently studied,

OR

2. Multiple studies reveal a consistent limitation,

OR

3. The literature shows a missing:
   - comparison
   - dataset
   - benchmark
   - evaluation method
   - experimental condition
   - methodology
   - unresolved research question

OR

4. Different studies produce findings that remain unresolved
   or inconsistent.

If the evidence only shows that a topic was not discussed
in one study, DO NOT call it a research gap.

Instead write:

"Insufficient evidence to establish a research gap."

IMPORTANT CONFIDENCE RULE:

Do NOT assign High confidence to a research gap based only
on absence of discussion.

Use:

High:
Strong evidence from multiple studies or explicit literature
statements supports the gap.

Medium:
Some evidence supports the gap, but additional literature
is required.

Low:
The gap is plausible but evidence is limited.

Insufficient:
The provided evidence does not establish that a gap exists.

FIELD-LEVEL CAUTION:

Do not generalize from one paper to the entire research field.

For example, do NOT say:

"This is a major gap in XAI research"

when the evidence consists of only one study.

Instead say:

"The provided literature suggests a potential gap..."

Do not claim that a topic is completely unexplored unless
the evidence supports that claim.

Clearly distinguish between:

- Confirmed limitation
- Missing information
- Potential research gap
- Evidence
- Confidence
"""
    ),
    (
        "human",
        """
Find potential research gaps in the following research.

Research Topic:
{topic}

Literature Analysis:
{literature_analysis}

Retrieved Research:
{research}

Provide the analysis in this structure:

1. Confirmed Limitations

List only limitations explicitly reported or strongly
supported by the provided literature.

If none are available:

"Not specified in the provided evidence."

2. Missing Areas

Identify areas that appear insufficiently studied.

IMPORTANT:
A missing area is NOT automatically a research gap.

If evidence is insufficient, clearly state:

"Insufficient evidence to establish a research gap."

3. Methodological Gaps

Identify missing:

- methods
- comparisons
- datasets
- benchmarks
- evaluation methods
- experimental conditions

Only identify these when supported by the literature.

4. Conflicting Findings

Identify genuine disagreements between studies.

If none are present:

"No clear conflicting findings identified in the
provided evidence."

5. Potential Research Gaps

For each potential gap provide:

Gap:
<potential research gap>

Evidence:
<specific evidence supporting the gap>

Why it matters:
<why investigating it could be useful>

Evidence Strength:
<Strong / Moderate / Weak / Insufficient>

Confidence:
<High / Medium / Low>

If the evidence is insufficient, write:

Gap:
<area that may require investigation>

Evidence:
<what is actually known>

Evidence Strength:
Insufficient

Confidence:
Low

Do NOT present it as a confirmed research gap.

6. Most Promising Gap

Identify ONE potential gap with the strongest evidence
from the provided literature.

IMPORTANT:

If no gap has sufficient evidence, write:

"No research gap can be confidently established from
the provided evidence."

Do not force a research gap.

Do not choose a gap simply because it sounds interesting.

The strongest gap must be selected based on evidence,
not creativity.

Final rule:

Do not confuse:

"the study did not discuss X"

with:

"the research field has not studied X."

The latter requires stronger evidence.
"""
    )
])

research_gap_chain = (
    research_gap_prompt
    | llm
    | StrOutputParser()
)