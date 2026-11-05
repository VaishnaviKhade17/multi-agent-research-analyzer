#pipeline.py
import re

from agents import (
    writer_chain,
    critic_chain,
    literature_analysis_chain,
    research_gap_chain
)

from rag.retriever import (
    create_vectorstore,
    retrieve_information
)

from research_agents.hypothesis_agent import generate_hypotheses
from research_agents.contradiction_agent import analyze_hypothesis
from research_agents.evolution_agent import evolve_hypothesis
from research_agents.experiment_designer import design_experiment
from research_agents.experiment_critic import critique_experiment

from paper_tools import arxiv_search, semantic_scholar_search
from tools import scrape_url


# ================================================================
# HELPER 1 - EXTRACT VERIFIED URLS
# ================================================================

def extract_verified_urls(search_results: str) -> list:
    """
    Extract only URLs explicitly returned by academic search tools.
    """

    urls = re.findall(
        r"URL:\s*(https?://[^\s]+)",
        search_results
    )

    # Remove trailing punctuation and duplicates
    urls = [
        url.rstrip(".,;)")
        for url in urls
    ]

    return list(dict.fromkeys(urls))


# ================================================================
# HELPER 2 - EXTRACT INDIVIDUAL HYPOTHESES
# ================================================================

def extract_hypotheses(hypothesis_text: str) -> list:
    """
    Extract H1, H2 and H3 independently.

    Supports headings such as:
    H1:
    **H1:**
    H1: hypothesis text
    **H1:** hypothesis text

    Returns a list of dictionaries containing the hypothesis
    number, full section, and actual hypothesis statement.
    """

    heading_pattern = re.compile(
        r"(?im)^[ \t]*\*{0,2}[ \t]*H([1-3])[ \t]*:"
        r"[ \t]*\*{0,2}[ \t]*$"
    )

    matches = list(
        heading_pattern.finditer(hypothesis_text)
    )

    hypotheses = []

    for index, match in enumerate(matches):

        number = int(match.group(1))

        start = match.end()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(hypothesis_text)

        section = hypothesis_text[start:end].strip()

        # Remove any leading Markdown symbols
        section = re.sub(
            r"^[ \t]*\*+[ \t]*",
            "",
            section
        )

        # Extract the actual hypothesis statement.
        # Stop before Independent Variable, Dependent Variable,
        # Measurement, Based On, or Confidence.

        claim_match = re.search(
            r"(?is)(.*?)(?=\n\s*\*{0,2}\s*"
            r"(?:Independent Variable|Dependent Variable|"
            r"Measurement|Based On|Confidence)\s*:)",
            section
        )

        if claim_match:
            claim = claim_match.group(1).strip()
        else:
            claim = section.strip()

        # Clean Markdown and line breaks
        claim = re.sub(
            r"^[ \t]*\*{1,2}|[ \t]*\*{1,2}$",
            "",
            claim
        ).strip()

        claim = re.sub(
            r"\s+",
            " ",
            claim
        ).strip()

        if claim:
            hypotheses.append({
                "number": number,
                "section": section,
                "claim": claim
            })

    # Fallback: support inline headings if the model
    # does not put H1/H2/H3 on separate lines.

    if not hypotheses:

        inline_pattern = re.compile(
            r"(?is)(?:^|\n)\s*\*{0,2}\s*"
            r"H([1-3])\s*:\s*\*{0,2}\s*"
            r"(.*?)(?=\n\s*\*{0,2}\s*H[1-3]\s*:|\Z)"
        )

        for match in inline_pattern.finditer(
            hypothesis_text
        ):
            number = int(match.group(1))
            section = match.group(2).strip()

            claim_match = re.search(
                r"(?is)(.*?)(?=\n\s*\*{0,2}\s*"
                r"(?:Independent Variable|Dependent Variable|"
                r"Measurement|Based On|Confidence)\s*:)",
                section
            )

            claim = (
                claim_match.group(1).strip()
                if claim_match
                else section
            )

            claim = re.sub(
                r"\s+",
                " ",
                claim
            ).strip()

            if claim:
                hypotheses.append({
                    "number": number,
                    "section": section,
                    "claim": claim
                })

    # Remove duplicate hypothesis numbers
    unique_hypotheses = []
    seen = set()

    for item in hypotheses:
        if item["number"] not in seen:
            unique_hypotheses.append(item)
            seen.add(item["number"])

    return unique_hypotheses


# ================================================================
# HELPER 3 - VALIDATE REPORT URLS
# ================================================================

def validate_report_sources(
    report: str,
    verified_urls: list
) -> str:
    """
    Remove URLs from the report that were not returned by
    the academic search tools.

    Rebuild the Verified Sources section using only verified URLs.
    """

    verified_set = set(verified_urls)

    # Remove Markdown links while preserving their visible text.
    # Example: [Paper](https://example.com) -> Paper

    markdown_link_pattern = re.compile(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)"
    )

    def replace_markdown_link(match):

        label = match.group(1)
        url = match.group(2).rstrip(".,;")

        if url in verified_set:
            return f"{label} ({url})"

        return (
            f"{label} "
            "[Unverified URL removed]"
        )

    report = markdown_link_pattern.sub(
        replace_markdown_link,
        report
    )

    # Remove any remaining unverified URLs

    url_pattern = re.compile(
        r"https?://[^\s\])>\"']+"
    )

    def replace_unverified_url(match):

        url = match.group(0).rstrip(".,;")
        trailing = match.group(0)[len(url):]

        if url in verified_set:
            return url + trailing

        return "[Unverified URL removed]" + trailing

    report = url_pattern.sub(
        replace_unverified_url,
        report
    )

    # Remove any source list generated by the LLM.
    # We will rebuild this section ourselves.

    report = re.sub(
        r"(?is)\n\s*(?:#{1,3}\s*)?"
        r"\*{0,2}\s*Verified Sources\s*\*{0,2}\s*:?"
        r".*$",
        "",
        report
    ).strip()

    # Append the verified sources directly from search results

    report += "\n\n## Verified Sources\n\n"

    if verified_urls:

        for index, url in enumerate(
            verified_urls,
            start=1
        ):
            report += f"{index}. {url}\n"

    else:
        report += (
            "No verified academic source URLs were returned "
            "by the search tools.\n"
        )

    return report


# ================================================================
# MAIN RESEARCH PIPELINE
# ================================================================

def run_research_pipeline(topic: str) -> dict:

    state = {}

    # ============================================================
    # STEP 1 - VERIFIED RESEARCH SEARCH
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 1 - Verified Research Search")
    print("Searching real academic sources...")
    print("=" * 60)

    arxiv_results = arxiv_search.invoke(topic)

    semantic_results = semantic_scholar_search.invoke(
        topic
    )

    state["search_results"] = (
        "ARXIV RESULTS:\n\n"
        + arxiv_results
        + "\n\n"
        + "=" * 70
        + "\n\nSEMANTIC SCHOLAR RESULTS:\n\n"
        + semantic_results
    )

    print("\nVERIFIED SEARCH RESULTS:\n")
    print(state["search_results"])

    # Save the URLs returned by the academic APIs
    state["verified_urls"] = extract_verified_urls(
        state["search_results"]
    )

    # ============================================================
    # STEP 2 - READER
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 2 - Reader")
    print("Reading verified academic sources...")
    print("=" * 60)

    urls = state["verified_urls"]

    print("\nVerified URLs found:", len(urls))

    scraped_results = []

    for url in urls[:2]:

        print(f"\nReading: {url}")

        try:
            reader_result = scrape_url.invoke(url)

            scraped_results.append(
                f"SOURCE URL: {url}\n\n"
                f"{reader_result}"
            )

        except Exception as e:
            print(f"Could not read {url}: {e}")

    if scraped_results:

        state["scraped_content"] = (
            "\n\n"
            + "\n\n====================\n\n".join(
                scraped_results
            )
        )

    else:
        state["scraped_content"] = (
            "No additional scraped content available."
        )

    print("\nSCRAPED CONTENT:\n")
    print(state["scraped_content"])

    # ============================================================
    # STEP 3 - COMBINE RESEARCH
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 3 - Combining Research")
    print("=" * 60)

    research_combined = (
        f"SEARCH RESULTS:\n\n"
        f"{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n\n"
        f"{state['scraped_content']}"
    )

    state["research_combined"] = research_combined

    # ============================================================
    # STEP 4 - RAG KNOWLEDGE BASE
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 4 - RAG Knowledge Base")
    print("Creating embeddings and FAISS vector store...")
    print("=" * 60)

    vectorstore = create_vectorstore(
        research_combined
    )

    print("\nRAG vector store created successfully.")

    # ============================================================
    # STEP 5 - RAG RETRIEVAL
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 5 - RAG Retrieval")
    print("Finding relevant research information...")
    print("=" * 60)

    retrieved_docs = retrieve_information(
        vectorstore,
        topic,
        k=4
    )

    retrieved_context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    state["retrieved_context"] = retrieved_context

    print("\nRETRIEVED CONTEXT:\n")
    print(retrieved_context)

    # ============================================================
    # STEP 6 - LITERATURE ANALYSIS
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 6 - Literature Analysis Agent")
    print("Analyzing existing research...")
    print("=" * 60)

    state["literature_analysis"] = (
        literature_analysis_chain.invoke({
            "topic": topic,
            "research": retrieved_context
        })
    )

    print("\nLITERATURE ANALYSIS:\n")
    print(state["literature_analysis"])

    # ============================================================
    # STEP 7 - RESEARCH GAP
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 7 - Research Gap Agent")
    print("Identifying potential research gaps...")
    print("=" * 60)

    state["research_gaps"] = research_gap_chain.invoke({
        "topic": topic,
        "literature_analysis": state[
            "literature_analysis"
        ],
        "research": retrieved_context
    })

    print("\nRESEARCH GAPS:\n")
    print(state["research_gaps"])

    # ============================================================
    # STEP 8 - HYPOTHESIS GENERATION
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 8 - Hypothesis Generation Agent")
    print("Generating testable research hypotheses...")
    print("=" * 60)

    state["hypotheses"] = generate_hypotheses(
        topic=topic,
        gaps=state["research_gaps"],
        research=retrieved_context
    )

    print("\nGENERATED HYPOTHESES:\n")
    print(state["hypotheses"])

    # Parse the hypotheses into individual records
    hypotheses = extract_hypotheses(
        state["hypotheses"]
    )

    state["parsed_hypotheses"] = hypotheses

    print("\nEXTRACTED HYPOTHESES:")
    print("Total hypotheses extracted:", len(hypotheses))

    for item in hypotheses:
        print(
            f"\nH{item['number']}: "
            f"{item['claim']}"
        )

    if not hypotheses:
        print(
            "\nERROR: No individual hypotheses could "
            "be extracted. Check hypothesis_agent.py output."
        )

        state["error"] = (
            "No hypotheses could be extracted. "
            "Check the output format of hypothesis_agent.py."
        )

        return state

    # ============================================================
    # STEP 9 - CONTRADICTION HUNTER
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 9 - Contradiction Hunter")
    print("Testing each hypothesis separately...")
    print("=" * 60)

    contradiction_results = []

    for item in hypotheses:

        number = item["number"]
        claim = item["claim"]

        print("\n" + "-" * 60)
        print(f"Analyzing H{number}")
        print("-" * 60)

        # IMPORTANT:
        # Pass only the actual hypothesis statement.
        # Do not pass other hypotheses, measurements,
        # confidence ratings, or proposed outcomes as evidence.

        isolated_hypothesis = (
            f"H{number}: {claim}"
        )

        result = analyze_hypothesis(
            topic=topic,
            hypothesis=isolated_hypothesis,
            research=retrieved_context
        )

        contradiction_results.append({
            "number": number,
            "hypothesis": isolated_hypothesis,
            "analysis": result
        })

        print(f"\nCONTRADICTION RESULT H{number}:\n")
        print(result)

    state["contradiction_results"] = (
        contradiction_results
    )

    state["contradiction_analysis"] = (
        "\n\n====================\n\n".join(
            (
                f"HYPOTHESIS {item['number']}\n\n"
                f"{item['analysis']}"
            )
            for item in contradiction_results
        )
    )

    print("\nCOMPLETE CONTRADICTION ANALYSIS:\n")
    print(state["contradiction_analysis"])

    # ============================================================
    # STEP 10 - HYPOTHESIS EVOLUTION
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 10 - Hypothesis Evolution Agent")
    print("Refining each hypothesis separately...")
    print("=" * 60)

    evolved_results = []

    for item in hypotheses:

        number = item["number"]
        claim = item["claim"]

        print("\n" + "-" * 60)
        print(f"Evolving H{number}")
        print("-" * 60)

        # Get only the contradiction analysis belonging
        # to this hypothesis.

        matching_result = next(
            (
                result
                for result in contradiction_results
                if result["number"] == number
            ),
            None
        )

        if matching_result:
            contradiction_for_hypothesis = (
                matching_result["analysis"]
            )
        else:
            contradiction_for_hypothesis = (
                "No contradiction analysis available."
            )

        evolved = evolve_hypothesis(
            topic=topic,
            hypothesis=f"H{number}: {claim}",
            contradiction_analysis=(
                contradiction_for_hypothesis
            ),
            research_gap=state["research_gaps"]
        )

        evolved_results.append({
            "number": number,
            "original_hypothesis": claim,
            "evolution": evolved
        })

        print(f"\nEVOLVED H{number}:\n")
        print(evolved)

    state["evolved_results"] = evolved_results

    state["evolved_hypothesis"] = (
        "\n\n====================\n\n".join(
            (
                f"HYPOTHESIS {item['number']} EVOLUTION\n\n"
                f"{item['evolution']}"
            )
            for item in evolved_results
        )
    )

    print("\nEVOLVED HYPOTHESES:\n")
    print(state["evolved_hypothesis"])

        # ============================================================
    # STEP 11 - EXPERIMENT DESIGNER
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 11 - Experiment Designer Agent")
    print("Designing experiments for each hypothesis...")
    print("=" * 60)

    experiment_results = []

    for item in hypotheses:

        number = item["number"]
        claim = item["claim"]

        print("\n" + "-" * 60)
        print(f"Designing experiment for H{number}")
        print("-" * 60)

        # --------------------------------------------------------
        # Get contradiction analysis for this hypothesis
        # --------------------------------------------------------

        matching_contradiction = next(
            (
                result
                for result in contradiction_results
                if result["number"] == number
            ),
            None
        )

        if matching_contradiction:

            contradiction_for_hypothesis = (
                matching_contradiction["analysis"]
            )

        else:

            contradiction_for_hypothesis = (
                "No contradiction analysis available."
            )

        # --------------------------------------------------------
        # Get evolved hypothesis for this hypothesis
        # --------------------------------------------------------

        matching_evolution = next(
            (
                result
                for result in evolved_results
                if result["number"] == number
            ),
            None
        )

        if matching_evolution:

            evolved_for_hypothesis = (
                matching_evolution["evolution"]
            )

        else:

            evolved_for_hypothesis = (
                "No evolved hypothesis available."
            )

        # --------------------------------------------------------
        # Design the experiment
        # --------------------------------------------------------

        experiment_design = design_experiment(

            topic=topic,

            hypothesis=f"H{number}: {claim}",

            evolved_hypothesis=evolved_for_hypothesis,

            contradiction_analysis=(
                contradiction_for_hypothesis
            ),

            research_gap=state["research_gaps"],

            research=retrieved_context
        )

        # --------------------------------------------------------
        # Store result
        # --------------------------------------------------------

        experiment_results.append({

            "number": number,

            "hypothesis": f"H{number}: {claim}",

            "experiment_design": experiment_design

        })

        print(
            f"\nEXPERIMENT DESIGN H{number}:\n"
        )

        print(experiment_design)


    # ------------------------------------------------------------
    # Combine all experiment designs
    # ------------------------------------------------------------

    state["experiment_results"] = experiment_results

    state["experiment_designs"] = (

        "\n\n====================\n\n".join(

            (

                f"HYPOTHESIS {item['number']} "
                f"EXPERIMENT DESIGN\n\n"

                f"{item['experiment_design']}"

            )

            for item in experiment_results

        )

    )

    print("\nCOMPLETE EXPERIMENT DESIGNS:\n")

    print(state["experiment_designs"])
    # ============================================================
    # STEP 12 - EXPERIMENT CRITIC
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 12 - Experiment Critic Agent")
    print("Critiquing experiment designs...")
    print("=" * 60)

    experiment_critic_results = []

    for item in experiment_results:

        number = item["number"]

        matching_evolution = next(
            (
                result
                for result in evolved_results
                if result["number"] == number
            ),
            None
        )

        if matching_evolution:
            evolved_for_hypothesis = matching_evolution["evolution"]
        else:
            evolved_for_hypothesis = "No evolved hypothesis available."

        print("\n" + "-" * 60)
        print(f"Critiquing experiment for H{number}")
        print("-" * 60)

        critique = critique_experiment(
            topic=topic,
            hypothesis=item["hypothesis"],
            evolved_hypothesis=evolved_for_hypothesis,
            experiment_design=item["experiment_design"]
        )

        experiment_critic_results.append({
            "number": number,
            "critique": critique
        })

        print(f"\nEXPERIMENT CRITIQUE H{number}:\n")
        print(critique)


    state["experiment_critic_results"] = experiment_critic_results

    state["experiment_critiques"] = (
        "\n\n====================\n\n".join(
            (
                f"HYPOTHESIS {item['number']} EXPERIMENT CRITIQUE\n\n"
                f"{item['critique']}"
            )
            for item in experiment_critic_results
        )
    )

    # ============================================================
    # STEP 13 - WRITER
    # ============================================================

    print("\n" + "=" * 60)
    print("STEP 13 - Writer")
    print("Preparing structured research report...")
    print("=" * 60)

    # ------------------------------------------------------------
    # Build a hypothesis-wise record.
    # This prevents the Writer from mixing H1, H2, and H3.
    # ------------------------------------------------------------

    writer_hypothesis_records = []

    for hypothesis_item in hypotheses:

        number = hypothesis_item["number"]

        # Find the matching contradiction analysis
        contradiction_item = next(
            (
                item for item in contradiction_results
                if item["number"] == number
            ),
            None
        )

        # Find the matching evolved hypothesis
        evolution_item = next(
            (
                item for item in evolved_results
                if item["number"] == number
            ),
            None
        )

        # Find the matching experiment design
        experiment_item = next(
            (
                item for item in experiment_results
                if item["number"] == number
            ),
            None
        )

        # Find the matching experiment critique
        critique_item = next(
            (
                item for item in experiment_critic_results
                if item["number"] == number
            ),
            None
        )

        # Use explicit fallbacks if an agent returned no result
        contradiction_text = (
            contradiction_item["analysis"]
            if contradiction_item
            else "No contradiction analysis available."
        )

        evolution_text = (
            evolution_item["evolution"]
            if evolution_item
            else "No hypothesis evolution available."
        )

        experiment_text = (
            experiment_item["experiment_design"]
            if experiment_item
            else "No experiment design available."
        )

        critique_text = (
            critique_item["critique"]
            if critique_item
            else "No experiment critique available."
        )


        # Keep all information for this hypothesis together
        record = f"""
============================================================
HYPOTHESIS H{number} - COMPLETE RESEARCH RECORD
============================================================

ORIGINAL HYPOTHESIS:
{hypothesis_item["claim"]}

CONTRADICTION ANALYSIS FOR H{number}:
{contradiction_text}

EVOLUTION FOR H{number}:
{evolution_text}

EXPERIMENT DESIGN FOR H{number}:
{experiment_text}

EXPERIMENT CRITIQUE FOR H{number}:
{critique_text}
"""

        writer_hypothesis_records.append(record)

    # ------------------------------------------------------------
    # Construct structured research input for the Writer
    # ------------------------------------------------------------

    hypothesis_records_text = "\n\n".join(
        writer_hypothesis_records
    )

    verified_sources_text = (
        "\n".join(state["verified_urls"])
        if state["verified_urls"]
        else "No verified academic source URLs available."
    )

    writer_research = f"""
============================================================
RESEARCH REPORT INPUT
============================================================

TOPIC:
{topic}

============================================================
1. VERIFIED ACADEMIC SEARCH RESULTS
============================================================

{state["search_results"]}

============================================================
2. RETRIEVED LITERATURE
============================================================

{retrieved_context}

============================================================
3. LITERATURE ANALYSIS
============================================================

{state["literature_analysis"]}

============================================================
4. RESEARCH GAPS
============================================================

{state["research_gaps"]}

============================================================
5. HYPOTHESIS-WISE RESEARCH RECORDS
============================================================

IMPORTANT:
The following records are grouped by their original hypothesis
number. Keep each hypothesis, contradiction analysis, evolved
hypothesis, experiment design, and critique together.

Do not transfer experimental details or critiques from one
hypothesis to another.

{hypothesis_records_text}

============================================================
6. VERIFIED SOURCE URL ALLOWLIST
============================================================

Only the following URLs were returned by the academic search
tools. Do not invent, modify, or guess URLs.

{verified_sources_text}

============================================================
7. REPORT WRITING RULES
============================================================

1. Write a structured research report on the supplied topic.

2. Include these sections:
   - Title
   - Abstract
   - Introduction
   - Literature Review
   - Research Gaps
   - Research Objectives
   - Original Hypotheses
   - Contradiction Analysis
   - Evolved Hypotheses
   - Proposed Experimental Designs
   - Experiment Critiques and Required Corrections
   - Limitations
   - Conclusion
   - Verified Sources

3. Include EVERY original hypothesis provided above.
   Do not omit H1, H2, or H3 if they are present in the input.

4. For each hypothesis, report its matching:
   - Original hypothesis
   - Contradiction analysis
   - Evolved hypothesis
   - Experimental design
   - Experiment critique

5. Keep H1, H2, and H3 separate. Never combine their
   experimental procedures, participants, variables, or critiques.

6. Preserve the actual content of the supplied experiment designs
   and critiques. Do not silently invent or alter details.

7. Clearly distinguish:
   - Findings reported in retrieved academic literature
   - Research gaps
   - Proposed hypotheses
   - Proposed experiments
   - Expected observations
   - Actual experimental results

8. Do not present a proposed hypothesis or expected observation
   as an established research finding.

9. Do not invent experimental results, participant numbers,
   sample sizes, recruitment sources, statistical significance,
   ethics approval, or completed experiments.

10. If sample size is undetermined, state:
    "To be determined using an appropriate power/sample-size
    analysis."

11. If actual findings are unavailable, explicitly state:
    "No experimental results are available; the experiments
    described are proposed designs."

12. Do not claim that an experiment has been conducted unless
    the supplied evidence explicitly confirms it.

13. Use only the verified source URLs listed above.
    Do not create citations or claim that a source supports
    something unless that support appears in the supplied
    research material.

14. If information is missing, state that it is not specified
    in the supplied evidence. Do not fill the gap with guesses.

15. Write a coherent academic report with clear headings,
    avoiding repeated content and unsupported claims.
"""

    # ------------------------------------------------------------
    # Generate the report
    # ------------------------------------------------------------

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": writer_research
    })

    # ------------------------------------------------------------
    # Validate URLs after report generation
    # ------------------------------------------------------------

    state["report"] = validate_report_sources(
        state["report"],
        state["verified_urls"]
    )

    print("\nFINAL REPORT:\n")
    print(state["report"])
    # ============================================================
    # STEP 14 - CRITIC
    # ============================================================

    state["feedback"] = critic_chain.invoke({
        "report": state["report"],
        "verified_urls": (
            "\n".join(state["verified_urls"])
            if state.get("verified_urls")
            else "None available."
        )
    })

    # DETERMINISTIC CORRECTION (not prompt-based):
    # A small local model can ignore the "don't flag these" instruction
    # even when given the verified URL list. Rather than relying on the
    # model to behave, attach a factual correction banner to whatever it
    # produced — this guarantees the final output is accurate regardless
    # of what the critic model actually said.
    if state.get("verified_urls"):
        correction_banner = (
            "NOTE (programmatically verified, not generated by the critic "
            "model): The following source URLs were confirmed directly "
            "against live arXiv/Semantic Scholar APIs BEFORE this report "
            "was written. They are real. Any statement below claiming "
            "these sources are fictional, unverified, or hallucinated is "
            "a critic model error and should be disregarded:\n\n"
            + "\n".join(state["verified_urls"])
            + "\n\n" + ("-" * 60) + "\n\n"
        )
        state["feedback"] = correction_banner + state["feedback"]

    print("\nCRITIC REPORT:\n")
    print(state["feedback"])

    # ============================================================
    # RETURN RESULTS
    # ============================================================

    return state


# ================================================================
# RUN PROGRAM
# ================================================================

if __name__ == "__main__":

    topic = input(
        "\nEnter a research topic: "
    ).strip()

    if not topic:
        print("Please enter a research topic.")

    else:
        run_research_pipeline(topic)