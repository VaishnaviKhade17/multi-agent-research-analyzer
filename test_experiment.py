from research_agents.experiment_designer import design_experiment


# ============================================================
# TEST INPUT
# ============================================================

topic = "Explainable AI in Healthcare"


hypothesis = (
    "H1: Developing methods to support the explanation "
    "needs of healthcare decision-makers will improve "
    "the effectiveness of AI systems in healthcare."
)


evolved_hypothesis = """
Improved Hypothesis:

A proposed explanation interface tailored to healthcare
decision-makers may improve decision-making performance
compared with a baseline interface.

This is a proposed refinement, not an established finding.
"""


contradiction_analysis = """
Evidence Assessment:

Insufficient direct evidence.

No direct experiment establishing this hypothesis
was identified in the supplied evidence.

The absence of contradictory evidence must not be
interpreted as confirmation of the hypothesis.
"""


research_gap = """
Research Gap Assessment:

Insufficient evidence to establish a confirmed
field-level research gap.

Further empirical evaluation may be required.
"""


research = """
Research Context:

The supplied literature concerns explainable AI,
interpretability, and healthcare decision-making.

No verified dataset or completed experiment directly
testing this hypothesis was supplied.

No verified participant pool, sample size, recruitment
source, or standardized benchmark was supplied.

Dataset availability and participant availability
require verification.
"""


# ============================================================
# RUN
# ============================================================

result = design_experiment(
    topic=topic,
    hypothesis=hypothesis,
    evolved_hypothesis=evolved_hypothesis,
    contradiction_analysis=contradiction_analysis,
    research_gap=research_gap,
    research=research
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT DESIGN TEST")
print("=" * 70)

print(result)

print("\n" + "=" * 70)
print("END OF EXPERIMENT DESIGN TEST")
print("=" * 70)