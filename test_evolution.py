
from research_agents.evolution_agent import evolve_hypothesis


topic = "Explainable AI in Healthcare"


hypothesis = """
Lightweight explainable AI models can achieve comparable diagnostic
performance to larger deep learning models while requiring less
computational cost.
"""


contradiction_analysis = """
Evidence Assessment: Insufficient

The available literature provides evidence about explainability
and computational efficiency, but there is limited direct evidence
comparing lightweight explainable models with larger deep learning
models on the same diagnostic tasks.
"""


research_gap = """
Potential Research Gap:

There is limited direct comparison between lightweight explainable
AI models and larger deep learning models using standardized
datasets, diagnostic performance metrics, and computational-cost
measurements.
"""


print("\n" + "=" * 70)
print("HYPOTHESIS EVOLUTION AGENT")
print("=" * 70)


result = evolve_hypothesis(
    topic=topic,
    hypothesis=hypothesis,
    contradiction_analysis=contradiction_analysis,
    research_gap=research_gap
)


print("\nFINAL EVOLVED HYPOTHESIS:\n")
print(result)

