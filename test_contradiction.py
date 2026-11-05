from research_agents.contradiction_agent import analyze_hypothesis


topic = "Explainable AI in Healthcare"


hypothesis = """
Lightweight explainable AI models can achieve comparable diagnostic
performance to larger deep learning models while requiring less
computational cost.
"""


research = """
Several studies report that deep learning models achieve strong
diagnostic performance but require substantial computational
resources.

Some research suggests that explainable AI can improve transparency,
but additional computational overhead may occur.

The provided literature contains limited direct comparison between
lightweight explainable models and larger deep learning models.
"""


print("\n" + "=" * 70)
print("CONTRADICTION HUNTER")
print("=" * 70)

result = analyze_hypothesis(
    topic=topic,
    hypothesis=hypothesis,
    research=research
)

print("\nFINAL CONTRADICTION ANALYSIS:\n")
print(result)