from research_agents.hypothesis_agent import generate_hypotheses


topic = "Explainable AI for Healthcare"


gaps = """
1. Few studies compare lightweight explainable AI models with
   larger deep learning models under the same conditions.

2. There is a lack of standardized evaluation of explainable AI
   methods beyond accuracy.
"""


research = """
Several studies report that deep learning models can achieve
strong diagnostic performance but often require large datasets.

Other studies report that explainable AI can improve transparency
but may introduce additional computational cost.

The provided literature contains limited direct comparison
between lightweight explainable models and larger models.
"""


result = generate_hypotheses(
    topic,
    gaps,
    research
)


print("\n" + "=" * 70)
print("HYPOTHESIS GENERATION")
print("=" * 70)

print(result)