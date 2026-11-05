from agents import research_gap_chain


literature_analysis = """
1. Research Methods
- Machine learning and deep learning models are commonly used.
- Explainable AI methods are increasingly being investigated.

2. Key Findings
- Deep learning models achieve strong diagnostic performance.
- Explainability improves transparency.

3. Common Limitations
- Many models require large datasets.
- Some models have limited interpretability.
- Explainable AI can introduce additional computational cost.

4. Datasets / Experimental Setup
- Medical image datasets are commonly used.
- Most studies evaluate models using accuracy and similar metrics.

5. Research Trends
- Increasing use of deep learning.
- Growing interest in explainable AI.

6. Areas of Disagreement
- Some studies report strong performance from explainable
  approaches, while others report computational overhead.

7. Potential Research Opportunities
- Efficient explainable AI methods for medical diagnosis.
"""


research = """
Several studies investigate deep learning for medical diagnosis.
Most focus on model accuracy. Some studies investigate
explainability, but these approaches can increase computational
requirements.

Few of the provided studies compare lightweight explainable
models with larger deep learning models under the same
experimental conditions.
"""


result = research_gap_chain.invoke({
    "topic": "Explainable AI for Healthcare",
    "literature_analysis": literature_analysis,
    "research": research
})


print("\n" + "=" * 70)
print("RESEARCH GAP ANALYSIS")
print("=" * 70)

print(result)