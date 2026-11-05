from agents import literature_analysis_chain


research = """
Artificial intelligence is increasingly being used in healthcare.

Study 1:
Machine learning models were used for disease diagnosis.
The study achieved high accuracy but reported limited
interpretability.

Study 2:
Deep learning was applied to medical image classification.
The model achieved strong performance but required a large
dataset.

Study 3:
Explainable AI methods were investigated to improve the
interpretability of healthcare models.
The study reported improved transparency but noted additional
computational cost.
"""


result = literature_analysis_chain.invoke({
    "topic": "Artificial Intelligence in Healthcare",
    "research": research
})


print("\n" + "=" * 70)
print("LITERATURE ANALYSIS")
print("=" * 70)

print(result)