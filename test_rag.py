from rag.retriever import create_vectorstore, retrieve_information


research = """
Artificial intelligence is increasingly being used in healthcare.
Machine learning models can assist doctors in diagnosis.
However, many models suffer from limited interpretability.
Researchers are investigating explainable AI methods.
"""


vectorstore = create_vectorstore(research)

results = retrieve_information(
    vectorstore,
    "What are the limitations of AI in healthcare?"
)

for result in results:
    print("\n--- RESULT ---")
    print(result.page_content)