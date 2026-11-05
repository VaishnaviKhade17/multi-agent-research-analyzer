from research_agents.experiment_critic import critique_experiment


topic = "AI Conversational Tutors for Underrepresented Languages"

hypothesis = """
H3: Developing AI Conversational Tutors for Underrepresented Languages
Can Improve Language Learning Outcomes by Enhancing Skill Development
and Engagement through Personalized Feedback Mechanisms.
"""

evolved_hypothesis = """
H3a: Developing AI Conversational Tutors for Underrepresented Languages
Can Improve Language Learning Outcomes by Enhancing Skill Development
and Engagement through Personalized Feedback Mechanisms.
"""

experiment_design = """
1. Experiment Objective:

To evaluate the effectiveness of AI conversational tutors in improving
language learning outcomes.

2. Hypothesis Under Test:

H3a: Developing AI Conversational Tutors for Underrepresented Languages
Can Improve Language Learning Outcomes.

3. Research Question:

Can AI conversational tutors improve language learning outcomes?

4. Experimental Methodology:

Between-subjects design with two independent groups.

5. Variables:

- Independent: Group assignment
- Dependent: Language learning outcomes
- Controlled: None
- Potential confounders: Prior knowledge and language proficiency

6. Participants / Experimental Units:

- Target population: Students learning underrepresented languages
- Sample size: To be determined
- Participant availability: Requires verification

7. Baseline / Comparison Groups:

- Experimental group: Students using the AI conversational tutor
- Control group: Students using a traditional learning resource

8. Evaluation Metrics:

- Language learning outcomes measured using a standardized test

9. Experimental Procedure:

Step 1: Recruit eligible participants.
Step 2: Randomly assign participants to Group A or Group B.
Step 3: Group A uses the AI conversational tutor.
Step 4: Group B uses the traditional learning resource.
Step 5: Both groups complete the standardized test.
Step 6: Compare outcomes between groups.

10. Statistical Analysis Plan:

- Primary analysis: Independent-samples t-test
- Assumption checks: Normality and equal variance
- Alternative analysis: Non-parametric test if assumptions fail

11. Ethical Considerations:

Participants should provide informed consent and their data should
be protected.
"""


print("=" * 60)
print("EXPERIMENT CRITIC TEST")
print("=" * 60)

result = critique_experiment(
    topic=topic,
    hypothesis=hypothesis,
    evolved_hypothesis=evolved_hypothesis,
    experiment_design=experiment_design
)

print("\nEXPERIMENT CRITIQUE:\n")
print(result)

print("\n" + "=" * 60)
print("TEST COMPLETED")
print("=" * 60)