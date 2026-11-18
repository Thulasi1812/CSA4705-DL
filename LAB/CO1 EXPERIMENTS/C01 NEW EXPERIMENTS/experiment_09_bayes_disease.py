"""Experiment 9: Bayes' theorem for a positive diagnostic test."""
prevalence = 0.01
sensitivity = 0.95
specificity = 0.90
false_positive_rate = 1 - specificity

p_positive = sensitivity * prevalence + false_positive_rate * (1 - prevalence)
p_disease_given_positive = sensitivity * prevalence / p_positive

print(f"Probability of a positive test: {p_positive:.4f}")
print(f"P(disease | positive) = {p_disease_given_positive:.4f}")
print(f"Posterior probability = {p_disease_given_positive * 100:.2f}%")
print("A positive result does not imply certainty; prevalence and false positives matter.")
