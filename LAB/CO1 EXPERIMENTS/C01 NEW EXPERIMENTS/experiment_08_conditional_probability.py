"""Experiment 8: Conditional probability using Bayes' theorem."""
# Machine A makes 60% of products and has a 2% defect rate.
# Machine B makes 40% of products and has a 5% defect rate.
p_a, p_b = 0.60, 0.40
p_d_given_a, p_d_given_b = 0.02, 0.05

p_defective = p_d_given_a * p_a + p_d_given_b * p_b
p_a_given_defective = p_d_given_a * p_a / p_defective

print(f"P(defective) = {p_defective:.4f}")
print(f"P(Machine A | defective) = {p_a_given_defective:.4f}")
print(f"Percentage from Machine A = {p_a_given_defective * 100:.2f}%")
