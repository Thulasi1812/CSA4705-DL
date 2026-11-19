# Simple Q-learning toy traffic signal agent
import numpy as np, matplotlib.pyplot as plt
rng=np.random.default_rng(42); Q=np.zeros((4,2)); rewards=[]
for ep in range(500):
 s=rng.integers(4); total=0
 for _ in range(25):
  a=rng.integers(2) if rng.random()<.15 else np.argmax(Q[s]); reward=1 if a==(s%2) else -1
  ns=rng.integers(4); Q[s,a]+=.1*(reward+.9*np.max(Q[ns])-Q[s,a]);s=ns;total+=reward
 rewards.append(total)
plt.plot(np.convolve(rewards,np.ones(25)/25,mode="valid"));plt.xlabel("Episode");plt.ylabel("Smoothed reward");plt.title("Toy traffic Q-learning");plt.tight_layout();plt.savefig("output.png")
print("Q table:",Q)
