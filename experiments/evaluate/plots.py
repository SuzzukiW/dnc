import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load the JSON file
with open('experiments/logs/single_agent_traffic/metrics_final.json', 'r') as file:
    data = json.load(file)

# Extract relevant data from the loaded JSON
episode_rewards = data['episode_rewards']
episode_lengths = data['episode_lengths']
waiting_times = data['waiting_times']
queue_lengths = data['queue_lengths']
throughput = data['throughput']
losses = data['losses']
traffic_pressure = data['traffic_pressure']

# Set up the plot styles
sns.set(style="whitegrid")

# 1. Plot Episode Rewards
fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.plot(episode_rewards, marker='o', color='b')
ax1.set_title('Episode Rewards')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Rewards')
fig1.savefig('experiments/evaluate/episode_rewards.png', dpi=300)
plt.close(fig1)

# 2. Plot Episode Lengths
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.bar(range(len(episode_lengths)), episode_lengths, color='green')
ax2.set_title('Episode Lengths')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Length')
fig2.savefig('experiments/evaluate/episode_lengths.png', dpi=300)
plt.close(fig2)

# 3. Plot Waiting Times
fig3, ax3 = plt.subplots(figsize=(7, 5))
ax3.plot(waiting_times, marker='x', color='orange')
ax3.set_title('Waiting Times')
ax3.set_xlabel('Episode')
ax3.set_ylabel('Time (s)')
fig3.savefig('experiments/evaluate/waiting_times.png', dpi=300)
plt.close(fig3)

# 4. Plot Queue Lengths
fig4, ax4 = plt.subplots(figsize=(7, 5))
ax4.plot(queue_lengths, marker='^', color='red')
ax4.set_title('Queue Lengths')
ax4.set_xlabel('Episode')
ax4.set_ylabel('Length (m)')
fig4.savefig('experiments/evaluate/queue_lengths.png', dpi=300)
plt.close(fig4)

# 5. Plot Throughput
fig5, ax5 = plt.subplots(figsize=(7, 5))
ax5.plot(throughput, marker='s', color='purple')
ax5.set_title('Throughput')
ax5.set_xlabel('Episode')
ax5.set_ylabel('Vehicles')
fig5.savefig('experiments/evaluate/throughput.png', dpi=300)
plt.close(fig5)

# 6. Plot Losses
fig6, ax6 = plt.subplots(figsize=(7, 5))
ax6.plot(losses, marker='d', color='brown')
ax6.set_title('Losses')
ax6.set_xlabel('Episode')
ax6.set_ylabel('Losses')
fig6.savefig('experiments/evaluate/losses.png', dpi=300)
plt.close(fig6)

# 7. Plot Average Reward per Episode
avg_rewards = [sum(episode_rewards[:i+1]) / (i+1) for i in range(len(episode_rewards))]
fig7, ax7 = plt.subplots(figsize=(7, 5))
ax7.plot(avg_rewards, marker='o', color='blue', linestyle='dashed')
ax7.set_title('Average Reward per Episode')
ax7.set_xlabel('Episode')
ax7.set_ylabel('Average Rewards')
fig7.savefig('experiments/evaluate/avg_rewards.png', dpi=300)
plt.close(fig7)

# 8. Plot Success Rate
threshold = 10  # Example threshold for success
success_rate = [1 if reward > threshold else 0 for reward in episode_rewards]
fig8, ax8 = plt.subplots(figsize=(7, 5))
ax8.plot(success_rate, marker='x', color='green')
ax8.set_title('Success Rate')
ax8.set_xlabel('Episode')
ax8.set_ylabel('Success (1 or 0)')
fig8.savefig('experiments/evaluate/success_rate.png', dpi=300)
plt.close(fig8)

# 9. Plot Traffic Pressure vs Throughput
fig9, ax9 = plt.subplots(figsize=(7, 5))
ax9.scatter(traffic_pressure, throughput, color='blue')
ax9.set_title('Traffic Pressure vs Throughput')
ax9.set_xlabel('Traffic Pressure')
ax9.set_ylabel('Throughput')
fig9.savefig('experiments/evaluate/traffic_pressure_vs_throughput.png', dpi=300)
plt.close(fig9)

# 10. Plot Cumulative Rewards
cumulative_rewards = [sum(episode_rewards[:i+1]) for i in range(len(episode_rewards))]
fig10, ax10 = plt.subplots(figsize=(7, 5))
ax10.plot(cumulative_rewards, marker='o', color='purple', linestyle='dotted')
ax10.set_title('Cumulative Rewards')
ax10.set_xlabel('Episode')
ax10.set_ylabel('Cumulative Rewards')
fig10.savefig('experiments/evaluate/cumulative_rewards.png', dpi=300)
plt.close(fig10)
