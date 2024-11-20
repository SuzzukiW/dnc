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

# Create a figure with subplots
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# 1. Plot episode rewards
axes[0, 0].plot(episode_rewards, marker='o', color='b')
axes[0, 0].set_title('Episode Rewards')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Rewards')

# 2. Plot episode lengths
axes[0, 1].bar(range(len(episode_lengths)), episode_lengths, color='green')
axes[0, 1].set_title('Episode Lengths')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Length')

# 3. Plot waiting times
axes[1, 0].plot(waiting_times, marker='x', color='orange')
axes[1, 0].set_title('Waiting Times')
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Time (s)')

# 4. Plot queue lengths
axes[1, 1].plot(queue_lengths, marker='^', color='red')
axes[1, 1].set_title('Queue Lengths')
axes[1, 1].set_xlabel('Episode')
axes[1, 1].set_ylabel('Length (m)')

# 5. Plot throughput
axes[2, 0].plot(throughput, marker='s', color='purple')
axes[2, 0].set_title('Throughput')
axes[2, 0].set_xlabel('Episode')
axes[2, 0].set_ylabel('Vehicles')

# 6. Plot losses
axes[2, 1].plot(losses, marker='d', color='brown')
axes[2, 1].set_title('Losses')
axes[2, 1].set_xlabel('Episode')
axes[2, 1].set_ylabel('Losses')

# Adjust layout
plt.tight_layout()

# Save the plot to a file (e.g., as a PNG file)
plt.savefig('experiments/evaluate/metrics_plot.png', dpi=300)  # You can adjust the path and filename

# Show the plot (optional, can be omitted if you only want to save it)
plt.show()
