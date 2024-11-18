# experiments/evaluate/fixed_time_baseline.py

import os
import sys
from pathlib import Path
import numpy as np
import yaml
import traci
from datetime import datetime
import json

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.environment.multi_agent_sumo_env import MultiAgentSumoEnvironment
from experiments.evaluate.evaluation_metrics import TrafficMetricsEvaluator

class FixedTimeController:
    """
    Fixed-time traffic light controller
    Uses pre-defined phase durations regardless of traffic conditions
    """
    def __init__(self, phases_config):
        """
        Args:
            phases_config: Dictionary with traffic light configurations
                {
                    'cycle_time': total cycle time in seconds,
                    'phases': list of phase durations in seconds,
                    'yellow_time': yellow phase duration in seconds
                }
        """
        self.cycle_time = phases_config.get('cycle_time', 120)
        self.phases = phases_config.get('phases', [30, 30, 30, 30])  # Default equal phases
        self.yellow_time = phases_config.get('yellow_time', 3)
        
        # Validate configuration
        assert sum(self.phases) + (len(self.phases) * self.yellow_time) <= self.cycle_time, \
            "Total phase time plus yellow phases exceeds cycle time"
    
    def get_phase(self, time_in_cycle):
        """
        Determine the current phase based on cycle time
        
        Args:
            time_in_cycle: Current time position in the cycle
            
        Returns:
            phase_index: Current phase index
            is_yellow: Whether current phase is yellow
        """
        if time_in_cycle >= self.cycle_time:
            time_in_cycle = time_in_cycle % self.cycle_time
        
        current_time = 0
        phase_length = len(self.phases)
        
        for i in range(phase_length):
            # Check main phase
            phase_end = current_time + self.phases[i]
            if current_time <= time_in_cycle < phase_end:
                return i, False
            
            # Check yellow phase
            yellow_end = phase_end + self.yellow_time
            if phase_end <= time_in_cycle < yellow_end:
                return i, True
            
            current_time = yellow_end
        
        # If we get here, we're at the end of the cycle
        return 0, False

class FixedTimeBaseline:
    """Implementation of fixed-time baseline for traffic control"""
    def __init__(self, env_config, phases_config):
        print("\nInitializing FixedTimeBaseline...")
        print("Environment config:", env_config)
        self.env = MultiAgentSumoEnvironment(**env_config)
        self.traffic_lights = self.env.traffic_lights
        print(f"Found {len(self.traffic_lights)} traffic lights: {self.traffic_lights}")
        
        self.controllers = {
            tl_id: FixedTimeController(phases_config)
            for tl_id in self.traffic_lights
        }
        print("Initialized controllers for all traffic lights")
    
    def run_episode(self, evaluator=None):
        """Run one episode with fixed-time control"""
        print("\nStarting new episode...")
        states, _ = self.env.reset()
        print(f"Environment reset complete. Initial states received for {len(states)} traffic lights")
        
        done = False
        step = 0
        
        try:
            while not done:
                if step % 100 == 0:
                    print(f"Step {step}")
                
                # Calculate current phase for each traffic light
                actions = {}
                for tl_id in self.traffic_lights:
                    controller = self.controllers[tl_id]
                    time_in_cycle = step % controller.cycle_time
                    phase, is_yellow = controller.get_phase(time_in_cycle)
                    
                    if is_yellow:
                        # Use yellow phase index from environment
                        if tl_id in self.env.yellow_phase_dict:
                            phase = self.env.yellow_phase_dict[tl_id][0]
                        else:
                            print(f"Warning: No yellow phase found for traffic light {tl_id}")
                            phase = 0
                    
                    actions[tl_id] = phase
                
                # Take step in environment
                try:
                    next_states, rewards, done, _, info = self.env.step(actions)
                    
                    # Collect metrics if evaluator is provided
                    if evaluator is not None:
                        evaluator.collect_step_metrics()
                    
                    states = next_states
                    step += 1
                    
                except Exception as e:
                    print(f"Error during environment step {step}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            
            print(f"\nEpisode completed after {step} steps")
            return info
            
        except Exception as e:
            print(f"Error in run_episode: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def close(self):
        """Clean up environment"""
        try:
            self.env.close()
            print("Environment closed successfully")
        except Exception as e:
            print(f"Error closing environment: {e}")

def run_fixed_time_evaluation(env_config, phases_config, num_episodes=5):
    """Run evaluation of fixed-time baseline with debug output"""
    print("\nStarting fixed-time evaluation...")
    print("\nEnvironment config:", env_config)
    print("\nPhases config:", phases_config)
    
    # Setup evaluation directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path("experiments/evaluation_results/fixed_time") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCreated output directory: {output_dir}")
    
    # Save configurations
    try:
        with open(output_dir / "env_config.yaml", 'w') as f:
            yaml.dump(env_config, f)
        with open(output_dir / "phases_config.yaml", 'w') as f:
            yaml.dump(phases_config, f)
        print("Saved configuration files")
    except Exception as e:
        print(f"Error saving configs: {e}")
    
    try:
        print("\nInitializing baseline and evaluator...")
        baseline = FixedTimeBaseline(env_config, phases_config)
        print(f"Found {len(baseline.traffic_lights)} traffic lights")
        
        evaluator = TrafficMetricsEvaluator(baseline.env, output_dir)
        print("Evaluator initialized")
        
        # Run evaluation episodes
        all_metrics = []
        
        for episode in range(num_episodes):
            print(f"\nStarting episode {episode + 1}/{num_episodes}")
            evaluator.reset_metrics()
            
            step = 0
            try:
                # Run episode
                info = baseline.run_episode(evaluator)
                print(f"Episode {episode + 1} completed. Calculating metrics...")
                
                # Calculate and save episode metrics
                episode_metrics = evaluator.calculate_final_metrics()
                all_metrics.append(episode_metrics)
                
                print(f"Episode {episode + 1} metrics:")
                print(f"- Average waiting time: {episode_metrics.get('mean_waiting_time', 'N/A')}")
                print(f"- Total throughput: {episode_metrics.get('total_throughput', 'N/A')}")
                
                # Plot episode metrics
                fig = evaluator.plot_metrics(f"Fixed-Time Baseline - Episode {episode + 1}")
                fig.savefig(output_dir / f"metrics_episode_{episode + 1}.png")
                print(f"Saved episode {episode + 1} plots")
                
                # Save episode metrics
                with open(output_dir / f"metrics_episode_{episode + 1}.json", 'w') as f:
                    json.dump(episode_metrics, f, indent=4)
                print(f"Saved episode {episode + 1} metrics")
                
            except Exception as e:
                print(f"Error in episode {episode + 1}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\nCalculating aggregate metrics...")
        # Calculate aggregate metrics across episodes
        aggregate_metrics = {
            metric: {
                'mean': float(np.mean([m[metric] for m in all_metrics if metric in m])),
                'std': float(np.std([m[metric] for m in all_metrics if metric in m]))
            }
            for metric in all_metrics[0].keys()
            if not isinstance(all_metrics[0][metric], dict)  # Skip nested metrics
        }
        
        print("\nGenerating final report...")
        # Generate and save report
        report = evaluator.generate_report(aggregate_metrics, 
                                         "Fixed-Time Baseline Evaluation Report")
        
        # Save aggregate metrics
        with open(output_dir / "aggregate_metrics.json", 'w') as f:
            json.dump(aggregate_metrics, f, indent=4)
        
        print("\nEvaluation completed successfully!")
        return aggregate_metrics, output_dir
    
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        print("\nClosing environment...")
        baseline.close()

if __name__ == "__main__":
    print("Starting fixed-time baseline evaluation script...")
    
    try:
        # Load configurations
        print("\nLoading environment configuration...")
        config_path = Path("config/env_config.yaml")
        if not config_path.exists():
            print(f"Error: Configuration file not found at {config_path}")
            sys.exit(1)
            
        with open(config_path, 'r') as f:
            env_config = yaml.safe_load(f)
        print("Environment configuration loaded successfully")
        
        # Define fixed-time phase configuration with corrected timings
        print("\nSetting up phase configuration...")
        phases_config = {
            'cycle_time': 120,            # 120 seconds total cycle
            'phases': [25, 25, 25, 25],   # 4 phases × 25 seconds = 100 seconds
            'yellow_time': 3              # 4 yellow phases × 3 seconds = 12 seconds
        }                                 # Total: 112 seconds < 120 cycle time
        
        # Verify configuration
        total_time = sum(phases_config['phases']) + (len(phases_config['phases']) * phases_config['yellow_time'])
        print(f"\nVerifying timing configuration:")
        print(f"Total phase time: {sum(phases_config['phases'])} seconds")
        print(f"Total yellow time: {len(phases_config['phases']) * phases_config['yellow_time']} seconds")
        print(f"Total cycle usage: {total_time} / {phases_config['cycle_time']} seconds")
        
        if total_time > phases_config['cycle_time']:
            print("Error: Total time exceeds cycle time!")
            sys.exit(1)
        
        # Run evaluation
        print("\nStarting evaluation...")
        metrics, output_dir = run_fixed_time_evaluation(env_config, phases_config)
        
        print(f"\nEvaluation completed. Results saved to: {output_dir}")
        print("\nSummary of Results:")
        if metrics:
            print(f"Average Waiting Time: {metrics['mean_waiting_time']['mean']:.2f} ± "
                  f"{metrics['mean_waiting_time']['std']:.2f} seconds")
            print(f"Total Throughput: {metrics['total_throughput']['mean']:.2f} ± "
                  f"{metrics['total_throughput']['std']:.2f} vehicles")
            print(f"Average Speed: {metrics['mean_speed']['mean']:.2f} ± "
                  f"{metrics['mean_speed']['std']:.2f} m/s")
        else:
            print("No metrics were collected during evaluation")
            
    except Exception as e:
        print(f"\nError in main execution: {e}")
        import traceback
        traceback.print_exc()