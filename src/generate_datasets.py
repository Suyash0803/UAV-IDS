"""
UAV Telemetry Dataset Generator for IDS Research

This module generates three synthetic datasets for UAV intrusion detection:
1. Dataset-1: Normal UAV telemetry (baseline behavior)
2. Dataset-2: Command injection and replay attacks
3. Dataset-3: GPS spoofing and mixed sensor anomalies

Each dataset contains:
- timestamp: Unix timestamp
- lat: Latitude (degrees)
- lon: Longitude (degrees)
- altitude: Altitude in meters
- velocity: Speed in m/s
- pitch: Pitch angle in degrees
- roll: Roll angle in degrees
- yaw: Yaw angle in degrees
- battery: Battery percentage
- command_id: Command identifier
- label: 0 (normal) or 1 (attack)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


class UAVDatasetGenerator:
    """Generate realistic UAV telemetry data with various attack patterns"""
    
    def __init__(self, random_seed=42):
        """
        Initialize the dataset generator with reproducible random seed
        
        Args:
            random_seed: Seed for reproducibility
        """
        np.random.seed(random_seed)
        
        # Normal UAV operational parameters (based on typical drone specifications)
        self.normal_ranges = {
            'lat': (40.0, 41.0),           # New York area
            'lon': (-74.0, -73.0),
            'altitude': (10, 120),          # meters, typical flight ceiling
            'velocity': (0, 15),            # m/s, typical cruise speed
            'pitch': (-15, 15),             # degrees
            'roll': (-15, 15),              # degrees
            'yaw': (0, 360),                # degrees
            'battery': (20, 100),           # percentage
            'command_id': (1000, 1050)      # command range
        }
        
    def _generate_normal_trajectory(self, n_samples):
        """
        Generate smooth, realistic normal UAV trajectory
        
        Uses smooth transitions to simulate realistic flight patterns:
        - GPS coordinates follow continuous paths
        - Altitude changes gradually
        - Velocity has inertia
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Dictionary of telemetry arrays
        """
        # Initialize arrays
        timestamps = [datetime.now() + timedelta(seconds=i) for i in range(n_samples)]
        
        # Generate smooth trajectories using cumulative sums with bounded random walk
        lat_drift = np.cumsum(np.random.normal(0, 0.0001, n_samples))
        lon_drift = np.cumsum(np.random.normal(0, 0.0001, n_samples))
        
        lat = 40.7128 + lat_drift + np.random.normal(0, 0.00005, n_samples)
        lon = -74.0060 + lon_drift + np.random.normal(0, 0.00005, n_samples)
        
        # Altitude: smooth changes with occasional maneuvers
        altitude_changes = np.cumsum(np.random.normal(0, 0.5, n_samples))
        altitude = np.clip(50 + altitude_changes, 10, 120)
        
        # Velocity: correlated with altitude changes
        velocity = np.clip(8 + np.random.normal(0, 2, n_samples), 0, 15)
        
        # Attitude angles: small random fluctuations (stable flight)
        pitch = np.random.normal(0, 3, n_samples)
        roll = np.random.normal(0, 3, n_samples)
        yaw = np.cumsum(np.random.normal(0, 2, n_samples)) % 360
        
        # Battery: gradual discharge
        battery = np.clip(100 - np.linspace(0, 30, n_samples) + np.random.normal(0, 1, n_samples), 20, 100)
        
        # Command IDs: mostly normal commands with occasional changes
        command_id = np.random.choice(range(1000, 1020), n_samples)
        
        return {
            'timestamp': timestamps,
            'lat': lat,
            'lon': lon,
            'altitude': altitude,
            'velocity': velocity,
            'pitch': pitch,
            'roll': roll,
            'yaw': yaw,
            'battery': battery,
            'command_id': command_id
        }
    
    def generate_dataset_1_normal(self, n_samples=10000, save_path='../data/dataset_1_normal.csv'):
        """
        Generate Dataset-1: Pure normal UAV telemetry
        
        This dataset represents baseline UAV behavior for training.
        All samples are labeled as normal (0).
        
        Args:
            n_samples: Total number of samples
            save_path: Path to save CSV file
            
        Returns:
            DataFrame with normal telemetry data
        """
        print(f"Generating Dataset-1: Normal telemetry ({n_samples} samples)...")
        
        data = self._generate_normal_trajectory(n_samples)
        data['label'] = 0  # All normal
        
        df = pd.DataFrame(data)
        df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())
        
        # Save to CSV
        output_path = os.path.join(os.path.dirname(__file__), save_path)
        df.to_csv(output_path, index=False)
        print(f"✓ Dataset-1 saved to {output_path}")
        print(f"  - Samples: {len(df)}, Attack rate: {df['label'].mean():.2%}\n")
        
        return df
    
    def generate_dataset_2_injection_replay(self, n_samples=10000, attack_ratio=0.3, 
                                           save_path='../data/dataset_2_injection_replay.csv'):
        """
        Generate Dataset-2: Command injection and replay attacks
        
        Attack patterns:
        1. Command Injection: Malicious command IDs outside normal range
        2. Replay Attack: Repeated identical command sequences (stale data)
        
        Args:
            n_samples: Total number of samples
            attack_ratio: Proportion of attack samples (0.3 = 30%)
            save_path: Path to save CSV file
            
        Returns:
            DataFrame with mixed normal and attack data
        """
        print(f"Generating Dataset-2: Command injection & replay attacks ({n_samples} samples, {attack_ratio:.0%} attacks)...")
        
        # Start with normal data
        data = self._generate_normal_trajectory(n_samples)
        labels = np.zeros(n_samples)
        
        # Inject attacks
        n_attacks = int(n_samples * attack_ratio)
        attack_indices = np.random.choice(n_samples, n_attacks, replace=False)
        
        for idx in attack_indices:
            attack_type = np.random.choice(['injection', 'replay'])
            
            if attack_type == 'injection':
                # Command Injection: malicious command IDs
                data['command_id'][idx] = np.random.choice(range(9000, 9100))  # Suspicious command range
                
                # May also cause erratic behavior
                if np.random.random() < 0.5:
                    data['velocity'][idx] *= np.random.uniform(1.5, 2.5)
                    data['altitude'][idx] += np.random.uniform(-20, 20)
                    
            elif attack_type == 'replay':
                # Replay Attack: copy data from previous timestep (stale data)
                if idx > 0:
                    # Repeat previous values exactly
                    data['lat'][idx] = data['lat'][idx-1]
                    data['lon'][idx] = data['lon'][idx-1]
                    data['altitude'][idx] = data['altitude'][idx-1]
                    data['velocity'][idx] = data['velocity'][idx-1]
                    data['command_id'][idx] = data['command_id'][idx-1]
                    
            labels[idx] = 1
        
        data['label'] = labels
        df = pd.DataFrame(data)
        df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())
        
        # Save to CSV
        output_path = os.path.join(os.path.dirname(__file__), save_path)
        df.to_csv(output_path, index=False)
        print(f"✓ Dataset-2 saved to {output_path}")
        print(f"  - Samples: {len(df)}, Attack rate: {df['label'].mean():.2%}\n")
        
        return df
    
    def generate_dataset_3_gps_spoofing(self, n_samples=10000, attack_ratio=0.3,
                                       save_path='../data/dataset_3_gps_spoofing.csv'):
        """
        Generate Dataset-3: GPS spoofing and mixed sensor anomalies
        
        Attack patterns:
        1. GPS Spoofing: Sudden jumps in location coordinates
        2. Sensor Anomalies: Inconsistent sensor readings (e.g., high velocity at low altitude)
        3. Telemetry Manipulation: Abnormal attitude angles or battery readings
        
        Args:
            n_samples: Total number of samples
            attack_ratio: Proportion of attack samples
            save_path: Path to save CSV file
            
        Returns:
            DataFrame with GPS spoofing and sensor anomaly attacks
        """
        print(f"Generating Dataset-3: GPS spoofing & sensor anomalies ({n_samples} samples, {attack_ratio:.0%} attacks)...")
        
        # Start with normal data
        data = self._generate_normal_trajectory(n_samples)
        labels = np.zeros(n_samples)
        
        # Inject attacks
        n_attacks = int(n_samples * attack_ratio)
        attack_indices = np.random.choice(n_samples, n_attacks, replace=False)
        
        for idx in attack_indices:
            attack_type = np.random.choice(['gps_spoof', 'sensor_anomaly', 'telemetry_manip'])
            
            if attack_type == 'gps_spoof':
                # GPS Spoofing: sudden large jumps in GPS coordinates
                data['lat'][idx] += np.random.uniform(-0.1, 0.1)  # ~11km jump
                data['lon'][idx] += np.random.uniform(-0.1, 0.1)
                
                # May affect multiple consecutive samples (persistent spoofing)
                if idx < n_samples - 3:
                    for offset in range(1, min(4, n_samples - idx)):
                        data['lat'][idx + offset] = data['lat'][idx] + np.random.normal(0, 0.001)
                        data['lon'][idx + offset] = data['lon'][idx] + np.random.normal(0, 0.001)
                        if idx + offset not in attack_indices:
                            labels[idx + offset] = 1
                            
            elif attack_type == 'sensor_anomaly':
                # Sensor Anomalies: physically impossible or inconsistent readings
                # E.g., high velocity at very low altitude (collision risk)
                data['altitude'][idx] = np.random.uniform(1, 5)  # Dangerously low
                data['velocity'][idx] = np.random.uniform(20, 30)  # Too fast for low altitude
                
                # Or extreme attitude angles
                data['pitch'][idx] = np.random.uniform(-45, -30)
                data['roll'][idx] = np.random.uniform(30, 45)
                
            elif attack_type == 'telemetry_manip':
                # Telemetry Manipulation: fake battery or command data
                # Sudden battery drop
                if np.random.random() < 0.5:
                    data['battery'][idx] = np.random.uniform(0, 15)
                    
                # Invalid command sequence
                data['command_id'][idx] = np.random.choice(range(8000, 8050))
                
                # Erratic yaw changes
                data['yaw'][idx] = (data['yaw'][idx] + np.random.uniform(90, 180)) % 360
                
            labels[idx] = 1
        
        data['label'] = labels
        df = pd.DataFrame(data)
        df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())
        
        # Save to CSV
        output_path = os.path.join(os.path.dirname(__file__), save_path)
        df.to_csv(output_path, index=False)
        print(f"✓ Dataset-3 saved to {output_path}")
        print(f"  - Samples: {len(df)}, Attack rate: {df['label'].mean():.2%}\n")
        
        return df


def main():
    """Generate all three datasets"""
    print("=" * 70)
    print("UAV IDS Dataset Generation")
    print("=" * 70 + "\n")
    
    generator = UAVDatasetGenerator(random_seed=42)
    
    # Generate all datasets
    df1 = generator.generate_dataset_1_normal(n_samples=10000)
    df2 = generator.generate_dataset_2_injection_replay(n_samples=10000, attack_ratio=0.3)
    df3 = generator.generate_dataset_3_gps_spoofing(n_samples=10000, attack_ratio=0.3)
    
    print("=" * 70)
    print("Dataset generation complete!")
    print("=" * 70)
    print("\nDataset Summary:")
    print(f"Dataset-1 (Normal):           {len(df1)} samples, {df1['label'].sum():.0f} attacks")
    print(f"Dataset-2 (Injection/Replay): {len(df2)} samples, {df2['label'].sum():.0f} attacks")
    print(f"Dataset-3 (GPS/Sensor):       {len(df3)} samples, {df3['label'].sum():.0f} attacks")


if __name__ == "__main__":
    main()
