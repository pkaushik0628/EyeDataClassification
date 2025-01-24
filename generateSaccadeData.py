import os
import pandas as pd
import numpy as np
from math import atan, sqrt

# Define constants
R = 0.02646  # Conversion factor to calculate distance


# Function to calculate angular velocity and count saccades
def count_saccades(file_path):
    df = pd.read_csv(file_path)

    # Dynamically find the TIME column
    time_column = [col for col in df.columns if col.startswith('TIME')][0]

    # Filter necessary columns
    df = df[[time_column, 'FPOGX', 'FPOGY']].dropna()

    # Convert time to seconds (assuming format 'YYYY/MM/DD HH:MM:SS.SSS')
    df['TIME'] = pd.to_datetime(df[time_column])
    df['TIME_DIFF'] = df['TIME'].diff().dt.total_seconds()

    # Calculate distance and angular position
    df['DISTANCE'] = R * np.sqrt((df['FPOGX'] - df['FPOGX'].shift(1)) ** 2 + (df['FPOGY'] - df['FPOGY'].shift(1)) ** 2)
    df['ANGULAR_POS'] = 2 * np.arctan(df['DISTANCE'] / (2 * R))

    # Calculate angular velocity
    df['ANGULAR_VELOCITY'] = df['ANGULAR_POS'].diff() / df['TIME_DIFF']

    # Count the number of saccades (angular velocity > 30 degrees/sec)
    saccades_count = (df['ANGULAR_VELOCITY'] > 30).sum()

    return saccades_count


# Directory path (modify to your local directory)
data_dir = 'Data'

# Initialize results dictionary
results = []

# Process each participant folder
for participant_id in os.listdir(data_dir):
    participant_folder = os.path.join(data_dir, participant_id)
    if os.path.isdir(participant_folder):
        participant_results = {'Participant ID': participant_id}

        # List of maze files
        maze_files = ['Maze5_fixations.csv', 'Maze6_fixations.csv', 'Maze8_fixations.csv', 'Maze11_fixations.csv',
                      'Maze12_fixations.csv']

        for maze_file in maze_files:
            maze_path = os.path.join(participant_folder, maze_file)
            if os.path.exists(maze_path):
                saccades_count = count_saccades(maze_path)
                maze_key = maze_file.split('_')[0] + '_Total Saccades'
                participant_results[maze_key] = saccades_count
            else:
                participant_results[maze_file.split('_')[0] + '_Total Saccades'] = 0

        results.append(participant_results)

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save to a CSV file
results_csv_path = 'saccade_results.csv'
results_df.to_csv(results_csv_path, index=False)

print("Saccade calculation complete! Results saved in 'saccade_results.csv'")

