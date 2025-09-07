#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_circumnavigation_data():
    # Load the CSV data
    csv_file = 'circumnavigation_data_20250907_143907.csv'
    data = pd.read_csv(csv_file)
    
    # Filter out entries where estimated state is [0,0]
    mask = ~((data['estimated_state_x'] == 0.0) & (data['estimated_state_y'] == 0.0))
    data_filtered = data[mask]
    
    print(f"Original data points: {len(data)}")
    print(f"Filtered data points: {len(data_filtered)} (removed {len(data) - len(data_filtered)} entries with estimated state [0,0])")
    
    # Extract x, y positions and estimated state positions from filtered data
    x_actual = data_filtered['x'].values
    y_actual = data_filtered['y'].values
    x_estimated = data_filtered['estimated_state_x'].values
    y_estimated = data_filtered['estimated_state_y'].values
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Plot actual x,y position in blue
    plt.plot(x_actual, y_actual, 'b-', linewidth=2, label='Actual Position (x,y)', alpha=0.7)
    
    # Plot estimated state x,y position in red
    plt.plot(x_estimated, y_estimated, 'r-', linewidth=2, label='Estimated State (x,y)', alpha=0.7)
    
    # Plot a circle at [0,5] in green
    circle_x, circle_y = 0, 5
    circle = plt.Circle((circle_x, circle_y), 0.2, color='green', fill=True, label='Target Point [0,5]')
    plt.gca().add_patch(circle)
    
    # Also add a marker for better visibility
    plt.plot(circle_x, circle_y, 'go', markersize=10, label='Target [0,5]')
    
    # Set equal aspect ratio and add grid
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # Add labels and title
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Circumnavigation Data: Actual vs Estimated Position')
    plt.legend()
    
    # Add some statistics
    print(f"Final data points plotted: {len(data_filtered)}")
    print(f"X range: {x_actual.min():.2f} to {x_actual.max():.2f}")
    print(f"Y range: {y_actual.min():.2f} to {y_actual.max():.2f}")
    print(f"Estimated X range: {x_estimated.min():.2f} to {x_estimated.max():.2f}")
    print(f"Estimated Y range: {y_estimated.min():.2f} to {y_estimated.max():.2f}")
    
    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_circumnavigation_data()
