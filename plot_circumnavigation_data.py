#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

def animate_circumnavigation_data():
    # Load the CSV data
    csv_file = 'kvr_sim.csv'
    data = pd.read_csv(csv_file)
    
    # Filter out entries where estimated state is [0,0]
    mask = ~((data['estimated_state_x'] == 0.0) & (data['estimated_state_y'] == 0.0))
    data_filtered = data[mask]
    
    print(f"Original data points: {len(data)}")
    print(f"Filtered data points: {len(data_filtered)} (removed {len(data) - len(data_filtered)} entries with estimated state [0,0])")
    
    # Extract data
    x_actual = data_filtered['x'].values
    y_actual = data_filtered['y'].values
    x_estimated = data_filtered['estimated_state_x'].values
    y_estimated = data_filtered['estimated_state_y'].values
    timestamps = data_filtered['timestamp'].values
    distance_error = data_filtered['distance_error'].values
    bearing = data_filtered['bearing'].values
    compass_hdg = data_filtered['compass_hdg'].values
    
    # Convert timestamps to relative time
    time_relative = timestamps - timestamps[0]
    
    # Create figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 12))
    
    # Initialize empty lines for animation
    line_actual, = ax1.plot([], [], 'b-', linewidth=2, label='Drone GPS Position (x,y)', alpha=0.7)
    line_estimated, = ax1.plot([], [], 'r-', linewidth=2, label='Estimated State (x,y)', alpha=0.7)
    point_current, = ax1.plot([], [], 'ko', markersize=8, label='Current Position')
    
    # Add estimated target circle at [0,5]

    target_circle = plt.Circle((0, 5), 0.5, color='orange', fill=False, linestyle='--', linewidth=2, label='Target Location (0,5)')
    ax1.add_patch(target_circle)
    
    # Drone orientation arrow (made bigger)
    arrow_length = 3.0  # Increased from 0.5
    arrow = patches.FancyArrowPatch((0, 0), (0, 0), 
                                   arrowstyle='->', mutation_scale=30,  # Increased from 20
                                   color='red', linewidth=3, alpha=0.9)  # Increased linewidth
    ax1.add_patch(arrow)
    
    line_distance_error, = ax2.plot([], [], 'g-', linewidth=2, label='Distance Error', alpha=0.7)
    point_distance_current, = ax2.plot([], [], 'go', markersize=6)
    
    line_bearing, = ax3.plot([], [], 'm-', linewidth=2, label='Bearing', alpha=0.7)
    point_bearing_current, = ax3.plot([], [], 'mo', markersize=6)
    
    # Set up axes
    ax1.set_xlim(x_actual.min() - 1, x_actual.max() + 1)
    ax1.set_ylim(y_actual.min() - 1, y_actual.max() + 1)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Circumnavigation Data: 2D state estimation')
    ax1.legend()
    
    ax2.set_xlim(0, time_relative.max())
    ax2.set_ylim(distance_error.min() - 0.1, distance_error.max() + 0.1)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Distance Error (m)')
    ax2.set_title('Distance Error Over Time')
    ax2.legend()
    
    ax3.set_xlim(0, time_relative.max())
    ax3.set_ylim(bearing.min() - 10, bearing.max() + 10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Bearing (degrees)')
    ax3.set_title('Bearing Over Time')
    ax3.legend()
    
    # Animation function
    def animate(frame):
        if frame < len(x_actual):
            # Update trajectory lines
            line_actual.set_data(x_actual[:frame+1], y_actual[:frame+1])
            line_estimated.set_data(x_estimated[:frame+1], y_estimated[:frame+1])
            
            # Update current position marker
            if frame > 0:
                point_current.set_data([x_actual[frame]], [y_actual[frame]])
                
                # Update drone orientation arrow
                current_x, current_y = x_actual[frame], y_actual[frame]
                compass_rad = np.deg2rad(compass_hdg[frame])
                
                # Calculate arrow end point (compass heading points north=0°, east=90°)
                # Convert to standard math coordinates (east=0°, north=90°)
                math_angle = 90 - compass_hdg[frame]
                math_rad = np.deg2rad(math_angle)
                
                end_x = current_x + arrow_length * np.cos(math_rad)
                end_y = current_y + arrow_length * np.sin(math_rad)
                
                arrow.set_positions((current_x, current_y), (end_x, end_y))
            
            # Update time series plots
            line_distance_error.set_data(time_relative[:frame+1], distance_error[:frame+1])
            point_distance_current.set_data([time_relative[frame]], [distance_error[frame]])
            
            line_bearing.set_data(time_relative[:frame+1], bearing[:frame+1])
            point_bearing_current.set_data([time_relative[frame]], [bearing[frame]])
            
            # Add time annotation
            fig.suptitle(f'Circumnavigation Animation - Time: {time_relative[frame]:.1f}s', fontsize=14)
        
        return line_actual, line_estimated, point_current, arrow, line_distance_error, point_distance_current, line_bearing, point_bearing_current
    
    # Create animation
    print("Creating animation... This may take a while.")
    ani = FuncAnimation(fig, animate, frames=len(x_actual), interval=150, blit=False, repeat=True)
    
    # Save animation as MOV for DaVinci Resolve free version compatibility
    print("Saving animation as MOV for DaVinci Resolve free version...")
    
    try:
        print("Creating MOV file optimized for DaVinci Resolve free...")
        ani.save('circumnavigation_animation.mov', writer='ffmpeg', fps=25, 
                extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', 
                           '-preset', 'medium', '-movflags', '+faststart'])
        print("MOV saved successfully! File: circumnavigation_animation.mov")
        print("This file is compatible with DaVinci Resolve free version on Ubuntu.")
    except Exception as e:
        print(f"Error saving MOV: {e}")
        print("Showing animation instead...")
        plt.tight_layout()
        plt.show()
    
    return ani

def plot_circumnavigation_data():
    # Load the CSV data
   #csv_file = 'circumnavigation_data_20250908_152645.csv'
    #csv_file = 'circumnavigation_data_20250908_153522.csv'  # IGNORE
    csv_file = 'KVR_2.csv'


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
    timestamps = data_filtered['timestamp'].values
    distance_error = data_filtered['distance_error'].values
    bearing = data_filtered['bearing'].values
    
    # Convert timestamps to relative time (seconds from start)
    time_relative = timestamps - timestamps[0]
    
    # Create the plot with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 12))
    
    # Plot actual x,y position in blue
    ax1.plot(x_actual, y_actual, 'b-', linewidth=2, label='Actual Position (x,y)', alpha=0.7)
    
    # Plot estimated state x,y position in red
    ax1.plot(x_estimated, y_estimated, 'r-', linewidth=2, label='Estimated Target State (x,y)', alpha=0.7)
    
    
    # Set equal aspect ratio and add grid for first subplot
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Plot the target location as a dot at (0, 5)
    ax1.plot(0, 5, 'o', color='orange', markersize=10, label='Target Location (0,5)')
    # Add labels and title for first subplot
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Circumnavigation Data: Actual vs Estimated Position')
    ax1.legend()
    
    # Second subplot: Distance error over time
    ax2.plot(time_relative, distance_error, 'g-', linewidth=2, label='Estimated Distance Error', alpha=0.7)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Estimated Distance Error (m)')
    ax2.set_title('Estimated Distance Error Over Time')
    ax2.legend()
    
    # Third subplot: Bearing over time
    ax3.plot(time_relative, bearing, 'm-', linewidth=2, label='Bearing', alpha=0.7)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Bearing (degrees)')
    ax3.set_title('Bearing Over Time')
    ax3.legend()
    
    # Add some statistics
    print(f"Final data points plotted: {len(data_filtered)}")
    print(f"X range: {x_actual.min():.2f} to {x_actual.max():.2f}")
    print(f"Y range: {y_actual.min():.2f} to {y_actual.max():.2f}")
    print(f"Estimated X range: {x_estimated.min():.2f} to {x_estimated.max():.2f}")
    print(f"Estimated Y range: {y_estimated.min():.2f} to {y_estimated.max():.2f}")
    print(f"Time range: {time_relative.min():.2f} to {time_relative.max():.2f} seconds")
    print(f"Distance error range: {distance_error.min():.2f} to {distance_error.max():.2f} m")
    print(f"Mean distance error: {distance_error.mean():.2f} m")
    print(f"Std distance error: {distance_error.std():.2f} m")
    print(f"Bearing range: {bearing.min():.2f} to {bearing.max():.2f} degrees")
    print(f"Mean bearing: {bearing.mean():.2f} degrees")
    print(f"Std bearing: {bearing.std():.2f} degrees")
    
    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'animate':
        animate_circumnavigation_data()
    else:
        print("Running static plot. Use 'python plot_circumnavigation_data.py animate' for animation.")
        plot_circumnavigation_data()
