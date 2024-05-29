import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import os
import numpy as np
from scipy.spatial.distance import euclidean

def linear_interpolate(start_pos, end_pos, start_frame, end_frame):
    """ Linearly interpolate positions between two frames. """
    frame_count = int(end_frame - start_frame - 1)
    delta = (end_pos - start_pos) / (frame_count + 1)
    return [start_pos + delta * (i + 1) for i in range(frame_count)]

def delete_tracks(df, track_ids):
    """ Delete specified tracks from the DataFrame. """
    return df[~df['ID'].isin(track_ids)]

def merge_tracks(df, track1_id, track2_id):
    """ Merge two tracks, average positions for overlapping frames, and interpolate if necessary. """
    track1 = df[df['ID'] == track1_id].sort_values(by='Frame')
    track2 = df[df['ID'] == track2_id].sort_values(by='Frame')

    # Find overlapping frames
    overlapping_frames = np.intersect1d(track1['Frame'], track2['Frame'])

    # Process each overlapping frame
    for frame in overlapping_frames:
        pos1 = track1[track1['Frame'] == frame][['centroid_x', 'centroid_y']].values[0]
        pos2 = track2[track2['Frame'] == frame][['centroid_x', 'centroid_y']].values[0]

        # Average positions for overlapping frames
        avg_pos = (pos1 + pos2) / 2
        df.loc[(df['ID'] == track1_id) & (df['Frame'] == frame), ['centroid_x', 'centroid_y']] = avg_pos

        # Remove overlapping frame from track2
        track2 = track2[track2['Frame'] != frame]

    # Merge tracks with or without interpolation
    if overlapping_frames.size > 0:
        # If tracks overlap, merge without interpolation
        new_track = pd.concat([track1, track2]).sort_values(by='Frame').reset_index(drop=True)
    else:
        # If tracks do not overlap, interpolate between them
        last_frame_track1 = int(track1['Frame'].max())
        first_frame_track2 = int(track2['Frame'].min())

        interpolated_positions = linear_interpolate(
            track1.iloc[-1][['centroid_x', 'centroid_y']],
            track2.iloc[0][['centroid_x', 'centroid_y']],
            last_frame_track1,
            first_frame_track2
        )

        # Create interpolated track
        interpolated_track = pd.DataFrame(interpolated_positions, columns=['centroid_x', 'centroid_y'])
        interpolated_track['Frame'] = range(last_frame_track1 + 1, first_frame_track2)
        interpolated_track['ID'] = track1_id

        # Merge with interpolation
        new_track = pd.concat([track1, interpolated_track, track2]).sort_values(by='Frame').reset_index(drop=True)

    # Remove old tracks
    df = df[df['ID'] != track1_id]
    df = df[df['ID'] != track2_id]

    # Ensure the merged track has the lower ID
    new_track['ID'] = track1_id

    # Add the merged track to the DataFrame
    df = pd.concat([df, new_track]).sort_values(by=['Frame', 'ID']).reset_index(drop=True)

    return df


def plot_tracks(df, output_directory, filename, num_people, near_threshold):
    fig = go.Figure()

    frame_counts = df.groupby('ID').size()
    lowest_frame_count_ids = frame_counts.nsmallest(40).index
    lowest_frame_count_tracks = df[df['ID'].isin(lowest_frame_count_ids)]

    # Print the frame count of the shortest tracks
    print("\nFrame count of the shortest tracks:")
    for track_id in lowest_frame_count_ids:
        print(f"Track ID {track_id}: {frame_counts[track_id]} frames")

    potential_matches = {}  # To store potential match colors

    for short_id in lowest_frame_count_ids:
        short_track = lowest_frame_count_tracks[lowest_frame_count_tracks['ID'] == short_id]
        end_point = short_track.iloc[-1][['centroid_x', 'centroid_y']].values.astype(float)

        # Skip if end_point contains NaN or inf
        if np.any(np.isnan(end_point)) or np.any(np.isinf(end_point)):
            continue

        match_ids = []
        for other_id in df['ID'].unique():
            if other_id != short_id:
                other_track = df[df['ID'] == other_id]
                if not other_track.empty:  # Check if other_track is not empty
                    start_point = other_track.iloc[0][['centroid_x', 'centroid_y']].values.astype(float)

                # Skip if start_point contains NaN or inf
                if np.any(np.isnan(start_point)) or np.any(np.isinf(start_point)):
                    continue

                if euclidean(end_point, start_point) <= near_threshold:
                    match_ids.append(other_id)

        for match_id in match_ids:
            potential_matches[match_id] = np.random.rand(3,)

        short_color = 'red' if short_id > num_people else 'blue'
        if match_ids:
            short_color = np.random.rand(3,)

        # Plot the short track
        fig.add_trace(go.Scatter(
            x=short_track['centroid_x'],
            y=short_track['centroid_y'],
            mode='lines+markers',
            name=f'End ID {short_id}',
            marker=dict(color=short_color),
            hoverinfo='text',
            text=[f'Track ID: {short_id}' for _ in range(len(short_track))]  # Hover text
        ))

    for id, color in potential_matches.items():
        track = df[df['ID'] == id]
        
        # Plot the matched track
        fig.add_trace(go.Scatter(
            x=track['centroid_x'],
            y=track['centroid_y'],
            mode='lines+markers',
            name=f'Match ID {id}',
            marker=dict(color=color),
            hoverinfo='text',
            text=[f'Track ID: {id}' for _ in range(len(track))]  # Hover text
        ))

    fig.update_layout(
        title=f'Trajectories in {filename}',
        xaxis_title='X Center',
        yaxis_title='Y Center'
    )

    fig.show()

def main():
    file_path = input("Enter the file path: ")
    output_directory = input("Enter the output directory path: ")
    near_threshold = 70  # Adjust as needed
    num_people = int(input("How many people were in this session? "))

    df = pd.read_csv(file_path)

    while True:
        plot_tracks(df, output_directory, os.path.basename(file_path), num_people, near_threshold)
        choice = input("Do you want to stitch, delete, or exit? ")

        if choice.lower() == 'stitch':
            track1_id = int(input("Enter the first track ID to stitch: "))
            track2_id = int(input("Enter the second track ID to stitch: "))
            df = merge_tracks(df, track1_id, track2_id)

        elif choice.lower() == 'delete':
            track_ids_input = input("Enter the track IDs to delete, separated by commas: ")
            track_ids = [int(id.strip()) for id in track_ids_input.split(',')]
            df = delete_tracks(df, track_ids)

        elif choice.lower() == 'exit':
            break
        else:
            print("Invalid choice. Please enter 'stitch', 'delete', or 'exit'.")
    # Construct the full path for the output file
    output_file_path = os.path.join(output_directory, 'R' + os.path.basename(file_path))
    
    try:    
        df.to_csv(output_file_path, index=False)
        print(f"File saved successfully at {output_file_path}.")
    except Exception as e:
        print(f"Error saving file: {e}")

main()