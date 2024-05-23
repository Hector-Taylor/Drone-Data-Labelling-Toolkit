import cv2
import pandas as pd
import numpy as np
import os
import re


# Class names Session 3 
'''
class_names = [
'Yellow0',
'YellowI',
'YellowII',
'YellowX',
'GreenI',
'GreenII',
'GreenX',
'Blue0',
'BlueI',
'BlueII',
'BlueX',
'Orange0',
'OrangeI',
'OrangeII',
'OrangeX',
'Red0',
'RedI',
'RedII',
'RedX']


Leader_IDs = ['Blue0', 'BlueII', 'BlueI', 'OrangeI']
'''
'''
'''
'''
#April 7th:
class_names = [
    'Black1V',
    'Black2V',
    'BlackX',
    'Black1Dot',
    'Black2Dot',
    'Black1H',
    'Black2H',
    'BlackY',
    'BlackH',
    'Orange1V',
    'Orange1Dot',
    'Orange2Dot',
    'Orange2H',
    'OrangeY',
    'OrangeH',
    'Orange1H'
]
Leader_IDs = ['Black1V', 'Black1Dot', 'Orange2Dot', 'Orange1H']
'''

'''
class_names = ['Black1V',
'Black1V',
'Black2V',
'BlackX',
'Black1Dot',
'Black2Dot',
'Black1H',
'Black2H',
'BlackY',
'BlackH',
'Orange1V',
'OrangeX',
'Orange2Dot',
'Orange1H',
'Orange2H',
'OrangeY',
'Blue2V',
'Blue1H',
'BlueY',
'Unmarked']
Leader_IDs = ['BlackX', 'Black2Dot', 'Orange1H', 'Black1H']
'''
#S5 - APRIL 10th session
'''
class_names = [    
'Black1V',               
'Black2V',
'BlackX',
'Black1Dot',
'Black2Dot',
'Black1H',
'Black2H',
'BlackY',
'BlackH',
'Orange1V',
'Orange2V',
'OrangeX',
'Orange2Dot',
'Orange1H',
'OrangeY',
'OrangeH',
'Blue2V',
'BlueX',
'Blue1H',
'BlueY',
'Unmarked']
Leader_IDs = ['BlackX', 'Black1Dot', 'BlackY', 'BlueY']
'''
'''
#Session 7, 4/24/23
class_names = [
'Black1V',
'Black2V',
'BlackX',
'Black1Dot',
'Black2Dot',
'Black1H',
'Black2H',
'BlackY',
'BlackH',
'Orange1V',
'Orange2V',
'OrangeX',
'Orange2Dot',
'Orange1H',
'Orange2H',
'OrangeY',
'OrangeH',
'Blue2V',
'BlueX',
'BlueY']
Leader_IDs = ['BlackH', 'Black2Dot', 'Orange2H', 'Black2H']
'''

'''
class_names = [
'Black1V',
'Black2V',
'BlackX',
'Black1Dot',
'Black2Dot',
'Black1H',
'Black2H',
'BlackY',
'BlackH',
'Orange1V',
'OrangeX',
'Orange2Dot',
'Orange1H',
'Orange2H',
'OrangeY',
'Blue2V',
'BlueY',
'Blue1H',
'Unmarked']

Leader_IDs = ['BlackX', 'Black2Dot', 'Black1H', 'Orange1H']
'''
'''
class_names = [
'Blue1Dot',
'Orange1V',
'Orange1H',
'Black1V',
'Black1H',
'Black2V',
'Black1Dot',
'Unmarked',
'Black2Dot',
'Blue1H',
'BlackY',
'Orange2V',
'Blue2Dot',
'BlackX',
'BlueY',
'Blue1H',
'Blue2H',
'Orange2H',
'Black2H',
'Blue1V',
'BlueH',
'Blue2V',
'BlackH']

Leader_IDs = ['Black1V', 'Blue1Dot', 'Black1Dot', 'Black2V']
'''
#0421 Session 6 
'''
class_names = [
'Black1V',
'Black2V',
'BlackX',
'Black1Dot',
'Black2Dot',
'Black1H',
'Black2H',
'BlackY',
'BlackH',
'Orange1V',
'Orange2V',
'OrangeX',
'Orange2Dot',
'Orange1H',
'Orange2H',
'OrangeY',
'OrangeH',
'Blue2V',
'BlueX',
'BlueY',
]
Leader_IDs = ['Black2H', 'Black2Dot', 'BlackH', 'Orange2H']
'''
#Session 9 identities
'''
class_names = [
    'Black1Dot',
    'Black1V',
    'Black2V',
    'Blue1Dot',
    'Blue1H',
    'Orange2V',
    'BlackX',
    'Blue2H',
    'BlackY',
    'Blue2V',
    'Blue1V',
    'Orange2H',
    'Orange1V',
    'Black2H',
    'BlueH',
    'BlackH',
    'Orange1H',
    'Black2Dot',
    'Black1H',
    'Unmarked',
    'BlueY',
    'Blue2Dot'
]
Leader_IDs = ['Black1Dot', 'Black1V', 'Black2V', 'Blue1Dot']
'''

#Session 8 identities
'''
class_names = [
    'Blue1H',
    'Blue2H',
    'Blue2V',
    'BlueH',
    'BlueY',
    'Blue1V',
    'Blue1H',
    'Black1Dot',
    'OrangeX',
    'BlueX',
    'Black1V',
    'BlackH',
    'Black2H',
    'Black1H',
    'BlackY',
    'Unmarked',
    'Black2V',
    'Black2Dot',
    'BlackX'
]
Leader_IDs = ['BlueH']
'''
'''
class_names = [
'Blue2V',
'Black2H',
'BlackX',
'Blue1V',
'Blue2Dot',
'Black1H',
'BlackH',
'Black2V',
'BlueY',
'BlackY',
'Black1Dot', 
'Blue1Dot',
'Black1V',
'Black2H',
'Black2Dot',
'Unmarked',
'Blue2H'
]
Leader_IDs = ['Black1Dot', 'Black1V', 'Blue1V', 'Blue1Dot']'''
#Session 10 identities

class_names = [
    'Blue1H',
    'Blue2H',
    'BlueH',
    'BlueY',
    'Blue1V',
    'Blue1H',
    'Black1Dot',
    'BlueX',
    'Black1V',
    'BlackH',
    'Black2H',
    'Black1H',
    'BlackY',
    'Unmarked',
    'Black2V',
    'Black2Dot',
    'BlackX'
]
Leader_IDs = ['Black2Dot', 'BlackX', 'BlueH', 'Black1V']


#session 11 identities
'''
class_names = [
    'Unmarked',
    'BlackX',
    'Black2Dot',
    'Black1H',
    'Black2V',
    'BlackH',
    'Blue1V',
    'Blue2V',
    'BlueH',
    'BlackH',
    'Blue1H',
    'Black2H',
    'BlueY',
    'Black1Dot',
    'Black1V',
    'OrangeX',
    'BlackY',
    'Blue1H',
    'Blue2H'
]
Leader_IDs = ['BlueH', 'Blue1H', 'Blue2H', 'Unmarked']
'''

DISPLAY_DURATION = 1000  # Duration to display each frame in milliseconds
WINDOW_SIZE = 1280 
available_labels = set(class_names)
assigned_labels = {}  # Store assigned labels and their respective unique_idblack

file_path = 'R:\HectorTaylor\Drone Datasets\EXTRA\Xtra4Kei - 040524\JustS10InHere\S10T33tracks_PREPROCESSED.txt'
output_directory = 'R:\HectorTaylor\Drone Datasets\EXTRA\Xtra4Kei - 040524\JustS10InHere'

match = re.search(r'(S10T\d+)tracks_PREPROCESSED\.txt', file_path) #S3T\d+tracks_PREPROCESSED\.txt

if match:
    video_number = match.group(1) 
    video_path = file_path.rsplit('\\', 1)[0] + '\\' + video_number + '.MP4'
    print("Video path:", video_path)
else:
    video_path = None  
    print("No match!!")
    
tracks_df = pd.read_csv(file_path, sep=",", skiprows=1, header=None)
#tracks_df.columns = ["Frame", "ID", "x_center", "y_center","x_centerfake"]
tracks_df.columns = ["Frame", "ID", "x_center", "y_center"]

# Converting the columns to appropriate data types
tracks_df = tracks_df.astype({"Frame": float, "ID": float, "x_center": float, "y_center": float})
tracks_df[["Frame", "ID"]] = tracks_df[["Frame", "ID"]].astype(int)
#tracks_df = tracks_df.drop("x_centerfake", axis=1)

# Extracting the original filename without extension
original_filename = os.path.basename(file_path).split('.')[0]

# Creating a new filename
output_filename = original_filename + '_Identified.txt'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Save the DataFrame to the specified file
tracks_df.to_csv(os.path.join(output_directory, output_filename), sep=",", index=False)

cap = cv2.VideoCapture(video_path)

used_labels = {}  # Store used labels and their respective unique_id

# Looping through each unique ID in the tracking data
for unique_id in tracks_df["ID"].unique():
    median_frame_num = int(np.round(tracks_df[tracks_df["ID"] == unique_id]["Frame"].median()))  
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, median_frame_num)
    ret, frame = cap.read()

    if not ret or frame is None:
        print(f"Warning: Unable to read frame med for ID {unique_id}.")
        continue

    # Extract centroid coordinates directly from the DataFrame
    query_result = tracks_df[(tracks_df["ID"] == unique_id) & (tracks_df["Frame"] == median_frame_num)][["x_center", "y_center"]]

    if query_result.empty:
        print(f"No data found for ID {unique_id} at frame med.")
        continue

    centroid_coordinates = query_result.values.astype(int).flatten()

    if len(centroid_coordinates) != 2:
        print(f"Unexpected data format for ID {unique_id} at frame med: {centroid_coordinates}")
        continue

    x_center, y_center = centroid_coordinates

    # Define the bounding box corners
    box_size = 20  # Size of the box in pixels
    x1, y1 = x_center - box_size // 2, y_center - box_size // 2
    x2, y2 = x_center + box_size // 2, y_center + box_size // 2
    
    x_start, y_start = max(x_center - WINDOW_SIZE // 2, 0), max(y_center - WINDOW_SIZE // 2, 0)
    x_end, y_end = x_start + WINDOW_SIZE, y_start + WINDOW_SIZE

    frame_height, frame_width, _ = frame.shape
    x_end, y_end = min(x_end, frame_width), min(y_end, frame_height)
    x_start, y_start = x_end - WINDOW_SIZE, y_end - WINDOW_SIZE

    cropped_frame = frame[y_start:y_end, x_start:x_end].copy()

    # Draw the bounding box
    cv2.rectangle(cropped_frame, (x1 - x_start, y1 - y_start), (x2 - x_start, y2 - y_start), (0, 255, 0), 1)


# Display the frame
    cv2.imshow('Frame', cropped_frame)

    # Wait until Enter key (key code 13) is pressed
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            break

    # Close the frame window
    cv2.destroyAllWindows()
    while True:
        # Display the available labels
        print("Available labels:", ', '.join(sorted(available_labels)))
        class_label = input(f"Enter class label for ID {unique_id} (options: {sorted(available_labels)}): ")
        
        if class_label.upper() == "EXIT":
            print("Exiting labeling process...")
            break

        if class_label == 'NOT':
            # Remove all entries of this track ID from the DataFrame
            tracks_df = tracks_df[tracks_df['ID'] != unique_id]
            print(f"Track ID {unique_id} deleted.")
            break

        if class_label == 'dontknow':
            break

        if class_label in available_labels:
            if class_label in assigned_labels and assigned_labels[class_label] != unique_id:
                print(f"Label {class_label} has already been used for ID {assigned_labels[class_label]}.")
                continue
            assigned_labels[class_label] = unique_id  # Track the assignment
            tracks_df.loc[tracks_df["ID"] == unique_id, "ClassLabel"] = class_label
            available_labels.remove(class_label)  # Remove the used label from available labels
            break
        else:
            print(f"Invalid label. Please choose from {sorted(available_labels)}")
    


# Ensure no deleted IDs are processed in the duplicate label check
if 'ClassLabel' in tracks_df.columns:
    label_counts = tracks_df['ClassLabel'].value_counts()
    duplicate_labels = label_counts[label_counts > 1].index.tolist()

# Check for duplicate labels after all assignments are done
duplicate_labels = set()
for label, id in assigned_labels.items():
    if list(assigned_labels.values()).count(id) > 1:
        duplicate_labels.add(label)

if duplicate_labels:
    print(f"Warning: The following labels have been used more than once: {', '.join(duplicate_labels)}")

    for label in duplicate_labels:
        uids = tracks_df[tracks_df['ClassLabel'] == label]['ID'].unique()
        for uid in uids:
            # Displaying the frame with the bounding box
            median_frame_num = tracks_df[tracks_df["ID"] == uid]["Frame"].median()
            median_frame_num = int(np.round(median_frame_num))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, median_frame_num)
            ret, frame = cap.read()
            
            # Extract centroid coordinates directly from the DataFrame
            centroid_coordinates = tracks_df[(tracks_df["ID"] == uid) & 
                                             (tracks_df["Frame"] == median_frame_num)][["x_center", "y_center"]].values.astype(int).flatten()
            x_center, y_center = centroid_coordinates
            
            # Define the bounding box corners
            box_size = 20  # Size of the box in pixels
            x1, y1 = x_center - box_size // 2, y_center - box_size // 2
            x2, y2 = x_center + box_size // 2, y_center + box_size // 2

            x_start, y_start = max(x_center - WINDOW_SIZE // 2, 0), max(y_center - WINDOW_SIZE // 2, 0)
            x_end, y_end = x_start + WINDOW_SIZE, y_start + WINDOW_SIZE
            
            frame_height, frame_width, _ = frame.shape
            x_end, y_end = min(x_end, frame_width), min(y_end, frame_height)
            x_start, y_start = x_end - WINDOW_SIZE, y_end - WINDOW_SIZE
            
            cropped_frame = frame[y_start:y_end, x_start:x_end].copy()
            
            cv2.rectangle(cropped_frame, (x1 - x_start, y1 - y_start), (x2 - x_start, y2 - y_start), (0, 255, 0), 1)
            
            cv2.imshow('Frame', cropped_frame)
            cv2.waitKey(0)

            
            # Asking the user to validate or change the label
            while True:
                print(f"Current label for ID {uid}: {label}")
                user_input = input("Do you want to change the label? (yes/no): ").strip().lower()
                if user_input == "yes":
                    new_label = input("Enter the new label: ").strip()
                    if new_label in tracks_df['ClassLabel'].values and new_label != label:
                        print(f"Label {new_label} is already used. Please choose another label.")
                    else:
                        tracks_df.loc[tracks_df["ID"] == uid, "ClassLabel"] = new_label
                        break
                elif user_input == "no":
                    break
                else:
                    print("Invalid input. Please enter 'yes' or 'no'.")
            
            cv2.destroyAllWindows()
else:
    print("No duplicate labels found.")


# Add a 'Leader' column and set 'y' or 'n' based on whether 'ClassLabel' is in Leader_IDs
tracks_df['Leader'] = tracks_df['ClassLabel'].apply(lambda x: 'y' if x in Leader_IDs else 'n')
tracks_df.to_csv(os.path.join(output_directory, output_filename), sep=",", index=False)

# Releasing the video capture object
cap.release()