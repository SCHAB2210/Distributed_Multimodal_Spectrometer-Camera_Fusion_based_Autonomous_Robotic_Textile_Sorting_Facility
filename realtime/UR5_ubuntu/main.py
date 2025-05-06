#!/usr/bin/env python3
import sys
import numpy as np
import cv2
import pyrealsense2 as rs
import os
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R
import time
import socket
import math
import threading

# Receive label from Windows over socket

def init_nir_connection():
    HOST = '192.168.56.101'  # Ubuntu IP (same as used in client)
    PORT = 12345
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)
    #print(f"[SYSTEM] Waiting for client connection ")
    conn, addr = s.accept()
    print(f"[SYSTEM] Connected to client")
    return conn

def wait_for_label(conn):
    try:
        while True:
            data = conn.recv(1024)
            if not data:
                print("[SYSTEM] Connection closed by client.")
                return None
            label = data.decode('utf-8').strip()
            print(f"[NIR] Received label: {label}")
            try:
                numeric_label = int(label)
                if 0 <= numeric_label <= 5:
                    return numeric_label
                else:
                    print(f"[NIR] Invalid label: {numeric_label}")
            except ValueError:
                print(f"[NIR] Failed to convert label to int: {label}")
    except Exception as e:
        print(f"[NIR] Error reading from socket: {e}")
        return None

# Positions & Classes (for robot control)
TEXTILE_CLASSES = {
    0: "white_cotton",
    1: "black_cotton",
    2: "other_cotton",
    3: "white_polyester",
    4: "black_polyester",
    5: "other_polyester"
}

STANDBY_POSITION_JOINTS = [106.34, -83.12, -109.86, -75.03, 87.04, 86.85]
INTERMEDIATE_STEP_CARTESIAN = [500, -300, 350, 0.36, -0.0785, -5.9]

SORTING_POSITIONS = {
    0: [83  , -250, 390, 0.441, -0.07, 0.356],
    1: [-166, -250, 390, 0.441, -0.07, 0.356],
    2: [-423, -250, 390, 0.441, -0.07, 0.356],
    3: [83  , -600, 390, 0.441, -0.07, 0.356],
    4: [-166, -600, 390, 0.441, -0.07, 0.358],
    5: [-423, -600, 390, 0.441, -0.07, 0.356]
}

def get_textile_class(class_number):
    return TEXTILE_CLASSES.get(class_number)

def get_placement_position(class_number):
    return SORTING_POSITIONS.get(class_number)

# Robot Control

ROBOT_IP = "192.168.10.100"
PORT = 30002

def send_command(command):
    if "set_digital_out(0, True)" in command:
        print(f"[GRIPPER] Closing gripper")
    elif "set_digital_out(0, False)" in command:
        print(f"[GRIPPER] Opening gripper")
    elif "smooth_move" in command:
        print(f"[ROBOT] moving to placement position")
    
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((ROBOT_IP, PORT))
            s.send((command + "\n").encode("utf-8"))
    except Exception as e:
        print(f"[ERROR] Failed to send command: {e}")

def move_to_position(x, y, z, roll=0.174, pitch=-0.021, yaw=0.459, blend_radius=0):
    if abs(x) > 10 or abs(y) > 10 or abs(z) > 10:
        x, y, z = x / 1000, y / 1000, z / 1000
    command = f"movel(p[{x}, {y}, {z}, {roll}, {pitch}, {yaw}], a=1.2, v=0.25, r={blend_radius})"
    send_command(command)
    # Reduce sleep time based on blend radius
    if blend_radius > 0:
        #print(f"[WAIT] Waiting for blended motion to complete (2s)")
        time.sleep(2)  # Shorter wait for blended motions
    else:
        #print(f"[WAIT] Waiting for precise positioning to complete (4s)")
        time.sleep(4)  # Full wait for precise positioning

def move_joints(joint_angles, blend_radius=0):
    angles_rad = [math.radians(angle) for angle in joint_angles]
    command = f"movej({angles_rad}, a=1.2, v=0.5, r={blend_radius})"
    send_command(command)
    if blend_radius > 0:
        #print(f"[WAIT] Waiting for blended joint motion to complete (2s)")
        time.sleep(2)
    else:
        #print(f"[WAIT] Waiting for precise joint positioning to complete (4s)")
        time.sleep(4)

def pick_textile():
    #print(f"[TASK] Picking up textile")
    send_command("set_digital_out(0, True)")
    send_command("set_digital_out(1, False)")
    #print(f"[WAIT] Waiting for gripper to close (0.5s)")
    time.sleep(0.5)  # Allow time for the gripper to close

def release_textile():
    #print(f"[TASK] Releasing textile")
    send_command("set_digital_out(0, False)")
    send_command("set_digital_out(1, True)")
    #print(f"[WAIT] Waiting for gripper to open (0.5s)")
    time.sleep(0.5)

def place_textile(label):
    # Use blend radius for intermediate movements
    move_to_position(*INTERMEDIATE_STEP_CARTESIAN, blend_radius=0.05)
    place_coords = get_placement_position(label)
    move_to_position(*place_coords)  # No blend on final placement for precision
    release_textile()

    # Define return waypoints for smooth motion back to standby
    return_waypoints = [
        place_coords,                 # Start from current position
        INTERMEDIATE_STEP_CARTESIAN,  # Move to intermediate position 
    ]

    # Use similar velocities and blend radii for consistent motion
    return_velocities = [0.25, 0.25]
    return_blend_radii = [0.1, 0.1]

    # Move smoothly back through intermediate position
    move_through_waypoints(return_waypoints, velocities=return_velocities, blend_radii=return_blend_radii)

    # Then use a higher velocity for the final joint movement for efficiency
    angles_rad = [math.radians(angle) for angle in STANDBY_POSITION_JOINTS]
    command = f"movej({angles_rad}, a=1.4, v=0.6, r=0.05)"
    send_command(command)
    time.sleep(2)  # Allow time for completion

def move_home():
    print("[ROBOT] Moving to home position")
    home_angles = [0, -1.57, 0, -1.57, 0, 0]
    command = f"movej({home_angles}, a=1.2, v=0.5)"
    send_command(command)
    #print("[WAIT] Waiting for home positioning to complete (5s)")
    time.sleep(5)

def move_through_waypoints(waypoints, velocities=None, blend_radii=None):
    """
    Move through a sequence of waypoints in one continuous motion
    
    Args:
        waypoints: List of position lists/tuples (x, y, z, roll, pitch, yaw)
        velocities: List of velocities for each segment (defaults to 0.25)
        blend_radii: List of blend radii for each waypoint (defaults to 0.05)
    """
    if velocities is None:
        velocities = [0.25] * len(waypoints)
    if blend_radii is None:
        # Make last point have 0 blend radius for precision
        blend_radii = [0.1] * (len(waypoints) - 1) + [0]
    
    # Make sure all waypoints are properly formatted
    formatted_waypoints = []
    for wp in waypoints:
        if isinstance(wp, (list, tuple)):
            # Ensure it's a list for consistent handling
            wp_list = list(wp)
            
            # Make sure we have exactly 6 values (x,y,z,rx,ry,rz)
            if len(wp_list) < 6:
                wp_list.extend([0] * (6 - len(wp_list)))
            elif len(wp_list) > 6:
                wp_list = wp_list[:6]
                
            formatted_waypoints.append(wp_list)
    
    # Convert to meters if needed
    script = "def smooth_move():\n"
    for i, (wp, v, r) in enumerate(zip(formatted_waypoints, velocities, blend_radii)):
        x, y, z, roll, pitch, yaw = wp
        
        # Scale if in millimeters
        if abs(x) > 10 or abs(y) > 10 or abs(z) > 10:
            x, y, z = x / 1000, y / 1000, z / 1000
            
        script += f"  movel(p[{x}, {y}, {z}, {roll}, {pitch}, {yaw}], a=1.2, v={v}, r={r})\n"
    
    script += "end\n"
    #print(f"[ROBOT] Generated smooth movement script with {len(waypoints)} waypoints")
    send_command(script)
    
    # Wait proportional to the number of waypoints
    wait_time = 2 + len(waypoints)
    #print(f"[WAIT] Waiting for waypoint movement to complete ({wait_time}s)")
    time.sleep(wait_time)

#############################################
# Camera & Detection
#############################################

fx, fy = 1360.670, 1361.407
cx, cy = 993.9897, 551.1417

def compute_rotation_matrix(camera_points, world_points):
    cam_centroid = np.mean(camera_points, axis=0)
    world_centroid = np.mean(world_points, axis=0)
    H = np.dot((camera_points - cam_centroid).T, (world_points - world_centroid))
    U, _, Vt = np.linalg.svd(H)
    R_mat = np.dot(Vt.T, U.T)
    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        R_mat = np.dot(Vt.T, U.T)
    return R_mat

def transform_point(cam_point, R_mat, T_vec):
    return np.dot(R_mat, cam_point) + T_vec

def calculate_xyz(x, y, depth):
    Z = depth
    X = (x - cx) * (Z / fx)
    Y = (y - cy) * (Z / fy)
    return X, Y, Z if Z > 0 else None

def save_image(image, save_dir, counter):
    filename = os.path.join(save_dir, f"capture_{counter}.png")
    cv2.imwrite(filename, image)

# Global Variables for Threading
robot_busy = False
img_counter = 1
current_nir = None


# Robot Command Thread Function

def process_robot_command(world_coords, detection_image, conn, save_dir):
    global img_counter, robot_busy, current_nir
    #print(f"[IMAGE] Saving detection image #{img_counter}")
    save_image(detection_image, save_dir, img_counter)
    img_counter += 1

    # Ensure we have a valid label before starting
    if current_nir is None or current_nir not in SORTING_POSITIONS:
        print(f"[ERROR] Invalid NIR label: {current_nir}")
        robot_busy = False
        return

    # Get placement coordinates early to verify
    place_coords = get_placement_position(current_nir)
    textile_class = get_textile_class(current_nir)
    #print(f"[TASK] Starting sorting of {textile_class} to bin #{current_nir}")

    pickup_coords = list(world_coords)
    pickup_coords[1] += 0.45
    
    # Single move to pickup position
    print("[ROBOT] Moving to pickup position")
    move_to_position(*pickup_coords, blend_radius=0)  # Precise positioning for pickup
    pick_textile()
    #print("[WAIT] Ensuring secure grip (0.2s)")
    time.sleep(0.20)  # Extra time to ensure grip is secure
    
    # Define waypoints for smooth motion through intermediate position to placement
    to_place_waypoints = [
        list(INTERMEDIATE_STEP_CARTESIAN),  # First go to intermediate position
        list(place_coords)                  # Then to placement position
    ]
    
    #print(f"[TASK] Moving through intermediate to bin #{current_nir}")
    # Increase blend radius for smoother motion to placement
    move_through_waypoints(to_place_waypoints, 
                          velocities=[0.25, 0.25], 
                          blend_radii=[0.10, 0])  # Increased from 0.05 to 0.10
    
    # Calculate additional wait time based on bin distance
    # Bins 0 and 3 are closest, 1 and 4 are middle, 2 and 5 are farthest
    additional_wait = 0
    if current_nir in [1, 4]:  # Middle distance bins
        additional_wait = 1.0
    elif current_nir in [2, 5]:  # Farthest bins
        additional_wait = 1.5
        
    # Wait to ensure robot has reached the position
    #print(f"[WAIT] Additional wait for bin #{current_nir}: {additional_wait}s")
    time.sleep(1.0 + additional_wait)  # Base wait + additional for distance
    
    release_textile()
    #print("[WAIT] Ensuring textile is released (0.1s)")
    time.sleep(0.1)  # Give time for release
    
    print("[TASK] Returning to standby position")
    # Define return waypoints for smooth motion back to intermediate position
    to_intermediate_waypoints = [
        list(place_coords),                 # Start from current position
        list(INTERMEDIATE_STEP_CARTESIAN)   # Move to intermediate position 
    ]
    
    # Use a script that combines all movements for seamless return motion
    script = "def smooth_return():\n"
    
    # First waypoint (current position) - use the same place_coords we used for placement
    x, y, z = place_coords[0]/1000, place_coords[1]/1000, place_coords[2]/1000
    roll, pitch, yaw = place_coords[3], place_coords[4], place_coords[5]
    script += f"  movel(p[{x}, {y}, {z}, {roll}, {pitch}, {yaw}], a=1.2, v=0.25, r=0.15)\n"
    
    # Second waypoint (intermediate position)
    x, y, z = INTERMEDIATE_STEP_CARTESIAN[0]/1000, INTERMEDIATE_STEP_CARTESIAN[1]/1000, INTERMEDIATE_STEP_CARTESIAN[2]/1000
    roll, pitch, yaw = INTERMEDIATE_STEP_CARTESIAN[3], INTERMEDIATE_STEP_CARTESIAN[4], INTERMEDIATE_STEP_CARTESIAN[5]
    script += f"  movel(p[{x}, {y}, {z}, {roll}, {pitch}, {yaw}], a=1.2, v=0.25, r=0.15)\n"
    
    # Finally, move directly to joint position with blending
    angles_rad = [math.radians(angle) for angle in STANDBY_POSITION_JOINTS]
    script += f"  movej({angles_rad}, a=1.4, v=0.6, r=0.05)\n"
    
    script += "end\n"
    #print(f"[ROBOT] Starting return journey: bin → intermediate → standby")
    send_command(script)
    #print(f"[WAIT] Waiting for return journey to complete (8s)")
    time.sleep(8)  # Allow time for the full return motion
    
    print("[SYSTEM] Sorting complete, robot ready")
    print("[SYSTEM] Waiting for next textile to be scanned...")
    new_label = wait_for_label(conn)
    if new_label is not None:
        current_nir = new_label
    print(f"[SYSTEM] Next textile type received: {get_textile_class(current_nir)} (bin #{current_nir})")
    robot_busy = False

# Main Process

def main():
    global current_nir, robot_busy
    print("[SYSTEM] ===== TEXTILE SORTING SYSTEM STARTUP =====")
    print("[SYSTEM] Initializing camera...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    pipeline.start(config)

    print("[SYSTEM] Initializing robot...")
    move_home()
    print("[SYSTEM] Moving to standby position...")
    move_joints(STANDBY_POSITION_JOINTS)
    #print("[SYSTEM] Establishing connection with NIR scanner...")
    print("[SYSTEM] Robot ready and waiting for connection...")
    conn = init_nir_connection()
    current_nir = wait_for_label(conn)
    print(f"[SYSTEM] First textile type: {get_textile_class(current_nir)} (bin #{current_nir})")

    save_dir = "./Taken_pictures"
    os.makedirs(save_dir, exist_ok=True)
    model = YOLO("/home/devtex/MAS500/UR5_ws/src/textile_sorting/textile_sorting/Color_best.pt")
    # Use the YOLO model's own label names (which are "black", "white", "other")
    yolo_labels = model.names

    x_scale = 640 / 1920
    y_scale = 480 / 1080

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            if not depth or not color:
                continue

            color_image = np.asanyarray(color.get_data())
            results = model(color_image, verbose=False)

            for result in results:
                for bbox, conf, label in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    if conf < 0.9:
                        continue

                    # Convert bounding box coordinates to integers
                    x_min, y_min, x_max, y_max = map(int, bbox)
                    
                    # Draw bounding box and label from YOLO on the image
                    class_text = yolo_labels[int(label)]
                    cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(color_image, f"{class_text} {conf:.2f}", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Compute center of the bounding box and scale to depth image dimensions
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    xd = int(x_center * x_scale)
                    yd = int(y_center * y_scale)

                    if not (0 <= xd < 640 and 0 <= yd < 480):
                        continue

                    depth_val = depth.get_distance(xd, yd)
                    if depth_val > 0:
                        cam_coords = calculate_xyz(x_center, y_center, depth_val)
                        if cam_coords:
                            # Compute world coordinates (using a fixed calibration here)
                            camera_points = np.array([[0.324, 0.016, 0.637],
                                                      [-0.375, -0.018, 0.636],
                                                      [-0.006, -0.115, 0.622],
                                                      [-0.116, 0.115, 0.643]])
                            world_points = np.array([[0.475, -0.201, -0.055],
                                                     [0.455, -0.860, -0.055],
                                                     [0.341, -0.520, -0.055],
                                                     [0.600, -0.630, -0.055]])
                            R_matrix = compute_rotation_matrix(camera_points, world_points)
                            T_vector = np.array([0.385, -0.520, 0.564])
                            world_coords = transform_point(np.array(cam_coords), R_matrix, T_vector)
                            world_coords[2] = -0.040

                            # If the detected object's bounding box crosses the vertical center line
                            if x_min < cx and x_max > cx:
                                if not robot_busy:
                                    robot_busy = True
                                    # Start a new thread for the robot control operations
                                    threading.Thread(target=process_robot_command, 
                                                     args=(world_coords, color_image.copy(), conn, save_dir)).start()
                                # Break after processing one detection to avoid multiple triggers
                                break

            # Draw a vertical center line for reference
            cv2.line(color_image, (int(cx), 0), (int(cx), 1080), (0, 0, 255), 2)
            cv2.imshow("Live Detection", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
