import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import random
import json
import numpy as np


class VideoFrameNavigator:

    pixel_size = ((661 - 29) / 31) * 100

    def __init__(self, root):
        self.root = root
        self.root.title("Video Frame Navigator")
        self.root.geometry("1000x700")
        
        self.video_path = None
        self.cap = None
        self.current_frame_number = 0
        self.total_frames = 0
        
        # Crop variables
        self.crop_mode = False
        self.crop_rect = None  # (x1, y1, x2, y2) in original frame coordinates
        self.crop_start = None
        self.temp_rect_id = None
        
        # Point marking variables
        self.marked_points = []  # List of (frame_number, x, y, color) in original frame coordinates
        self.point_ids = []  # Canvas IDs for drawn points
        self.dot_color = "red"  # Color of marked points
        self.bubble_index = 0  # Index for cycling through bubble colors
        
        # Create UI elements
        self.create_widgets()
        
        # Ask for video file on startup
        self.root.after(100, self.select_video_file)
    
    def create_widgets(self):
        # Top frame for file selection
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        self.file_label = tk.Label(top_frame, text="No video loaded", fg="gray")
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        select_btn = tk.Button(top_frame, text="Select Video", command=self.select_video_file)
        select_btn.pack(side=tk.RIGHT, padx=5)
        
        self.crop_btn = tk.Button(top_frame, text="Set Crop Area", command=self.toggle_crop_mode)
        self.crop_btn.pack(side=tk.RIGHT, padx=5)
        
        self.reset_crop_btn = tk.Button(top_frame, text="Reset Crop", command=self.reset_crop)
        self.reset_crop_btn.pack(side=tk.RIGHT, padx=5)

        self.undo_point_btn = tk.Button(top_frame, text="Undo Point", command=self.undo_last_point)
        self.undo_point_btn.pack(side=tk.RIGHT, padx=5)
        
        self.next_bubble_btn = tk.Button(top_frame, text="Next Bubble", command=self.next_bubble)
        self.next_bubble_btn.pack(side=tk.RIGHT, padx=5)
        self.save_points_btn = tk.Button(top_frame, text="Save Points", command=self.save_points)
        self.save_points_btn.pack(side=tk.RIGHT, padx=5)
        
        self.load_points_btn = tk.Button(top_frame, text="Load Points", command=self.load_points)
        self.load_points_btn.pack(side=tk.RIGHT, padx=5)
        self.export_bubbles_btn = tk.Button(top_frame, text="Export Bubbles", command=self.export_bubbles)
        self.export_bubbles_btn.pack(side=tk.RIGHT, padx=5)
        
            # Canvas for video display
        self.canvas = tk.Canvas(self.root, bg="black")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Bind mouse events for crop selection
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # Frame info label
        self.info_label = tk.Label(self.root, text="Frame: 0 / 0", font=("Arial", 10))
        self.info_label.pack(side=tk.TOP, pady=5)
        
        # Control frame
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.BOTTOM, pady=10)
        
        # Navigation buttons
        btn_backward_10 = tk.Button(control_frame, text="◀◀ -10", command=lambda: self.step_frames(-10), 
                                     width=10, height=2)
        btn_backward_10.grid(row=0, column=0, padx=5)
        
        btn_backward_1 = tk.Button(control_frame, text="◀ -1", command=lambda: self.step_frames(-1), 
                                    width=10, height=2)
        btn_backward_1.grid(row=0, column=1, padx=5)
        
        btn_forward_1 = tk.Button(control_frame, text="+1 ▶", command=lambda: self.step_frames(1), 
                                   width=10, height=2)
        btn_forward_1.grid(row=0, column=2, padx=5)
        
        btn_forward_10 = tk.Button(control_frame, text="+10 ▶▶", command=lambda: self.step_frames(10), 
                                    width=10, height=2)
        btn_forward_10.grid(row=0, column=3, padx=5)
        
        # Keyboard bindings
        self.root.bind('<Left>', lambda e: self.step_frames(-1))
        self.root.bind('<Right>', lambda e: self.step_frames(1))
        self.root.bind('<Shift-Left>', lambda e: self.step_frames(-10))
        self.root.bind('<Shift-Right>', lambda e: self.step_frames(10))
    
    def select_video_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("MP4 files", "*.mp4"), ("All video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        
        if file_path:
            self.load_video(file_path)
    
    def load_video(self, video_path):
        # Release previous video if exists
        if self.cap is not None:
            self.cap.release()
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open video file")
            return
        
        # Get total frames
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_number = 0
        
        # Update file label
        filename = os.path.basename(video_path)
        self.file_label.config(text=f"Video: {filename}", fg="black")
        
        # Display first frame
        self.display_frame()
    
    def step_frames(self, step):

        self.change_dot_color()
        self.bubble_index += 1
        
        if self.cap is None:
            messagebox.showwarning("No Video", "Please select a video file first")
            return
        
        # Calculate new frame number
        new_frame_number = self.current_frame_number + step
        
        # Clamp to valid range
        new_frame_number = max(0, min(new_frame_number, self.total_frames - 1))
        
        # Set the frame position
        self.current_frame_number = new_frame_number
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
        
        # Display the frame
        self.display_frame()
    
    def display_frame(self ,new_frame_number=None):
        if self.cap is None:
            return
        
        ret, frame = self.cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply crop if set
            if self.crop_rect is not None:
                x1, y1, x2, y2 = self.crop_rect
                frame_rgb = frame_rgb[y1:y2, x1:x2]
            
            # Get canvas dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Resize frame to fit canvas while maintaining aspect ratio
            frame_height, frame_width = frame_rgb.shape[:2]
            
            if canvas_width > 1 and canvas_height > 1:  # Ensure canvas is properly sized
                scale = min(canvas_width / frame_width, canvas_height / frame_height)
                new_width = int(frame_width * scale)
                new_height = int(frame_height * scale)
                
                frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
            else:
                frame_resized = frame_rgb
            
            # Convert to PhotoImage
            img = Image.fromarray(frame_resized)
            self.photo = ImageTk.PhotoImage(image=img)
            
            # Clear canvas and display new image
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width // 2, canvas_height // 2, 
                                    image=self.photo, anchor=tk.CENTER)
            
            # Draw marked points for current frame
            self.draw_marked_points()
            
            # Update info label
            crop_text = " [CROPPED]" if self.crop_rect is not None else ""
            points_on_frame = len([p for p in self.marked_points if p[0] == self.current_frame_number])
            point_text = f" | Points: {points_on_frame}/{len(self.marked_points)}" if self.marked_points else ""
            self.info_label.config(text=f"Frame: {self.current_frame_number + 1} / {self.total_frames}{crop_text}{point_text}")
        else:
            messagebox.showwarning("Error", "Failed to read frame")
    
    def toggle_crop_mode(self):
        if self.cap is None:
            messagebox.showwarning("No Video", "Please select a video file first")
            return
        
        self.crop_mode = not self.crop_mode
        if self.crop_mode:
            self.crop_btn.config(relief=tk.SUNKEN, bg="lightblue")
            self.info_label.config(text=f"Frame: {self.current_frame_number + 1} / {self.total_frames} - CROP MODE: Click and drag to select area")
        else:
            self.crop_btn.config(relief=tk.RAISED, bg="SystemButtonFace")
            self.display_frame()
    
    def on_canvas_click(self, event):
        if self.crop_mode:
            # Crop mode: start drawing rectangle
            self.crop_start = (event.x, event.y)
            if self.temp_rect_id:
                self.canvas.delete(self.temp_rect_id)
        else:
            # Point marking mode: add point at clicked location
            self.add_point_at_canvas_position(event.x, event.y)
    
    def on_canvas_drag(self, event):
        if not self.crop_mode or self.crop_start is None:
            return
        
        if self.temp_rect_id:
            self.canvas.delete(self.temp_rect_id)
        
        self.temp_rect_id = self.canvas.create_rectangle(
            self.crop_start[0], self.crop_start[1],
            event.x, event.y,
            outline="red", width=2
        )
    
    def on_canvas_release(self, event):
        if not self.crop_mode or self.crop_start is None:
            return
        
        # Get canvas and frame dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Get original frame dimensions
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
        ret, frame = self.cap.read()
        if not ret:
            return
        
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate scale and offset
        scale = min(canvas_width / frame_width, canvas_height / frame_height)
        display_width = int(frame_width * scale)
        display_height = int(frame_height * scale)
        offset_x = (canvas_width - display_width) // 2
        offset_y = (canvas_height - display_height) // 2
        
        # Convert canvas coordinates to frame coordinates
        x1_canvas = min(self.crop_start[0], event.x)
        y1_canvas = min(self.crop_start[1], event.y)
        x2_canvas = max(self.crop_start[0], event.x)
        y2_canvas = max(self.crop_start[1], event.y)
        
        # Adjust for offset and scale
        x1_frame = int((x1_canvas - offset_x) / scale)
        y1_frame = int((y1_canvas - offset_y) / scale)
        x2_frame = int((x2_canvas - offset_x) / scale)
        y2_frame = int((y2_canvas - offset_y) / scale)
        
        # Clamp to frame boundaries
        x1_frame = max(0, min(x1_frame, frame_width))
        y1_frame = max(0, min(y1_frame, frame_height))
        x2_frame = max(0, min(x2_frame, frame_width))
        y2_frame = max(0, min(y2_frame, frame_height))
        
        # Only set crop if area is valid
        if x2_frame > x1_frame and y2_frame > y1_frame:
            self.crop_rect = (x1_frame, y1_frame, x2_frame, y2_frame)
        
        # Exit crop mode and refresh display
        self.crop_mode = False
        self.crop_btn.config(relief=tk.RAISED, bg="SystemButtonFace")
        self.crop_start = None
        if self.temp_rect_id:
            self.canvas.delete(self.temp_rect_id)
            self.temp_rect_id = None
        
        self.display_frame()
    
    def reset_crop(self):
        self.crop_rect = None
        self.crop_mode = False
        self.crop_btn.config(relief=tk.RAISED, bg="SystemButtonFace")
        self.display_frame()
    
    def add_point_at_canvas_position(self, canvas_x, canvas_y, color="red"):
        if self.cap is None:
            return
        
        # Get canvas and current frame dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Get original frame dimensions
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
        ret, frame = self.cap.read()
        if not ret:
            return
        
        original_height, original_width = frame.shape[:2]
        
        # Account for crop if active
        if self.crop_rect is not None:
            crop_x1, crop_y1, crop_x2, crop_y2 = self.crop_rect
            display_width = crop_x2 - crop_x1
            display_height = crop_y2 - crop_y1
        else:
            display_width = original_width
            display_height = original_height
            crop_x1, crop_y1 = 0, 0
        
        # Calculate scale and offset
        scale = min(canvas_width / display_width, canvas_height / display_height)
        scaled_width = int(display_width * scale)
        scaled_height = int(display_height * scale)
        offset_x = (canvas_width - scaled_width) // 2
        offset_y = (canvas_height - scaled_height) // 2
        
        # Convert canvas coordinates to frame coordinates
        frame_x = int((canvas_x - offset_x) / scale) + crop_x1
        frame_y = int((canvas_y - offset_y) / scale) + crop_y1
        
        # Clamp to original frame boundaries
        frame_x = max(0, min(frame_x, original_width - 1))
        frame_y = max(0, min(frame_y, original_height - 1))
        
        # Store point with frame number and current color
        self.marked_points.append((self.current_frame_number, frame_x, frame_y, self.dot_color, self.bubble_index))
        #print(f"Point added: Frame {self.current_frame_number + 1}, Position ({frame_x}, {frame_y}), Color: {self.dot_color}")
        
        # Redraw to show new point
        self.draw_marked_points()
    
    def draw_marked_points(self):
        # Clear existing point drawings
        for point_id in self.point_ids:
            self.canvas.delete(point_id)
        self.point_ids.clear()
        
        if not self.marked_points:
            return
        
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Get current frame dimensions
        if self.cap is None:
            return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
        ret, frame = self.cap.read()
        if not ret:
            return
        
        original_height, original_width = frame.shape[:2]
        
        # Account for crop if active
        if self.crop_rect is not None:
            crop_x1, crop_y1, crop_x2, crop_y2 = self.crop_rect
            display_width = crop_x2 - crop_x1
            display_height = crop_y2 - crop_y1
        else:
            display_width = original_width
            display_height = original_height
            crop_x1, crop_y1 = 0, 0
        
        # Calculate scale and offset
        scale = min(canvas_width / display_width, canvas_height / display_height)
        scaled_width = int(display_width * scale)
        scaled_height = int(display_height * scale)
        offset_x = (canvas_width - scaled_width) // 2
        offset_y = (canvas_height - scaled_height) // 2
        
        # Draw points for current frame
        for frame_num, frame_x, frame_y, point_color, bubble_index in self.marked_points:
            if frame_num == self.current_frame_number:
                # Adjust for crop
                adj_x = frame_x - crop_x1
                adj_y = frame_y - crop_y1
                
                # Only draw if point is within cropped region
                if 0 <= adj_x < display_width and 0 <= adj_y < display_height:
                    # Convert to canvas coordinates
                    canvas_x = int(adj_x * scale) + offset_x
                    canvas_y = int(adj_y * scale) + offset_y
                    
                    # Draw point as circle with its stored color
                    radius = 3
                    point_id = self.canvas.create_oval(
                        canvas_x - radius, canvas_y - radius,
                        canvas_x + radius, canvas_y + radius,
                        fill=point_color
                    )
                    self.point_ids.append(point_id)

        # Draw MVEE for completed bubbles on current frame
        for i in range(self.bubble_index + 1):
            points_in_bubble = [(x, y) for (f, x, y, c, b_idx) in self.marked_points if b_idx == i and f == self.current_frame_number]
            if len(points_in_bubble) < 3:
                continue
            
            center, radii, angle, ellipse_volume_in_cm, sphere_radius_in_cm = self.MVEE(points_in_bubble)
            
            # Draw ellipse on current frame
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
            ret, frame = self.cap.read()
            if ret:
                riginal_height, original_width = frame.shape[:2]
            
            # Account for crop if active
            if self.crop_rect is not None:
                crop_x1, crop_y1, crop_x2, crop_y2 = self.crop_rect
                display_width = crop_x2 - crop_x1
                display_height = crop_y2 - crop_y1
            else:
                display_width = original_width
                display_height = original_height
                crop_x1, crop_y1 = 0, 0
            
            # Calculate scale and offset
            scale = min(canvas_width / display_width, canvas_height / display_height)
            scaled_width = int(display_width * scale)
            scaled_height = int(display_height * scale)
            offset_x = (canvas_width - scaled_width) // 2
            offset_y = (canvas_height - scaled_height) // 2
            
            # Convert center to canvas coordinates
            adj_center_x = center[0] - crop_x1
            adj_center_y = center[1] - crop_y1
            canvas_center_x = int(adj_center_x * scale) + offset_x
            canvas_center_y = int(adj_center_y * scale) + offset_y
            
            # Draw ellipse on canvas
            a_scaled = radii[0] * scale
            b_scaled = radii[1] * scale
            angle_deg = np.degrees(angle)
            
            ellipse_id = self.canvas.create_oval(
            canvas_center_x - a_scaled, canvas_center_y - b_scaled,
            canvas_center_x + a_scaled, canvas_center_y + b_scaled,
            outline="white", width=2
            )
            self.point_ids.append(ellipse_id)
    

    
    def undo_last_point(self):
        # Find the last point on the current frame
        for i in range(len(self.marked_points) - 1, -1, -1):
            if self.marked_points[i][0] == self.current_frame_number:
                removed_point = self.marked_points.pop(i)
                frame_num, x, y, color, bubble_index = removed_point
                self.dot_color = color  # Restore dot color to the undone point's color
                self.bubble_index = bubble_index  # Restore bubble index to the undone point's bubble index
                print(f"Undone: Point at Frame {frame_num + 1}, Position ({x}, {y}), Color: {color}")
                
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()

                # Clear canvas and display new image
                self.canvas.delete("all")
                self.canvas.create_image(canvas_width // 2, canvas_height // 2, 
                                        image=self.photo, anchor=tk.CENTER)

                # Draw marked points for current frame
                self.draw_marked_points()

                return
        print(f"No points to undo on current frame {self.current_frame_number + 1}")
    

    def next_bubble(self):
        self.change_dot_color()
        self.bubble_index += 1


    def change_dot_color(self):
        """Change dot color to a randomly selected different color"""
        colors = ["red", "blue", "green", "yellow", "orange", "purple", "cyan", "magenta", 
                  "lime", "pink", "brown", "navy", "teal", "gold", "coral"]
        
        # Remove current color from choices
        available_colors = [c for c in colors if c != self.dot_color]
        
        # Select random color from remaining options
        self.dot_color = random.choice(available_colors)
        print(f"Dot color changed to: {self.dot_color} (will be used for next points)")
    
    def save_points(self):
        """Save marked points to a JSON file"""
        if not self.marked_points:
            messagebox.showwarning("No Points", "No points to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Points",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Convert list of tuples to list of lists for JSON serialization
                data = {
                    "video_path": self.video_path,
                    "points": [list(point) for point in self.marked_points]
                }
                
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"Saved {len(self.marked_points)} points to {file_path}")
                messagebox.showinfo("Success", f"Saved {len(self.marked_points)} points")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save points: {str(e)}")
                print(f"Error saving points: {e}")
    
    def load_points(self):
        """Load marked points from a JSON file"""
        file_path = filedialog.askopenfilename(
            title="Load Points",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Convert list of lists back to list of tuples
                self.marked_points = [tuple(point) for point in data["points"]]
                
                print(f"Loaded {len(self.marked_points)} points from {file_path}")
                if "video_path" in data:
                    print(f"Points were saved from video: {data['video_path']}")
                
                messagebox.showinfo("Success", f"Loaded {len(self.marked_points)} points")
                self.display_frame()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load points: {str(e)}")
                print(f"Error loading points: {e}")
    
    def export_bubbles(self):
        if not self.marked_points:
            messagebox.showwarning("No Points", "No points to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Bubbles",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        bubble_list = []

        for i in range(self.bubble_index + 1):
            points_in_bubble = [(x, y) for (f, x, y, c, b_idx) in self.marked_points if b_idx == i]
            if len(points_in_bubble) < 3:
                continue

            center, radii, angle, ellipse_volume_in_cm, sphere_radius_in_cm = self.MVEE(points_in_bubble)
            
            
    
    def on_closing(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


    def MVEE(self, points, tol=1e-4, max_iter=10_000, debug=False):
        P = np.asarray(points, dtype=float)
        n, d = P.shape
        assert d == 2

        Q = np.vstack([P.T, np.ones(n)])
        u = np.ones(n) / n

        for _ in range(max_iter):
            X = Q @ np.diag(u) @ Q.T
            X_inv = np.linalg.inv(X)
            M = np.einsum('ij,ji->i', Q.T @ X_inv, Q)

            j = np.argmax(M)
            max_M = M[j]

            if max_M - d - 1 <= tol:
                break

            step = (max_M - d - 1) / ((d + 1) * (max_M - 1))
            u = (1 - step) * u
            u[j] += step

        center = P.T @ u
        cov = (P.T @ np.diag(u) @ P) - np.outer(center, center)
        A = (1.0 / d) * np.linalg.inv(cov)

        eigenvals, eigenvecs = np.linalg.eigh(A)
        radii = 1.0 / np.sqrt(eigenvals)
        angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
        a, b = radii

        ellipse_volume_in_pix = (4/3) * np.pi * a**2 * b
        ellipse_volume_in_cm =  ellipse_volume_in_pix / (self.pixel_size**3)  # ppc pixels per cm
        sphere_radius_in_cm = ((4 * ellipse_volume_in_cm) / (3* np.pi)) ** (1/3)

        return center, radii, angle, ellipse_volume_in_cm, sphere_radius_in_cm



def main():
    root = tk.Tk()
    app = VideoFrameNavigator(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
