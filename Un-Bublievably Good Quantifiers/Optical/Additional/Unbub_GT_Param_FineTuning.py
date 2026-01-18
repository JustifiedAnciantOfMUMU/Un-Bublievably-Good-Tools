import cv2, numpy as np
import time, matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import linear_sum_assignment


thresh = 20
area_thresh = 8
y_cutoff = 150
bubble_list = []
tracking_bubble_list = []

class bubble:
    """Class to represent an identified Bubble"""
    pixel_size = ((661 - 29) / 31) * 100  # in pixels per cm

    def __init__(self, _id, centeroid = [0, 0], timestamp = 0):
        self.id = _id
        self.current_centroid = centeroid
        self.centroid_position = []
        self.current_volume = 0
        self.volume_history = []
        self.radii_history = []
        self.x_coords = []
        self.y_coords = []
        self.tracked = False
        self.timestamp = timestamp

    def set_current_centeroid(self, centeroid):
        """Set the current centroid position of the bubble."""
        self.centroid_position.append(self.current_centroid)
        self.current_centroid = centeroid

    def get_distances_to_nodes(self, current_conteroids):
        distances = []
        for c in current_conteroids:
            if self.current_centroid[1] < (c[1] + 5):
                dist = 100000                                                                # if the new position is below the current position + a threshold, assign a large distance
            else:
                dist = np.sqrt((self.current_centroid[0] - c[0])**2 + (self.current_centroid[1] - c[1])**2)  
                if dist > 40:
                    dist = 100000         # else Calculate the distance between the current centroid and the node centroid
            distances.append(dist)
        return distances

    def calculate_and_store_volume(self, image, labels, componant_label = 0):
        #get contours using current position - convex hull then calculate volume
        
        bubble_mask = np.zeros_like(image, dtype=np.uint8)
        bubble_mask[labels == componant_label] = 255
        #cv2.imshow('Component Mask', bubble_mask)

        contours, _ = cv2.findContours(bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)

        # Plot the contour line on the bubble mask
        # bubble_mask_with_contour = cv2.cvtColor(bubble_mask, cv2.COLOR_GRAY2BGR)  # Convert to BGR for visualization
        # cv2.drawContours(bubble_mask_with_contour, [contour], -1, (0, 255, 0), 2)  # Draw the contour in green
        # resized_bubble_mask = cv2.resize(bubble_mask_with_contour, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)  # Resize the image to 4x
        # cv2.imshow('Bubble Mask with Contour', resized_bubble_mask)

        points = np.squeeze(contour)
        if len(points) < 3:
            return 0  # Not enough points to form a convex hull
        
        # Compute the convex hull of the points
        dx = self.pixel_size
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        x = self.MVEE(hull_points)
        self.current_volume = x[3]   #volume in cm^3
        self.volume_history.append(x[3])
        self.radii_history.append(x[4])  #radii in cm

        # surface_area = hull.volume * (dx ** 2)            # Calculate sureace area in cm^2
        # r = np.sqrt(surface_area / (4 * np.pi))         # calculate the radius of circle with surface area of convex hull
        # V = (4/3) * np.pi * (r ** 3)        # Calculate the volume of a sphere using the equivalent radiu
        # self.current_volume = V
        # self.volume_history.append(V)
        # self.radii_history.append(r)
        return hull_points
    

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



def get_frame_series(vidObj, start_point, number_of_frames):
    """create array of frame series"""
    frame_series = []
    vidObj.set(cv2.CAP_PROP_POS_FRAMES, start_point)
    for i in range(int(number_of_frames)):
        ret, frame = vidObj.read()
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not ret:
            break
        frame_series.append(grey_frame)

    return frame_series

def on_background_threshold_change(val):
    global thresh
    thresh = val
    
def on_cluster_area_threshold_change(val):
    global area_thresh
    area_thresh = val


def loop_through_video(video_path):
    """
    Loop through a video file, displaying each frame with adjustable delay and playback speed.
    
    Args:
        video_path (str): Path to the video file
    """
    cap = cv2.VideoCapture(video_path)

    
    background_frame_width = 20
    image_crop = (750, 300, 250, 450)  # (x, y, w, h)
    search_box = (50, 365, 75, 40)  # (x, y, w, h)


    end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    bubble_count = 0
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    
    
    cv2.namedWindow('Video Frame')
    cv2.createTrackbar('background Threshold', 'Video Frame', 14, 50, on_background_threshold_change)
    cv2.createTrackbar('cluster area threshold', 'Video Frame',5, 20, on_cluster_area_threshold_change)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        
        k = cap.get(cv2.CAP_PROP_POS_FRAMES)

         ############### Generate background frame ##############
        result1 = k - (background_frame_width / 2)       #gets rolling average background frames
        result2 = k + 1 + (background_frame_width / 2)
        before, after = [], []
        if result1 < 0 :  # if the frame is before the start of the video
            before = get_frame_series(cap, k - 1 - (background_frame_width / 2) - result1, (background_frame_width / 2) - result1)
            after = get_frame_series(cap, k + 1, (background_frame_width / 2) - result1)
        elif  result2 > end_frame: 
            remainder = end_frame - (k + 1)
            buffer_size = background_frame_width / 2
            before = get_frame_series(cap, k - 1 - (buffer_size + (buffer_size - remainder)), background_frame_width - remainder)
            if remainder > 0:
                after = get_frame_series(cap, k + 1, remainder)
        else:
            before = get_frame_series(cap, k - 1 - (background_frame_width / 2), (background_frame_width / 2))
            after = get_frame_series(cap, k + 1, (background_frame_width / 2))


        median_background = np.median(np.array(before + after), axis=0).astype(np.uint8)[image_crop[1]:(image_crop[1]+image_crop[3]), image_crop[0]:(image_crop[0]+image_crop[2])]

        cropped_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[image_crop[1]:(image_crop[1]+image_crop[3]), image_crop[0]:(image_crop[0]+image_crop[2])]
        subtracted_background = ((abs(cropped_gray_frame.astype(np.int8)) - abs(median_background.astype(np.int8))) > thresh).astype(np.uint8)

        #Filling in small gaps between indentified componants
        kernel = np.ones((3, 3), np.uint8)  # 7x7 kernel to fill 3-pixel gaps  - could be changed to 5x5 or 3x3
        processed_frame = cv2.morphologyEx(subtracted_background, cv2.MORPH_CLOSE, kernel) # Apply morphological closing (dilation followed by erosion)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_frame) # Find connected components in the binary image

        for n in range(num_labels - 1, 0, -1):
            if stats[n, cv2.CC_STAT_AREA] < area_thresh:     # Remove small components based on area threshold
                processed_frame[labels == n] = 0

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_frame)
        _hull_points = []


        ####### if there are bubbles being tracked already ##########
        if len(tracking_bubble_list) != 0:                                                                                                                  # !!!! track existing bubbles !!!!                                         
            distances = []
            for _bubble in tracking_bubble_list:                                                                                                            # For each bubble in the tracking list, calculate the distance to the current centroids                                         
                distances.append(bubble.get_distances_to_nodes(_bubble, centroids[1:]))
            row_ind, col_ind = linear_sum_assignment(distances)                                                                                             # Use the Hungarian algorithm to find the optimal assignment of bubbles to centroids
            for i in range(len(row_ind)):                                                                                                                   # Iterate through the assigned bubbles    
                if col_ind[i] != 100000:                                                                                                                    # If the distance is not too large, update the bubble's centroid and calculate its volume  
                    tracking_bubble_list[i].set_current_centeroid(centroids[col_ind[i] + 1]) #volume
                    _hull_points.append(tracking_bubble_list[i].calculate_and_store_volume(processed_frame, labels, componant_label = col_ind[i] + 1))      # Calculate the volume of the bubble using the convex hull of the current component     
                    tracking_bubble_list[i].tracked = True
                else:
                    pass   
                    
            for i in range(len(tracking_bubble_list) - 1, -1, -1):  # Iterate in reverse to avoid indexing issues
                if tracking_bubble_list[i].tracked == False:
                    tracking_bubble_list.pop(i)                                                                                               # changes the tracked status of the bubble to True


            for n in range(1, num_labels): # identify new bubbles                                                                                           # !!!! track new bubbles !!!!
                _x= centroids[n][0] 
                _y= centroids[n][1]
                if search_box[0] <= _x <= (search_box[0] + search_box[2]) and search_box[1] <= _y <= (search_box[1] + search_box[3]):                                                       # Check if the centroid is within the search box bounds   
                    if (n-1) not in col_ind:                                                                                                                # If the centroid is not already assigned to a bubble, create a new bubble object 
                        tracking_bubble_list.append(bubble(bubble_count, centroids[n], k/frame_rate))
                        bubble_count += 1



        ####### if there are no bubbles being tracked ##########
        else:
            for n in range(1, num_labels):                                                                                                                                                                      # If there are no bubbles in the tracking list, create new bubble objects for all centroids within the search box bounds        
                if search_box[0] <= centroids[n][0] <= (search_box[0] + search_box[2]) and search_box[1] <= centroids[n][1] <= (search_box[1] + search_box[3]):
                    tracking_bubble_list.append(bubble(bubble_count, centroids[n], k/frame_rate))                                                                                                               # Create a new bubble object with timestamp of current frame number
                    bubble_count += 1





        ####### Remove bubbles that have crossed the y_cutoff ##########
        if len(tracking_bubble_list) != 0:                                                      # Cycles through tracked bubbles removes if centeroid position is above yaxis cuttoff. When removed bubble is added to main bubble list and removed from tracking list
            for i in range(len(tracking_bubble_list) - 1, -1, -1):    
                if tracking_bubble_list[i].current_centroid[1] < y_cutoff:
                    tracking_bubble_list[i].set_current_centeroid([0, 0])
                    bubble_list.append(tracking_bubble_list[i])
                    tracking_bubble_list.pop(i)



        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # Plot cropped gray frame with contours
        axs[0].imshow(cropped_gray_frame, cmap='gray')
        axs[0].set_title('Gray Frame with Contours')
        axs[0].axis('off')
        axs[0].add_patch(plt.Rectangle((search_box[0], search_box[1]), search_box[2], search_box[3], edgecolor='white', facecolor='none', linewidth=1))
        axs[0].axhline(y=y_cutoff, color='red', linestyle='--', linewidth=1)
        # Plot convex hulls for each bubble
        for hull in _hull_points:
            if isinstance(hull, np.ndarray):
                if hull is not None and len(hull) > 2:
                    hull = np.vstack([hull, hull[0]])
                    axs[0].plot(hull[:, 0], hull[:, 1], color='yellow', linewidth=1)
        # Plot processed frame
        axs[1].imshow(processed_frame, cmap='gray')
        axs[1].set_title(f'Processed Frame {k}')
        axs[1].axis('off')
        axs[1].add_patch(plt.Rectangle((search_box[0], search_box[1]), search_box[2], search_box[3], edgecolor='white', facecolor='none', linewidth=1))
        axs[1].axhline(y=y_cutoff, color='red', linestyle='--', linewidth=1)
        for _bubble in tracking_bubble_list:
            axs[0].plot(_bubble.current_centroid[0], _bubble.current_centroid[1], 'bo', markersize=2)
            axs[0].annotate(f'ID: {_bubble.id}', (_bubble.current_centroid[0] + 5, _bubble.current_centroid[1] + 5), color='white', fontsize=8)
            axs[1].plot(_bubble.current_centroid[0], _bubble.current_centroid[1], 'bo', markersize=2)
            axs[1].annotate(f'ID: {_bubble.id}', (_bubble.current_centroid[0] + 5, _bubble.current_centroid[1] + 5), color='white', fontsize=8)
        
        plt.show()
        time.sleep(1)
        plt.close(fig)
        
         ############### Frame display and control ##############
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    video_path = r"C:\Users\jkf1g22\OneDrive - University of Southampton\Documents\_PostGraduateResearch\_Tools\Unbub-lievably_Good_Tracker\test_video.mp4"
    loop_through_video(video_path)