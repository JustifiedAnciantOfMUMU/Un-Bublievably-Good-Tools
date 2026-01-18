import sys, cv2, numpy as np, matplotlib.pyplot as plt, sys, os
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
import json, time
import argparse

def plot_frame(tracking_bubble_list, processed_frame, cropped_gray_frame, searchBox_bounds, y_cutoff, _hull_points, k):
    contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # Plot cropped gray frame with contours
    axs[0].imshow(cropped_gray_frame, cmap='gray')
    axs[0].set_title('Gray Frame with Contours')
    axs[0].axis('off')
    axs[0].add_patch(plt.Rectangle((searchBox_bounds[0], searchBox_bounds[1]), searchBox_bounds[2], searchBox_bounds[3], edgecolor='white', facecolor='none', linewidth=1))
    axs[0].axhline(y=y_cutoff, color='red', linestyle='--', linewidth=1)
    # Draw contours on the gray frame
    # for contour in contours:
    #     contour = contour.squeeze()
    #     if contour.ndim == 2:
    #         axs[0].plot(contour[:, 0], contour[:, 1], color='red', linewidth=1)
    # Plot convex hulls for each bubble
    for hull in _hull_points:
        if isinstance(hull, np.ndarray):  # Check if hull is an array
            if hull is not None and len(hull) > 2:
                hull = np.vstack([hull, hull[0]])  # Close the hull
                axs[0].plot(hull[:, 0], hull[:, 1], color='yellow', linewidth=1)
    # Plot processed frame
    axs[1].imshow(processed_frame, cmap='gray')
    axs[1].set_title(f'Processed Frame {k}')
    axs[1].axis('off')
    axs[1].add_patch(plt.Rectangle((searchBox_bounds[0], searchBox_bounds[1]), searchBox_bounds[2], searchBox_bounds[3], edgecolor='white', facecolor='none', linewidth=1))
    axs[1].axhline(y=y_cutoff, color='red', linestyle='--', linewidth=1)
    for _bubble in tracking_bubble_list:
        axs[0].plot(_bubble.current_centroid[0], _bubble.current_centroid[1], 'bo', markersize=2)  # Plot centroids as smaller blue circles
        axs[0].annotate(f'ID: {_bubble.id}', (_bubble.current_centroid[0] + 5, _bubble.current_centroid[1] + 5), color='white', fontsize=8)  # Add label with bubble ID using annotate
        axs[1].plot(_bubble.current_centroid[0], _bubble.current_centroid[1], 'bo', markersize=2)  # Plot centroids as blue circles
        axs[1].annotate(f'ID: {_bubble.id}', (_bubble.current_centroid[0] + 5, _bubble.current_centroid[1] + 5), color='white', fontsize=8)  # Add label with bubble ID using annotate
        
    plt.tight_layout()
    plt.savefig(r"C:\Users\jkf1g22\OneDrive - University of Southampton\Documents\_PostGraduateResearch\_Tools\Unbub-lievably_Good_Tracker\test2\\" + f'frame_{k:04d}.png')
    # plt.show()
    # time.sleep(0.5)
    plt.close(fig)

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





def detect_and_track_bubbles(out_dir, measured_flowrate, video_filepath, start_time=0, end_time=120, image_crop=(750, 300, 250, 450), searchBox_bounds=(0, 355, 250, 50), y_cutoff = 150):
    """Main function to process the video and track bubbles."""
    background_frame_width = 20

    # Define thresholds for object detection
    Thresh = 20   #20
    AreaThresh = 5

    # Initialise bubble identification lists
    bubble_list = []
    tracking_bubble_list = []
    bubble_count = 0

    ## Main Code ##
    vidObj = cv2.VideoCapture(video_filepath)                            # Check if the video file is opened successfully
    assert vidObj.isOpened(), "Failed to open the video file."
    frame_count = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    assert frame_count > 0, "The video file has no frames."         # Check if the video has frames

    num_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(vidObj.get(cv2.CAP_PROP_FPS))
    start_frame = int(start_time * frame_rate)
    end_frame = min(int(end_time * frame_rate), num_frames)

    # Start loop through the time frame
    for k in range(start_frame, end_frame):

        # Read the current frame
        vidObj.set(cv2.CAP_PROP_POS_FRAMES, k)
        ret, frame = vidObj.read()

                ############### Generate background frame ##############
        result1 = k - (background_frame_width / 2)       #gets rolling average background frames
        result2 = k + 1 + (background_frame_width / 2)
        before, after = [], []
        if result1 < 0 :  # if the frame is before the start of the video
            before = get_frame_series(vidObj, k - 1 - (background_frame_width / 2) - result1, (background_frame_width / 2) - result1)
            after = get_frame_series(vidObj, k + 1, (background_frame_width / 2) - result1)
        elif  result2 > end_frame: 
            remainder = end_frame - (k + 1)
            buffer_size = background_frame_width / 2
            before = get_frame_series(vidObj, k - 1 - (buffer_size + (buffer_size - remainder)), background_frame_width - remainder)
            if remainder > 0:
                after = get_frame_series(vidObj, k + 1, remainder)
        else:
            before = get_frame_series(vidObj, k - 1 - (background_frame_width / 2), (background_frame_width / 2))
            after = get_frame_series(vidObj, k + 1, (background_frame_width / 2))

        median_background = np.median(np.array(before + after), axis=0).astype(np.uint8)[image_crop[1]:(image_crop[1]+image_crop[3]), image_crop[0]:(image_crop[0]+image_crop[2])]
        
        # image processing
        cropped_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[image_crop[1]:(image_crop[1]+image_crop[3]), image_crop[0]:(image_crop[0]+image_crop[2])]
        subtracted_background = ((abs(cropped_gray_frame.astype(np.int8)) - abs(median_background.astype(np.int8))) > Thresh).astype(np.uint8)
        #cv2.imshow('Subtracted Background', subtracted_background*255)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(subtracted_background)  # Find connected components in the binary image
        #Filling in small gaps between indentified componants
        kernel = np.ones((2, 2), np.uint8)  # 7x7 kernel to fill 3-pixel gaps  - could be changed to 5x5 or 3x3
        processed_frame = cv2.morphologyEx(subtracted_background, cv2.MORPH_CLOSE, kernel) # Apply morphological closing (dilation followed by erosion)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_frame) # Find connected components in the binary image

        for n in range(num_labels - 1, 0, -1):
            if stats[n, cv2.CC_STAT_AREA] < AreaThresh:     # Remove small components based on area threshold
                processed_frame[labels == n] = 0
                
        # Recalculate the number of connected components after removing small components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_frame)

        _hull_points = []








        #########################################################
        ########### Bubble identification and tracking ##########
        #########################################################


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
                if searchBox_bounds[0] <= _x <= (searchBox_bounds[0] + searchBox_bounds[2]) and searchBox_bounds[1] <= _y <= (searchBox_bounds[1] + searchBox_bounds[3]):                                                       # Check if the centroid is within the search box bounds   
                    if (n-1) not in col_ind:                                                                                                                # If the centroid is not already assigned to a bubble, create a new bubble object 
                        tracking_bubble_list.append(bubble(bubble_count, centroids[n], k/frame_rate))
                        bubble_count += 1



        ####### if there are no bubbles being tracked ##########
        else:
            for n in range(1, num_labels):                                                                                                                                                                      # If there are no bubbles in the tracking list, create new bubble objects for all centroids within the search box bounds        
                if searchBox_bounds[0] <= centroids[n][0] <= (searchBox_bounds[0] + searchBox_bounds[2]) and searchBox_bounds[1] <= centroids[n][1] <= (searchBox_bounds[1] + searchBox_bounds[3]):
                    tracking_bubble_list.append(bubble(bubble_count, centroids[n], k/frame_rate))                                                                                                               # Create a new bubble object with timestamp of current frame number
                    bubble_count += 1





        ####### Remove bubbles that have crossed the y_cutoff ##########
        if len(tracking_bubble_list) != 0:                                                      # Cycles through tracked bubbles removes if centeroid position is above yaxis cuttoff. When removed bubble is added to main bubble list and removed from tracking list
            for i in range(len(tracking_bubble_list) - 1, -1, -1):    
                if tracking_bubble_list[i].current_centroid[1] < y_cutoff:
                    tracking_bubble_list[i].set_current_centeroid([0, 0])
                    bubble_list.append(tracking_bubble_list[i])
                    tracking_bubble_list.pop(i)






        progress = (k - start_frame + 1) / (end_frame - start_frame)
        bar_length = 40
        block = int(round(bar_length * progress))
        progress_bar = f"\rProcessing frames: [{'#' * block + '-' * (bar_length - block)}] {int(progress * 100)}%"
        sys.stdout.write(progress_bar)
        sys.stdout.flush()
        if k == end_frame - 1:
            print()

        plot_frame(tracking_bubble_list, processed_frame, cropped_gray_frame, searchBox_bounds, y_cutoff, _hull_points, k)







    ##Calculations##
    # moves all bubbles from tracking to bubble list
    if len(tracking_bubble_list) != 0: 
            for i in range(len(tracking_bubble_list) - 1, -1, -1):  # Iterate in reverse to avoid indexing issues
                tracking_bubble_list[i].set_current_centeroid([0, 0])
                bubble_list.append(tracking_bubble_list[i])
                tracking_bubble_list.pop(i)

    timeline = []
    volumes = []
    radii = []

    for _bubble in bubble_list:
        volumes.append(np.mean(_bubble.volume_history))
        radii.append(np.mean(_bubble.radii_history))
        timeline.append({_bubble.timestamp: [np.mean(_bubble.radii_history), np.mean(_bubble.volume_history)]})
    volumes = [v for v in volumes if not np.isnan(v)] # Remove NaN values from volume and radii history
    radii = [r for r in radii if not np.isnan(r)]
    
    # plot distributions volumeand radii
    # calc time from frame start to end
    duration = end_time - start_time
    multiplier = 60 / duration
    total_volume_L = 0
    for v in volumes:
        total_volume_L += v * 1000  # in L
    gas_flux = multiplier * total_volume_L  # in l/min

    print(f"Measured Flowrate: {measured_flowrate} L/min started at {start_time}s to {end_time}s")
    print(f"Total Volume: {total_volume_L} L")
    print(f"Gas Flux: {gas_flux} L/min")
    print(f"Duration: {(end_frame - start_frame) / frame_rate}")

    results = {
        "gas_flux_L_per_min": gas_flux,
        "duration_s": (end_frame - start_frame) / frame_rate,
        "total_volume_L": total_volume_L,
        "number_of_bubbles": len(bubble_list),
        "timeline": timeline
    }



    with open(os.path.join(out_dir, (str(measured_flowrate)+"_results_CO2.json")), "w") as f:
        json.dump(results, f, indent=4)




if __name__ == "__main__":

    ###########     0.02 L/min     ###########  
    parser = argparse.ArgumentParser(description="Domain generalization testbed")
    parser.add_argument("--video_filepath", type=str, default=r"C:\Users\jkf1g22\OneDrive - University of Southampton\Desktop\Optical varying Flow Rate\CO2\individual\CO2_0.02Lmin.mp4")
    parser.add_argument("--outDir", type=str, default=r"C:\Users\jkf1g22\OneDrive - University of Southampton\Documents\_PostGraduateResearch\_Tools\Unbub-lievably_Good_Tracker")
    parser.add_argument("--measured_flowrate", type=float, default=0.02)  # in L/min
    parser.add_argument("--frame_crop_bounds", type=tuple, default=(750, 300, 250, 450))    #(x, y, w, h)
    parser.add_argument("--searchBox_bounds", type=tuple, default=(50, 365, 75, 40))    #(x, y, w, h)  small search box(50, 365, 50, 40)
    parser.add_argument("--y_cutoff", type=int, default=150)

    args = parser.parse_args()

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 0, 30, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    # detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 30, 60, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    # detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 60, 120, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    # detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 120, 180, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles


    # ###########     0.043 L/min     ###########  
    # parser = argparse.ArgumentParser(description="Domain generalization testbed")
    # parser.add_argument("--video_filepath", type=str, default=r"C:\Users\jkf1g22\OneDrive - University of Southampton\Desktop\Optical varying Flow Rate\CO2\individual\CO2_0.043Lmin.mp4")
    # parser.add_argument("--outDir", type=str, default=r"C:\Users\jkf1g22\OneDrive - University of Southampton\Documents\_PostGraduateResearch\_Tools\Unbub-lievably_Good_Tracker")
    # parser.add_argument("--measured_flowrate", type=float, default=0.043)  # in L/min
    # parser.add_argument("--frame_crop_bounds", type=tuple, default=(750, 300, 250, 450))    #(x, y, w, h)
    # parser.add_argument("--searchBox_bounds", type=tuple, default=(50, 365, 75, 40))    #(x, y, w, h)  small search box(50, 365, 50, 40)
    # parser.add_argument("--y_cutoff", type=int, default=150)

    # args = parser.parse_args()

    # detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 0, 30, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    # detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 30, 60, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    # detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 60, 120, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    # detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 120, 180, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles



    # ###########     0.093 L/min     ###########  
    # parser = argparse.ArgumentParser(description="Domain generalization testbed")
    # parser.add_argument("--video_filepath", type=str, default=r"C:\Users\jkf1g22\OneDrive - University of Southampton\Desktop\Optical varying Flow Rate\CO2\individual\CO2_0.093Lmin.mp4")
    # parser.add_argument("--outDir", type=str, default=r"C:\Users\jkf1g22\OneDrive - University of Southampton\Documents\_PostGraduateResearch\_Tools\Unbub-lievably_Good_Tracker")
    # parser.add_argument("--measured_flowrate", type=float, default=0.093)  # in L/min
    # parser.add_argument("--frame_crop_bounds", type=tuple, default=(750, 300, 250, 450))    #(x, y, w, h)
    # parser.add_argument("--searchBox_bounds", type=tuple, default=(50, 365, 75, 40))    #(x, y, w, h)  small search box(50, 365, 50, 40)
    # parser.add_argument("--y_cutoff", type=int, default=150)

    # args = parser.parse_args()

    # detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 0, 30, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    # detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 30, 60, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    # detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 60, 120, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    # detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 120, 180, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles



    # ###########     0.2 L/min     ###########  
    # parser = argparse.ArgumentParser(description="Domain generalization testbed")
    # parser.add_argument("--video_filepath", type=str, default=r"C:\Users\jkf1g22\OneDrive - University of Southampton\Desktop\Optical varying Flow Rate\CO2\individual\CO2_0.2Lmin.mp4")
    # parser.add_argument("--outDir", type=str, default=r"C:\Users\jkf1g22\OneDrive - University of Southampton\Documents\_PostGraduateResearch\_Tools\Unbub-lievably_Good_Tracker")
    # parser.add_argument("--measured_flowrate", type=float, default=0.2)  # in L/min
    # parser.add_argument("--frame_crop_bounds", type=tuple, default=(750, 300, 250, 450))    #(x, y, w, h)
    # parser.add_argument("--searchBox_bounds", type=tuple, default=(50, 365, 75, 40))    #(x, y, w, h)  small search box(50, 365, 50, 40)
    # parser.add_argument("--y_cutoff", type=int, default=150)

    # args = parser.parse_args()

    # detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 0, 30, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    # detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 30, 60, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    # detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 60, 120, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    # detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 120, 180, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles


    ###########     0.431 L/min     ###########  
    parser = argparse.ArgumentParser(description="Domain generalization testbed")
    parser.add_argument("--video_filepath", type=str, default=r"C:\Users\jkf1g22\OneDrive - University of Southampton\Desktop\Optical varying Flow Rate\CO2\individual\CO2_0.431Lmin.mp4")
    parser.add_argument("--outDir", type=str, default=r"C:\Users\jkf1g22\OneDrive - University of Southampton\Documents\_PostGraduateResearch\_Tools\Unbub-lievably_Good_Tracker")
    parser.add_argument("--measured_flowrate", type=float, default=0.431)  # in L/min
    parser.add_argument("--frame_crop_bounds", type=tuple, default=(750, 300, 250, 450))    #(x, y, w, h)
    parser.add_argument("--searchBox_bounds", type=tuple, default=(50, 365, 75, 40))    #(x, y, w, h)  small search box(50, 365, 50, 40)
    parser.add_argument("--y_cutoff", type=int, default=150)

    args = parser.parse_args()

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 0, 30, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 30, 60, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 60, 120, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 120, 180, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles



    ###########     0.92 L/min     ###########  
    parser = argparse.ArgumentParser(description="Domain generalization testbed")
    parser.add_argument("--video_filepath", type=str, default=r"C:\Users\jkf1g22\OneDrive - University of Southampton\Desktop\Optical varying Flow Rate\CO2\individual\CO2_0.92Lmin.mp4")
    parser.add_argument("--outDir", type=str, default=r"C:\Users\jkf1g22\OneDrive - University of Southampton\Documents\_PostGraduateResearch\_Tools\Unbub-lievably_Good_Tracker")
    parser.add_argument("--measured_flowrate", type=float, default=0.92)  # in L/min
    parser.add_argument("--frame_crop_bounds", type=tuple, default=(750, 300, 250, 450))    #(x, y, w, h)
    parser.add_argument("--searchBox_bounds", type=tuple, default=(50, 365, 100, 40))    #(x, y, w, h)  small search box(50, 365, 50, 40)
    parser.add_argument("--y_cutoff", type=int, default=150)

    args = parser.parse_args()

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 0, 30, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 30, 60, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 60, 120, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 120, 180, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles



    ###########     2 L/min     ###########  
    parser = argparse.ArgumentParser(description="Domain generalization testbed")
    parser.add_argument("--video_filepath", type=str, default=r"C:\Users\jkf1g22\OneDrive - University of Southampton\Desktop\Optical varying Flow Rate\CO2\individual\CO2_2Lmin.mp4")
    parser.add_argument("--outDir", type=str, default=r"C:\Users\jkf1g22\OneDrive - University of Southampton\Documents\_PostGraduateResearch\_Tools\Unbub-lievably_Good_Tracker")
    parser.add_argument("--measured_flowrate", type=float, default=2)  # in L/min
    parser.add_argument("--frame_crop_bounds", type=tuple, default=(750, 300, 250, 450))    #(x, y, w, h)
    parser.add_argument("--searchBox_bounds", type=tuple, default=(50, 365, 125, 40))    #(x, y, w, h)  small search box(50, 365, 50, 40)
    parser.add_argument("--y_cutoff", type=int, default=150)

    args = parser.parse_args()

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 0, 30, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 30, 60, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 60, 120, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 120, 180, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    ###########     4.3 L/min     ###########  
    parser = argparse.ArgumentParser(description="Domain generalization testbed")
    parser.add_argument("--video_filepath", type=str, default=r"C:\Users\jkf1g22\OneDrive - University of Southampton\Desktop\Optical varying Flow Rate\CO2\individual\CO2_4.3Lmin.mp4")
    parser.add_argument("--outDir", type=str, default=r"C:\Users\jkf1g22\OneDrive - University of Southampton\Documents\_PostGraduateResearch\_Tools\Unbub-lievably_Good_Tracker")
    parser.add_argument("--measured_flowrate", type=float, default=4.3)  # in L/min
    parser.add_argument("--frame_crop_bounds", type=tuple, default=(750, 300, 250, 450))    #(x, y, w, h)
    parser.add_argument("--searchBox_bounds", type=tuple, default=(50, 365, 150, 40))    #(x, y, w, h)  small search box(50, 365, 50, 40)
    parser.add_argument("--y_cutoff", type=int, default=150)

    args = parser.parse_args()

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 0, 30, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 30, 60, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 60, 120, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 120, 180, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    ###########     9.2 L/min     ###########  
    parser = argparse.ArgumentParser(description="Domain generalization testbed")
    parser.add_argument("--video_filepath", type=str, default=r"C:\Users\jkf1g22\OneDrive - University of Southampton\Desktop\Optical varying Flow Rate\CO2\individual\CO2_9.2Lmin.mp4")
    parser.add_argument("--outDir", type=str, default=r"C:\Users\jkf1g22\OneDrive - University of Southampton\Documents\_PostGraduateResearch\_Tools\Unbub-lievably_Good_Tracker")
    parser.add_argument("--measured_flowrate", type=float, default=9.2)  # in L/min
    parser.add_argument("--frame_crop_bounds", type=tuple, default=(750, 300, 250, 450))    #(x, y, w, h)
    parser.add_argument("--searchBox_bounds", type=tuple, default=(50, 365, 200, 40))    #(x, y, w, h)  small search box(50, 365, 50, 40)
    parser.add_argument("--y_cutoff", type=int, default=150)

    args = parser.parse_args()

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 0, 30, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 30, 60, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 60, 120, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 120, 180, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    ###########     20 L/min     ###########  
    parser = argparse.ArgumentParser(description="Domain generalization testbed")
    parser.add_argument("--video_filepath", type=str, default=r"C:\Users\jkf1g22\OneDrive - University of Southampton\Desktop\Optical varying Flow Rate\CO2\individual\CO2_20Lmin.mp4")
    parser.add_argument("--outDir", type=str, default=r"C:\Users\jkf1g22\OneDrive - University of Southampton\Documents\_PostGraduateResearch\_Tools\Unbub-lievably_Good_Tracker")
    parser.add_argument("--measured_flowrate", type=float, default=20)  # in L/min
    parser.add_argument("--frame_crop_bounds", type=tuple, default=(750, 300, 250, 450))    #(x, y, w, h)
    parser.add_argument("--searchBox_bounds", type=tuple, default=(50, 365, 200, 40))    #(x, y, w, h)  small search box(50, 365, 50, 40)
    parser.add_argument("--y_cutoff", type=int, default=150)

    args = parser.parse_args()

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 0, 30, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 30, 60, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 60, 120, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles

    detect_and_track_bubbles(args.outDir, args.measured_flowrate, args.video_filepath, 120, 180, image_crop=args.frame_crop_bounds, searchBox_bounds=args.searchBox_bounds, y_cutoff=args.y_cutoff)  # Call the main function to process the video and track bubbles


    print("done")