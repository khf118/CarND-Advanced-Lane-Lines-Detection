import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import listdir
from moviepy.editor import VideoFileClip
import random

class Line():
		def __init__(self):
		    # was the line detected in the last iteration?
		    self.detected = False  
		    # x values of the last n fits of the line
		    self.recent_xfitted = [] 
		    # y values of the last n fits of the line
		    self.recent_yfitted = [] 
		    #average x values of the fitted line over the last n iterations
		    self.bestx = None     
		    #polynomial coefficients averaged over the last n iterations
		    self.best_fit = None  
		    #polynomial coefficients for the most recent fit
		    self.current_fit = [np.array([False])]  
		    #radius of curvature of the line in some units
		    self.radius_of_curvature = None 
		    #distance in meters of vehicle center from the line
		    self.line_base_pos = None 
		    #difference in fit coefficients between last and new fits
		    self.diffs = np.array([0,0,0], dtype='float') 
		    #x values for detected line pixels
		    self.allx = None  
		    #y values for detected line pixels
		    self.ally = None

		def append(self,xfitted,binary_warped):
			if len(self.recent_xfitted) < 10:
				self.recent_xfitted.append(xfitted)
			else:
				self.recent_xfitted= self.recent_xfitted[1:]
				self.recent_xfitted.append(xfitted)
			ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

			#repeat ploty len(recent_xfitted) times
			y = np.array([])
			for i in range(0,len(self.recent_xfitted)):
				y = np.append(ploty,y)
			
			self.best_fit = np.polyfit(y, np.array(self.recent_xfitted).flatten(),2)
			self.bestx = self.best_fit[0]*ploty**2 + self.best_fit[1]*ploty + self.best_fit[2]

		def best_fit(self,ploty):
			y = np.copy(ploty)
			for x in xrange(1,10):
				y = y.append(y)
			self.best_fit = np.polyfit(y, self.recent_xfitted.reshape(self.recent_xfitted.shape[0]*self.recent_xfitted.shape[1],1), 2)
			
		

#Read Caliberation images
images = listdir('./camera_cal')
last_Right = Line()
last_Left = Line()
Left = Line()
Right = Line()

i = 0
error = 0
img_points = []
obj_points = []

obj_p = np.zeros((6*9,3), np.float32)
obj_p[:,:2] = np.mgrid[0:9,0:6].T.reshape([-1,2])

for path in images:
	print(path)
	img = cv2.imread('./camera_cal/'+path)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
	
	if ret == True:
		img_points.append(corners)
		obj_points.append(obj_p)

		# draw and display the corners
		#img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
		#plt.imshow(img)

ret, mtx, dist, rvecs, tvecs= cv2.calibrateCamera(obj_points,img_points, gray.shape[::-1],None,None)

#perspective transform
def perspective_transform(img):
    # Convert undistorted image to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image
    gray = img
    # Grab the image shape
    img_size = (gray.shape[1], gray.shape[0])

    # For source points I'm grabbing the outer four detected corners
    src = np.float32([[253, 680], [604,443], [670,443], [1051, 680]])
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    dst = np.float32([[253, 680], [253, 0], [1051, 0], [1051, 680]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    # Return the resulting image and matrix
    return warped, Minv



def dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def pipeline(img, s_thresh=(70, 255), sx_thresh=(40, 255)):
    global i
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]
    h_channel[(h_channel < 17)] = 0
    h_channel[(h_channel > 20)] = 0
    image = hls
    hls[:,:,1] = cv2.equalizeHist(hls[:,:,1].astype(np.uint8))
    hls[:,:,2] = cv2.equalizeHist(hls[:,:,2].astype(np.uint8))

    l_channel = np.copy(hls[:,:,1])
    s = np.copy(hls[:,:,2])
    s[(s<=250)] = 0
    s[(s>250)] = 1
    l_channel_aux = np.copy(l_channel)
    l_channel_aux[(l_channel_aux<=250)] = 0
    l_channel_aux[(l_channel_aux>250)] = 1
    l_channel = np.zeros_like(l_channel)
    l_channel_right= l_channel_aux[:,640:]
    l_channel[:,640:] = l_channel_right
    sobelx = cv2.Sobel(np.copy(hls[:,:,1]), cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
     
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0])] = 1
    color_binary = np.dstack(( l_channel, sxbinary, s)) * 255
    color_binary = color_binary[:,:,0] * 0.33 + color_binary[:,:,1] * 0.33 + color_binary[:,:,2] * 0.33
    return color_binary

def sanity_check(right_line, left_line):
	#Set the top and bottom distances between the left and right lines and check if the lines are approxiametly parallel
	top_distance = right_line[0] - left_line[0]
	bottom_distance = right_line[right_line.shape[0]-1] - left_line[right_line.shape[0]-1]
	if np.abs(top_distance - bottom_distance) > 100 or np.abs(bottom_distance - 800) > 100:
		return False
	return True

def curvature_sanity_check(left,right):
	if np.abs(left - right) > 200:
		return False
	return True

def process_image(image):

	global last_right_fitx
	global last_left_fitx
	global i
	global last_ploty
	global error
	global Left
	global Right
	global last_Left
	global last_Right
	img= image
	#undistort
	dst = cv2.undistort(img, mtx, dist, None, mtx)
	img = dst
	#perspective tranform
	img, Minv = perspective_transform(img)
	#color selection
	img = pipeline(img)
	
	i+=1
	
	img= img.reshape(img.shape[0],img.shape[1],1)
	binary = img

	binary_warped = binary
	# histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	# output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint
	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
	    # Identify window boundaries in x and y (and right and left)
	    win_y_low = binary_warped.shape[0] - (window+1)*window_height
	    win_y_high = binary_warped.shape[0] - window*window_height
	    win_xleft_low = leftx_current - margin
	    win_xleft_high = leftx_current + margin
	    win_xright_low = rightx_current - margin
	    win_xright_high = rightx_current + margin
	    # Draw the windows on the visualization image
	    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
	    (0,255,0), 2) 
	    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
	    (0,255,0), 2) 
	    # Identify the nonzero pixels in x and y within the window
	    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
	    (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
	    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
	    (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
	    # Append these indices to the lists
	    left_lane_inds.append(good_left_inds)
	    right_lane_inds.append(good_right_inds)
	    # If you found > minpix pixels, recenter next window on their mean position
	    if len(good_left_inds) > minpix:
	        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
	    if len(good_right_inds) > minpix:        
	        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Fit a second order polynomial to each
	if rightx.shape[0] != 0 and leftx.shape[0] != 0 and righty.shape[0] != 0 and lefty.shape[0] != 0:
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)

	# Generate x and y values for plotting
		ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
		Left.append(left_fitx,binary_warped)
		Right.append(right_fitx,binary_warped)
		out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
		if sanity_check(Right.bestx,Left.bestx) == False:
			Right = last_Right
			Left = last_Left
	else:
		Right = last_Right
		Left = last_Left

	if Right.bestx is None:
		return binary_warped 
	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension
	y_eval = 70
	left_fitx = Left.bestx
	right_fitx = Right.bestx
	left_fitx = np.abs(left_fitx)
	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
	# Calculate the new radii of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	Left.curvature = left_curverad
	Right.curvature = right_curverad
	# Now our radius of curvature is in meters
	print(left_curverad, 'm', right_curverad, 'm')
	if curvature_sanity_check(left_curverad,right_curverad) == False:
		Right = last_Right
		Left = last_Left
		error+=1
	# Create an image to draw the lines on
	warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))
	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (binary_warped.shape[1], binary_warped.shape[0])) 
	# Combine the result with the original image
	#dst = cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)
	result = cv2.addWeighted(dst, 1, newwarp, 0.3, 0)
	last_Right = Right
	last_Left = Left
	return result


clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile("project_video_output3.mp4", audio=False)

print(error)
#img = cv2.imread('./test_images/straight_lines1.jpg')
#process_image(img)
