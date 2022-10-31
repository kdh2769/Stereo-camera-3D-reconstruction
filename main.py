import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utils import stereo_calibration, closestDistanceBetweenLines, read_aruco_ponints
import cv2.aruco as aruco
import open3d as o3d
import pyrealsense2 as rs
import pandas as pd 

def plot_target_point(left_uv, right_uv, img_undistL, img_undistR): 
    # left camera 
    plt.subplot(1, 2, 1)
    plt.title('left')
    plt.imshow(img_undistL[:,:,[2,1,0]])
    plt.scatter(left_uv[0], left_uv[1])

    # right camera 
    plt.subplot(1, 2, 2)
    plt.title('right')
    plt.imshow(img_undistR[:,:,[2,1,0]])
    plt.scatter(right_uv[0], right_uv[1])
    plt.show()

def find_projection_matrix(left_intrinsic, right_intrinsic):
    left_extrinsic = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    left_projection_matrix = left_intrinsic @ left_extrinsic 

    right_extrinsic = np.concatenate([rot, trans], axis = -1)
    right_projection_matrix = right_intrinsic @ right_extrinsic

    return left_projection_matrix, right_projection_matrix, left_extrinsic, right_extrinsic

def find_depth (left_uv, right_uv, left_intrinsic, right_intrinsic, right_extrinsic):
    left_origin = np.array([[0, 0, 0, 1]])
    right_origin = right_extrinsic @ left_origin.T

    left_uv_homo = np.concatenate([left_uv, [1]], axis=0)
    left_xyz = np.linalg.inv(left_intrinsic) @ left_uv_homo.reshape(3, -1)

    right_uv_homo = np.concatenate([right_uv, [1]], axis=0)
    right_vector = np.linalg.inv(right_intrinsic) @ right_uv_homo.reshape(3, -1)
    
    right_xyz = np.concatenate([right_vector, [[1]]], axis=0)
    right_xyz = right_extrinsic@ right_xyz 
    left_origin = np.array([0, 0, 0])
    right_origin = np.squeeze(right_origin.T)
    left_xyz = np.squeeze(left_xyz.T)
    right_xyz = np.squeeze(right_xyz.T)
    right_world_point, left_world_point, distance = closestDistanceBetweenLines (right_origin, right_xyz, left_origin, left_xyz, clampAll=False)

    return left_world_point, right_world_point, left_origin, right_origin

def plot_projection_ray(left_world_point, right_world_point, left_origin, right_origin): 
    fig = plt.figure(figsize=(20,20))
    ax = fig.gca(projection='3d')
    right_origin_to_xyz = np.concatenate([[right_origin],[right_world_point]], axis=0)
    X = right_origin_to_xyz[:,0]
    Y = right_origin_to_xyz[:,1]
    Z = right_origin_to_xyz[:,2]
    ax.plot3D(X,Y,Z)

    left_origin_to_xyz = np.concatenate([[left_origin], [left_world_point]], axis=0)
    X = left_origin_to_xyz[:,0]
    Y = left_origin_to_xyz[:,1]
    Z = left_origin_to_xyz[:,2]
    ax.plot3D(X,Y,Z)

    points = np.concatenate([right_origin_to_xyz, left_origin_to_xyz], axis=0)
    X = points[:,0]
    Y = points[:,1]
    Z = points[:,2]

    ax.scatter3D(X, Y, Z, s= 40)
    ax.set_xlabel('X label')
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')
    axes_limit = 1000
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-1000, 1000)
    ax.set_zlim(-axes_limit, axes_limit)

    plt.show()

def scatter_aruco_points(left_world_points, right_world_points) : 
    left_world_points, right_world_points = np.array(left_world_points), np.array(right_world_points)
    
    fig = plt.figure(figsize=(20,20))
    ax = fig.gca(projection='3d')
    
    X = left_world_points[:,0]
    Y = left_world_points[:,1]
    Z = left_world_points[:,2]
    ax.scatter3D(X, Y, Z, s= 40)

    X = right_world_points[:,0]
    Y = right_world_points[:,1]
    Z = right_world_points[:,2]
    ax.scatter3D(X, Y, Z, s= 40)
    ax.set_xlabel('X label')
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')

    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    ax.set_zlim(-1000, 1000)
    plt.show()
    return 0

def video(left_intrinsic, right_intrinsic, right_extrinsic): 
    cap2 = cv.VideoCapture(0)
    
    points = []
    pipeline = rs.pipeline()
    config_left = rs.config()

    config_left.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config_left.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    param_markers = aruco.DetectorParameters_create()

    pipeline.start(config_left)

    vis = o3d.visualization.Visualizer()
    vis.create_window("depth", width = 640, height = 480)
    pointcloud = o3d.geometry.PointCloud()
    text = 'Depth : 0cm'
    while True:
        
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        succes2, frame2 = cap2.read()
        if not depth_frame or not color_frame:
            continue
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if color_frame : 
            gray_frame_left = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
            marker_corners_left, marker_ID, reject = aruco.detectMarkers(
                gray_frame_left, marker_dict, parameters = param_markers
            )
            gray_frame_right = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
            marker_corners_right, marker_ID, reject = aruco.detectMarkers(
                gray_frame_right, marker_dict, parameters = param_markers
            )

        if marker_corners_left and marker_corners_right: 
            
            vis.add_geometry(pointcloud)
            pointcloud.clear()
            for corners_left, corners_right in zip(marker_corners_left, marker_corners_right):
                cv.polylines(
                    color_image, [corners_left.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
                )
                cv.polylines(
                    frame2, [corners_right.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
                )
            
            mean_x_left, mean_y_left = corners_left[0].mean(axis=0)
            mean_x_right, mean_y_right = corners_right[0].mean(axis=0)

            left_uv = np.array([mean_x_left, mean_y_left])
            right_uv = np.array([mean_x_right, mean_y_right])

            left_world_point, right_world_point, _, _ = find_depth(left_uv, right_uv, left_intrinsic, right_intrinsic, right_extrinsic)
            real_depth = np.round(right_world_point[2], -1) / 10
            text = f'Depth  : {real_depth}cm'
            print(text)
             
            points.append(list(right_world_point))
            points_np = np.array(points)
            points_np = points_np @ np.array([[-1, 0, 0], [0, -1, 0],[0, 0, 1]])
            a = o3d.utility.Vector3dVector(points_np)
            
            pointcloud.points = a

            vis.update_geometry(pointcloud)
            vis.poll_events()
            vis.update_renderer()

        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.31), cv.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv.INTER_AREA)
            depth_colormap = cv.flip(depth_colormap, 1)
            cv.putText(depth_colormap, text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            cv.flip(resized_color_image, 1)
            cv.flip(frame2, 1)
            images = np.hstack((cv.flip(resized_color_image, 1), depth_colormap, cv.flip(frame2, 1)))
        else:
            depth_colormap = cv.flip(depth_colormap , 1)
            cv.putText(depth_colormap, text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            images = np.hstack((cv.flip(color_image, 1), depth_colormap, cv.flip(frame2, 1)))


        cv.namedWindow('left_depth_right', cv.WINDOW_AUTOSIZE)
        cv.imshow('left_depth_right', images)
        
        k = cv.waitKey(1)
        if k == 27:
            cv.destroyAllWindows()
            break
    
    fig = plt.figure(figsize=(20,20))
    ax = fig.gca(projection='3d')
    X = points_np[:,0]
    Y = points_np[:,1]
    Z = points_np[:,2]
    ax.scatter3D(X, Y, Z, s= 20)

    ax.set_xlabel('X label')
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')

    ax.set_xlim(-500, 500)
    ax.set_ylim(-200, 200)
    ax.set_zlim(-30, 1000)
    plt.show()


if __name__ == '__main__' :
    # calibrate 
    img_undistL, img_undistR, left_intrinsic, right_intrinsic, rot, trans = stereo_calibration() 
    
    # projection matrix 
    left_projection_matrix, right_projection_matrix, left_extrinsic, right_extrinsic = find_projection_matrix(left_intrinsic, right_intrinsic)

    video(left_intrinsic, right_intrinsic, right_extrinsic)
    ######### find one point #########
    # # set left/right origin and target point(u, v)
    # left_uv = # left
    # right_uv = # right

    # # draw target point
    # plot_target_point(left_uv, right_uv,img_undistL, img_undistR)

    # # find world point 
    # left_world_point, right_world_point, left_origin, right_origin = find_depth(left_uv, right_uv, left_intrinsic, right_intrinsic, right_extrinsic) 

    # # draw projection ray
    # plot_projection_ray(left_world_point, right_world_point, left_origin, right_origin)


    ######### find moving points  #########
    # read aruco moving points 
    # left_moving_points, right_moving_points = read_aruco_ponints()

    # # find depth 
    # left_world_points = []
    # right_world_points = []
    # for left_uv, right_uv in zip(left_moving_points, right_moving_points) :
    #     left_world_point, right_world_point, left_origin, right_origin = find_depth (left_uv, right_uv, left_intrinsic, right_intrinsic, right_extrinsic)
    #     left_world_points.append(left_world_point.tolist())
    #     right_world_points.append(right_world_point.tolist())

    # scatter_aruco_points(left_world_points, right_world_points)
    
    



