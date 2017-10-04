import os
import cv2
import numpy as np
import math
import dlib

predictor_path = "shape_predictor_68_face_landmarks.dat"


def detect_landmarks(image):

    # obtain detector and predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # convert image to numpy array
    numpy_image = np.asanyarray(image)
    numpy_image.flags.writeable = True

    # output list
    face_landmark_tuples = []

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should up sample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    detected_faces = detector(numpy_image, 1)

    print("Number of faces detected: {}".format(len(detected_faces)))

    for k, rect in enumerate(detected_faces):
        # Get the landmarks/parts for the face in box rect.
        shape = predictor(numpy_image, rect)
        face_landmark_tuples.append((k, rect, shape))

    return face_landmark_tuples


# Create points for each image in folder.
def create_points(path):

    points_array = []

    # List all files in the directory and read points from text files one by one
    for filePath in sorted(os.listdir(path)):

        if filePath.endswith(".jpg"):
            # Read image found.
            image = cv2.imread(os.path.join(path, filePath))

            # detect faces and landmarks
            face_landmark_tuples = detect_landmarks(image)

            for k, rect, shape in face_landmark_tuples:
                # calculate points for photo
                points = []

                for i in xrange(0, shape.num_parts, 1):
                    x, y = "{}\n".format(shape.part(i)).replace("(", "").replace(")", "").replace(",", "").split()
                    points.append((int(x), int(y)))

                print("created points for " + filePath)

                points_array.append(points)

    return points_array


# Read all jpg images in folder.
def read_images(path):
    
    # Create array of array of images.
    images_array = []
    
    # List all files in the directory and read points from text files one by one
    for filePath in sorted(os.listdir(path)):
       
        if filePath.endswith(".jpg"):
            # Read image found.
            read_image = cv2.imread(os.path.join(path,filePath))

            # Convert to floating point
            read_image = np.float32(read_image)/255.0

            # Add to array of images
            images_array.append(read_image)
            
    return images_array


# Compute similarity transform given two sets of two points.
# OpenCV requires 3 pairs of corresponding points.
# We are faking the third one.
def similarity_transform(inPoints, outPoints):
    s60 = math.sin(60*math.pi/180)
    c60 = math.cos(60*math.pi/180)
  
    in_pts = np.copy(inPoints).tolist()
    out_pts = np.copy(outPoints).tolist()
    
    xin = c60*(in_pts[0][0] - in_pts[1][0]) - s60*(in_pts[0][1] - in_pts[1][1]) + in_pts[1][0]
    yin = s60*(in_pts[0][0] - in_pts[1][0]) + c60*(in_pts[0][1] - in_pts[1][1]) + in_pts[1][1]

    in_pts.append([np.int(xin), np.int(yin)])
    
    xout = c60*(out_pts[0][0] - out_pts[1][0]) - s60*(out_pts[0][1] - out_pts[1][1]) + out_pts[1][0]
    yout = s60*(out_pts[0][0] - out_pts[1][0]) + c60*(out_pts[0][1] - out_pts[1][1]) + out_pts[1][1]

    out_pts.append([np.int(xout), np.int(yout)])
    
    tform = cv2.estimateRigidTransform(np.array([in_pts]), np.array([out_pts]), False)
    
    return tform


# Check if a point is inside a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

# Calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    # Create subdiv
    subdiv = cv2.Subdiv2D(rect)
   
    # Insert points into subdiv
    for p in points:
        subdiv.insert((p[0], p[1]))

    # List of triangles. Each triangle is a list of 3 points ( 6 numbers )
    triangleList = subdiv.getTriangleList()

    # Find the indices of triangles in the points array

    delaunayTri = []
    
    for t in triangleList:
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        
        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            ind = []
            for j in xrange(0, 3):
                for k in xrange(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)                            
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))

    return delaunayTri


def constrain_point(p, w, h):
    p = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))
    return p


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2):

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in xrange(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ((1.0, 1.0, 1.0) - mask)
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect


if __name__ == '__main__' :
    
    path = 'women/'

    # Dimensions of output image
    w = 600
    h = 600

    # Read points for all images
    allPoints = create_points(path)
    
    # Read all images
    images = read_images(path)
    
    # Eye corners
    eyecornerDst = [(np.int(0.3 * w),
                     np.int(h / 3)),
                     (np.int(0.7 * w),
                     np.int(h / 3))]
    
    imagesNorm = []
    pointsNorm = []
    
    # Add boundary points for delaunay triangulation
    boundaryPts = np.array([(0, 0), (w/2, 0), (w-1, 0), (w-1, h/2), (w-1, h-1), (w/2, h-1), (0, h-1), (0,h/2)])
    
    # Initialize location of average points to 0s
    pointsAvg = np.array([(0,0)] * (len(allPoints[0]) + len(boundaryPts)), np.float32)
    
    n = len(allPoints[0])

    numImages = len(images)
    
    # Warp images and transform landmarks to output coordinate system,
    # and find average of transformed landmarks.
    
    for i in xrange(0, numImages):

        points1 = allPoints[i]

        # Corners of the eye in input image
        eyecornerSrc = [allPoints[i][36], allPoints[i][45]]
        
        # Compute similarity transform
        tform = similarity_transform(eyecornerSrc, eyecornerDst)
        
        # Apply similarity transformation
        img = cv2.warpAffine(images[i], tform, (w, h))

        # Apply similarity transform on points
        points2 = np.reshape(np.array(points1), (68, 1, 2))
        
        points = cv2.transform(points2, tform)
        
        points = np.float32(np.reshape(points, (68, 2)))
        
        # Append boundary points. Will be used in Delaunay Triangulation
        points = np.append(points, boundaryPts, axis=0)
        
        # Calculate location of average landmark points.
        pointsAvg = pointsAvg + points / numImages
        
        pointsNorm.append(points)
        imagesNorm.append(img)

    # Delaunay triangulation
    rect = (0, 0, w, h)
    dt = calculateDelaunayTriangles(rect, np.array(pointsAvg))

    # Output image
    output = np.zeros((h,w,3), np.float32)

    # Warp input images to average image landmarks
    for i in xrange(0, len(imagesNorm)):
        img = np.zeros((h, w, 3), np.float32)
        # Transform triangles one by one
        for j in xrange(0, len(dt)):
            tin = []
            tout = []
            
            for k in xrange(0, 3):
                pIn = pointsNorm[i][dt[j][k]]
                pIn = constrain_point(pIn, w, h)
                
                pOut = pointsAvg[dt[j][k]]
                pOut = constrain_point(pOut, w, h)
                
                tin.append(pIn)
                tout.append(pOut)

            warpTriangle(imagesNorm[i], img, tin, tout)

        # Add image intensities for averaging
        output = output + img

    # Divide by numImages to get average
    output = output / numImages

    # Display result
    cv2.imshow('image', output)
    cv2.waitKey(0)
