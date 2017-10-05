## Face Overlay

The idea of the algorithm is to create an averaged face from the uploaded photos.

## Algorithm's description

#### Step 1.

Read all the pictures that have the ".jpg" extension and are in specified folder.

Calculate the landmarks for each uploaded face in folder, then create a two-dimensional array. Use the library dlib and model shape_predictor_68_face_landmarks.dat to calculate landmarks.

#### Step 2.

Calculate the boundary points for the Delaunay triangulation for the averaged face image. Also, consider the points of the boundaries of the eyes.

#### Step 3.

Warp images and transform landmarks to output coordinate system, and find average of transformed landmarks.

#### Step 4.

Warp input images to average image landmarks via transform triangles one by one with Delaunay Triangulation.

#### Step 5. 

Display the result with average face.

## Results

### Woman

#### Input images array

| | | | | |
|:--:|:--:|:--:|:--:|:--:|
|<img width=200 src="/women/01_woman.jpg">| <img src="/women/02_woman.jpg" width="200"> | <img src="/women/03_woman.jpg" width="200"> | <img src="/women/04_woman.jpg" width="200">| <img src="/women/05_woman.jpg" width="200">|
|<img src="/women/06_woman.jpg" width="200">| <img src="/women/07_woman.jpg" width="200"> | <img src="/women/08_woman.jpg" width="200"> | <img src="/women/09_woman.jpg" width="200">| <img src="/women/10_woman.jpg" width="200">|

Can be overlayed to "average" face

#### Output result

<img width=400 src="/results/woman_beauty.png">

### Man

#### Input images array

| | | | | |
|:--:|:--:|:--:|:--:|:--:|
|<img width=200 src="/men/01_man.jpg">| <img src="/men/02_man.jpg" width="200"> | <img src="/men/03_man.jpg" width="200"> | <img src="/men/04_man.jpg" width="200">| <img src="/men/05_man.jpg" width="200">|
|<img src="/men/06_man.jpg" width="200">| <img src="/men/07_man.jpg" width="200"> | <img src="/men/08_man.jpg" width="200"> | <img src="/men/09_man.jpg" width="200">| <img src="/men/10_man.jpg" width="200">|

Can be overlayed to "average" face

#### Output result

<img width=400 src="/results/man_beauty.png">
