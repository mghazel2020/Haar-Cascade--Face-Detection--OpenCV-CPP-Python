
# Face and Eyes Detection using Haar Cascade Algorithms in OpenCV with C++ and Python APIs

<table>
  <tr>
    <td> <img src="figures/face-detection-haar-cascades.webp" width="1000"  ></td>
  </tr>
</table>

## 1. Objective

The objective of his task is to demonstrate the application of trained Haar Cascades face and eyes detector in OpenCV-Python.

## 2. Haar Cascades Face Detector

Face Detection, a widely popular subject with a huge range of applications. Modern day Smartphones and Laptops come with in-built face detection software, which can authenticate the identity of the user. There are numerous apps that can capture, detect and process a face in real time, can identify the age and the gender of the user, and also can apply some really cool filters. The list is not limited to these mobile apps, as Face Detection also has a wide range of applications in Surveillance, Security and Biometrics as well

Viola and Jones proposed popularly,  Haar Cascades face detector, the first ever object detection framework for real time face detection in video footage. Additional information about the Haar Cascades face detector can be found on the references section.

OpenCV comes with a pre-trained Haar Cascades for detecting human faces and eyes. In this project, we shall deploy these detectors to detect human faces and eyes from various types types of test images as well as live video stream. 

## 3. Data

We shall deploy the  Haar Cascades face detector  to detect human faces and eyes from various still test images as well as live video stream. 

## 4. Development

* Project: Face Detection:
* The objective of this project is to implement and demonstrate the Haar Cascades face detector, as implemented in OpenCV - Python:
  * We shall illustrate its performance on individual images as well as live video stream.
  * Author: Mohsen Ghazel (mghazel)
  * Date: April 15th, 2021

### 4.1. Step 1: Imports and global variables
#### 4.1.1. Python import:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Python imports and environment setup</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># opencv</span>
<span style="color:#200080; font-weight:bold; ">import</span> cv2
<span style="color:#595979; "># numpy</span>
<span style="color:#200080; font-weight:bold; ">import</span> numpy <span style="color:#200080; font-weight:bold; ">as</span> np
<span style="color:#595979; "># matplotlib</span>
<span style="color:#200080; font-weight:bold; ">import</span> matplotlib<span style="color:#308080; ">.</span>pyplot <span style="color:#200080; font-weight:bold; ">as</span> plt
<span style="color:#200080; font-weight:bold; ">import</span> matplotlib<span style="color:#308080; ">.</span>image <span style="color:#200080; font-weight:bold; ">as</span> mpimg

<span style="color:#595979; "># input/output OS</span>
<span style="color:#200080; font-weight:bold; ">import</span> os 

<span style="color:#595979; "># date-time to show date and time</span>
<span style="color:#200080; font-weight:bold; ">import</span> datetime

<span style="color:#595979; "># to display the figures in the notebook</span>
<span style="color:#44aadd; ">%</span>matplotlib inline

<span style="color:#595979; ">#------------------------------------------</span>
<span style="color:#595979; "># Test imports and display package versions</span>
<span style="color:#595979; ">#------------------------------------------</span>
<span style="color:#595979; "># Testing the OpenCV version</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"OpenCV : "</span><span style="color:#308080; ">,</span>cv2<span style="color:#308080; ">.</span>__version__<span style="color:#308080; ">)</span>
<span style="color:#595979; "># Testing the numpy version</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Numpy : "</span><span style="color:#308080; ">,</span>np<span style="color:#308080; ">.</span>__version__<span style="color:#308080; ">)</span>

OpenCV <span style="color:#308080; ">:</span>  <span style="color:#008000; ">4.5</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">1</span>
Numpy <span style="color:#308080; ">:</span>  <span style="color:#008000; ">1.19</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">2</span>
</pre>

### 4.2. Step 2: Input data:

* Setup the input data and configuration files paths.

#### 4.2.1. Setup the folder containing the input images:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#----------------------------------------------------</span>
<span style="color:#595979; "># Setup the folder comtaining the test images:</span>
<span style="color:#595979; ">#----------------------------------------------------</span>
test_images_folder <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">"../data/test-images/"</span>
</pre>

#### 4.2.2. Setup the folder containing the Haar Cascades configuration files:

* OpenCV comes with a pre-trained Haar Cascades with configuration xml files.


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#----------------------------------------------------</span>
<span style="color:#595979; "># Setup the folder comtaining the pre-trained Haar </span>
<span style="color:#595979; "># Cascades detector configuration XML files:</span>
<span style="color:#595979; ">#----------------------------------------------------</span>
haar_cascades_configs_folder <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">"../Human-Face/resources/DATA/haarcascades/"</span>
</pre>


#### 4.2.3. Setup the Haar Cascades face detector:

* Instantiate the the pre-trained Haar Cascades face detector:

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;">face_cascade <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>CascadeClassifier<span style="color:#308080; ">(</span>haar_cascades_configs_folder <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">'haarcascade_frontalface_default.xml'</span><span style="color:#308080; ">)</span>
</pre>

#### 4.2.4. Setup the Haar Cascades eye detector:

* Instantiate the pre-trained Haar Cascades eye detector:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;">eye_cascade <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>CascadeClassifier<span style="color:#308080; ">(</span>haar_cascades_configs_folder <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">'haarcascade_eye.xml'</span><span style="color:#308080; ">)</span>
</pre>


### 4.3. Step 3: Detect faces and eyes from images:

* We are now ready to run the Haar Cascades to detect human faces and eyes from images.

#### 4.3.1. Define the Haar Cascades face detector utility function:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">"""Use Haar Cascades factor detector to detect faces:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;# Argument:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;img: input image</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;# Returns:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;None.</span>
<span style="color:#595979; ">"""</span>
<span style="color:#200080; font-weight:bold; ">def</span> detect_face<span style="color:#308080; ">(</span>img<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
  
    face_img <span style="color:#308080; ">=</span> img<span style="color:#308080; ">.</span>copy<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
  
    face_rects <span style="color:#308080; ">=</span> face_cascade<span style="color:#308080; ">.</span>detectMultiScale<span style="color:#308080; ">(</span>face_img<span style="color:#308080; ">)</span> 
    
    <span style="color:#200080; font-weight:bold; ">for</span> <span style="color:#308080; ">(</span>x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">)</span> <span style="color:#200080; font-weight:bold; ">in</span> face_rects<span style="color:#308080; ">:</span> 
        <span style="color:#595979; "># cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10) </span>
        cv2<span style="color:#308080; ">.</span>rectangle<span style="color:#308080; ">(</span>face_img<span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span>x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span>x<span style="color:#44aadd; ">+</span>w<span style="color:#308080; ">,</span>y<span style="color:#44aadd; ">+</span>h<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">255</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#308080; ">)</span> 
        
    <span style="color:#200080; font-weight:bold; ">return</span> face_img
</pre>

#### 4.3.2. Define the Haar Cascades eyes detector utility function:

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">"""Use Haar Cascades factor detector to detect eyes:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;# Argument:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;img: input image</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;# Returns:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;None.</span>
<span style="color:#595979; ">"""</span>
<span style="color:#200080; font-weight:bold; ">def</span> detect_eyes<span style="color:#308080; ">(</span>img<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    
    face_img <span style="color:#308080; ">=</span> img<span style="color:#308080; ">.</span>copy<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
  
    eyes <span style="color:#308080; ">=</span> eye_cascade<span style="color:#308080; ">.</span>detectMultiScale<span style="color:#308080; ">(</span>face_img<span style="color:#308080; ">)</span> 
    
    
    <span style="color:#200080; font-weight:bold; ">for</span> <span style="color:#308080; ">(</span>x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">)</span> <span style="color:#200080; font-weight:bold; ">in</span> eyes<span style="color:#308080; ">:</span> 
        <span style="color:#595979; "># cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10) </span>
        cv2<span style="color:#308080; ">.</span>rectangle<span style="color:#308080; ">(</span>face_img<span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span>x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span>x<span style="color:#44aadd; ">+</span>w<span style="color:#308080; ">,</span>y<span style="color:#44aadd; ">+</span>h<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">255</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#308080; ">)</span>
        
    <span style="color:#200080; font-weight:bold; ">return</span> face_img
</pre>

#### 4.3.3. Process the test images to detect faces and eyes:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># 3.3.1) Itetate over all the images in the test images </span>
<span style="color:#595979; ">#      folder:</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#200080; font-weight:bold; ">for</span> filename <span style="color:#200080; font-weight:bold; ">in</span> os<span style="color:#308080; ">.</span>listdir<span style="color:#308080; ">(</span>test_images_folder<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">#------------------------------------------------------</span>
    <span style="color:#595979; "># 3.3.2) read the test image</span>
    <span style="color:#595979; ">#------------------------------------------------------</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Processing image: '</span> <span style="color:#44aadd; ">+</span> os<span style="color:#308080; ">.</span>path<span style="color:#308080; ">.</span>join<span style="color:#308080; ">(</span>test_images_folder<span style="color:#308080; ">,</span>filename<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    img <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>imread<span style="color:#308080; ">(</span>os<span style="color:#308080; ">.</span>path<span style="color:#308080; ">.</span>join<span style="color:#308080; ">(</span>test_images_folder<span style="color:#308080; ">,</span>filename<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    <span style="color:#595979; "># process the valid images</span>
    <span style="color:#200080; font-weight:bold; ">if</span> img <span style="color:#200080; font-weight:bold; ">is</span> <span style="color:#200080; font-weight:bold; ">not</span> <span style="color:#074726; ">None</span><span style="color:#308080; ">:</span>
        <span style="color:#595979; ">#------------------------------------------------------</span>
        <span style="color:#595979; "># 3.3.3) deploy the Haar Cascades to detect faces</span>
        <span style="color:#595979; ">#------------------------------------------------------</span>
        faces <span style="color:#308080; ">=</span> detect_face<span style="color:#308080; ">(</span>img<span style="color:#308080; ">)</span>
        <span style="color:#595979; ">#------------------------------------------------------</span>
        <span style="color:#595979; "># 3.3.4) deploy the Haar Cascades to detect eyes</span>
        <span style="color:#595979; ">#------------------------------------------------------</span>
        faces_eyes <span style="color:#308080; ">=</span> detect_eyes<span style="color:#308080; ">(</span>faces<span style="color:#308080; ">)</span>
        <span style="color:#595979; "># create a figure</span>
        plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">8</span><span style="color:#308080; ">,</span> np<span style="color:#308080; ">.</span>uint8<span style="color:#308080; ">(</span><span style="color:#008c00; ">8</span> <span style="color:#44aadd; ">*</span> img<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#44aadd; ">/</span>img<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
        <span style="color:#595979; "># visualize detection results</span>
        plt<span style="color:#308080; ">.</span>subplot<span style="color:#308080; ">(</span><span style="color:#008c00; ">111</span><span style="color:#308080; ">)</span>
        plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Haar Cascades: Faces and Eyes detection"</span><span style="color:#308080; ">,</span> fontsize<span style="color:#308080; ">=</span><span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
        plt<span style="color:#308080; ">.</span>xticks<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> plt<span style="color:#308080; ">.</span>yticks<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
        plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>cv2<span style="color:#308080; ">.</span>cvtColor<span style="color:#308080; ">(</span>faces_eyes<span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>COLOR_BGR2RGB<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
</pre>

### 4.4. Step 4: Face and eyes detection from video:

* We can also run the Haar Cascades to detect human faces and eyes from video frames:
  * This is done independently for each frame in a straightforward way as follows:
    * grab each video frame and treat it as an image
    * run the Haar Cascades to detect human faces and eyes from the frame, as done above.
  * This can be done for live video or from a video file
  * Next, we illustrate face and eyes detection from live video.
  

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># 4.1) Detect faces and eyes detection from live video</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># start the camera capture</span>
cap <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>VideoCapture<span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span> 

<span style="color:#595979; "># iterate over the camera captured frames</span>
<span style="color:#200080; font-weight:bold; ">while</span> <span style="color:#074726; ">True</span><span style="color:#308080; ">:</span> 
    <span style="color:#595979; ">#------------------------------------------------------</span>
    <span style="color:#595979; "># Step 1: grab the next frame</span>
    <span style="color:#595979; ">#------------------------------------------------------</span>
    ret<span style="color:#308080; ">,</span> frame <span style="color:#308080; ">=</span> cap<span style="color:#308080; ">.</span>read<span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span> 
    <span style="color:#595979; ">#------------------------------------------------------</span>
    <span style="color:#595979; "># Step 2: deploy the Haar Cascades to detect faces</span>
    <span style="color:#595979; ">#------------------------------------------------------</span>
    frame <span style="color:#308080; ">=</span> detect_face<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">)</span>
    <span style="color:#595979; ">#------------------------------------------------------</span>
    <span style="color:#595979; "># Step 3: deploy the Haar Cascades to detect eyes</span>
    <span style="color:#595979; ">#------------------------------------------------------</span>
    frame <span style="color:#308080; ">=</span> detect_eyes<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">)</span>
    <span style="color:#595979; ">#------------------------------------------------------</span>
    <span style="color:#595979; "># Step 4: display the frame with overlaid face and eyes </span>
    <span style="color:#595979; ">#         detections.</span>
    <span style="color:#595979; ">#------------------------------------------------------</span>
    cv2<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Video Face &amp; Eyes Detection'</span><span style="color:#308080; ">,</span> frame<span style="color:#308080; ">)</span> 
    
    <span style="color:#595979; ">#------------------------------------------------------</span>
    <span style="color:#595979; "># Quit if user presses: ESC key</span>
    <span style="color:#595979; ">#------------------------------------------------------</span>
    c <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>waitKey<span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span> 
    <span style="color:#200080; font-weight:bold; ">if</span> c <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">27</span><span style="color:#308080; ">:</span> 
        <span style="color:#200080; font-weight:bold; ">break</span> 

<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Close the camera feed </span>
<span style="color:#595979; ">#------------------------------------------------------</span>
cap<span style="color:#308080; ">.</span>release<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span> 
<span style="color:#595979; "># close all windows.</span>
cv2<span style="color:#308080; ">.</span>destroyAllWindows<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

### 4.5. Step 5: Display a successful execution message:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># display a final message</span>
<span style="color:#595979; "># current time</span>
now <span style="color:#308080; ">=</span> datetime<span style="color:#308080; ">.</span>datetime<span style="color:#308080; ">.</span>now<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># display a message</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Program executed successfully on: '</span><span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>now<span style="color:#308080; ">.</span>strftime<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"%Y-%m-%d %H:%M:%S"</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">"...Goodbye!</span><span style="color:#0f69ff; ">\n</span><span style="color:#1060b6; ">"</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

Program executed successfully on<span style="color:#308080; ">:</span> <span style="color:#008c00; ">2021</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">04</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">15</span> <span style="color:#008c00; ">10</span><span style="color:#308080; ">:</span><span style="color:#008c00; ">36</span><span style="color:#308080; ">:</span><span style="color:#008000; ">19.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span>Goodbye!
</pre>

## 5. Sample Face and Eyes Detection Results

In this section, we illustrate sample face and eyes detection results from still imagery and live video frames.

### 5.1. Detection from Images

<table>
  <tr>
    <td> <img src="figures/Still-Images-Face-Eyes-Detections.png" width="800" ></td>
  </tr>
</table>

### 5.2 Detection from Live Video Stream

<table>
  <tr>
    <td> <img src="figures/Video-Frames-Face-Eyes-Detections.png" width="800" ></td>
  </tr>
</table>

## 5. Analysis

In view of the presented results, we make the following observations:

* In spite of its speeds and ability to detect clearly visible faces, the Haar Cascades face and eyes detectors are not without limitations:
* Some of the erroneous detections are puzzling and difficult to explain:
* Too many false-positives for some of the images.
* Some of the missed detections for clearly visible faces are just puzzling
* The eye detector appear much worse than the face detector, especially when using live camera streaming video.
* The face detector appear very sensitive to:
* The face pose: It is not able to detect most of the turned faces
* The face scale: It is not able to detect most of the far and small faces
* Skin tone: It is able to able to detect some of the faces with darker skin tones.
* Obstruction: It is not able to detect some of the partially obscured faces.

## 6. Future Work

We plan to investigate some of the related issues:

* Get a better understanding of the details involved in implementing the Haar Cascades detectors
* Test these detector with more diverse data and identity their limitations
* Search for and explore other pre-trained Haarr Cascades face and eyes detectors

## 7. References

1. Paul Viola, Michael Jones. Rapid Object Detection using a Boosted Cascade of Simple Features. Retrieved from: https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf.
2. OpenCV. Face Detection using Haar Cascades. Retrieved from: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html.
3. Girija Shankar Behera. Face Detection with Haar Cascade. Retrieved from: https://towardsdatascience.com/face-detection-with-haar-cascade-727f68dafd08.
4. Adarsh Menon. Face Detection in 2 Minutes using OpenCV & Python. Retrieved from: https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81.
5. Parul Pandey. Face Detection with Python using OpenCV. Retrieved from: https://www.datacamp.com/community/tutorials/face-detection-python-opencv 
6. Stack Abuse. Facial Detection in Python with OpenCV. Retrieved from: https://stackabuse.com/facial-detection-in-python-with-opencv/.
7. Hussain Mujtaba. Face Recognition with Python and OpenCV. Retrieved from: https://www.mygreatlearning.com/blog/face-recognition/.
8. GeekforGeeks. OpenCV C++ Program for Face Detection. Retrieved from: https://www.geeksforgeeks.org/opencv-c-program-face-detection/.
9. Valeriia Koriukina. Using Facial Landmarks for Overlaying Faces with Masks. Retrieved from: https://learnopencv.com/tag/face-detection/.
10. Vikas Gupta. Face Detection - OpenCV, Dlib and Deep Learning ( C++ / Python ). Retrieved from: https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/.
11. Min Sun. OpenCV - Face Detection. Retrieved from: https://www.cs.princeton.edu/courses/archive/fall08/cos429/CourseMaterials/Precept1/facedetect.pdf.
12. BeWagner. Building a face detector with OpenCV in C++. Retrieved from: https://bewagner.net/programming/2020/04/12/building-a-face-detector-with-opencv-in-cpp/.
