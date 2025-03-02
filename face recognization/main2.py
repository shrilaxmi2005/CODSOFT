import cv2

# Load the pre-trained face detection model
face_cap = cv2.CascadeClassifier("c:/Users/hp/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
video_cap = cv2.VideoCapture(0)

while True:
    ret, video_data = video_cap.read()  # Capture each frame
    if not ret:
        print("Failed to capture video")
        break  # Break the loop if video capture fails

    # Convert the frame to grayscale for face detection
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    face = face_cap.detectMultiScale(col,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in face:
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the video feed
    cv2.imshow("video_live", video_data)

    # Check for keypress (10ms delay)
    key = cv2.waitKey(10)
    if key == ord("a"):  # If 'a' key is pressed
        print("The 'a' key was pressed!")

    # Exit the loop if 'q' is pressed
    if key == ord("q"):
        break

# Release the video capture and close all OpenCV windows
video_cap.release()
cv2.destroyAllWindows()