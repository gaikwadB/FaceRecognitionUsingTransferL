# FaceRecognitionUsingTransferL
This project is a face recognition application using transfer learning

## What is transfer learning?

According to Wikipedia, Transfer learning is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. For example, the knowledge gained while learning to recognize cars could apply when trying to recognize trucks.
dataset:![dataset](/Screenshot/0.png)

## Why use Transfer Learning?

In today's digital world there is a huge amount of data, commonly known as Big Data.

In the data science world, we face a lot of situation which require a lot of computational power or resources like RAM, CPU or GPU, etc. to train our models. To cope up with this situation one of the best ways is to use the pre-trained model as in Transfer Learning. This will also help to achieve more accuracy with less amount of data.

## What is VGG16?

No alt text provided for this image
VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. The model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes. It was one of the famous models submitted to ILSVRC-2014. It makes the improvement over AlexNet by replacing large kernel-sized filters (11 and 5 in the first and second convolutional layer, respectively) with multiple 3×3 kernel-sized filters one after another. VGG16 was trained for weeks and was using NVIDIA Titan Black GPUs.

## Face Detection and Face Recognition

Face Detection: It is an (AI) based computer technology used to find and identify human faces in digital images or videos. It now plays an important role as the first step in many key applications including face tracking, face analysis, and facial recognition.

## Our Task: To create a Face Recognition model using a pre-trained Deep Learning model VGG16.

Let's break our task into sub-tasks:

1.Generation of data using Open CV for face extraction for the training part of the model.

2. After Extraction of the faces divide our whole data generated in the form of images into two parts 1.) Training Part 2.) Testing Part

3. In the next step, we will use a pre-trained Deep Learning Model called VGG16 to recognize the faces.

## Prerequisites:

List of Python libraries need to be installed:

    tensorflow
    keras
    opencv
    pillow
    numpy
    
## for this you can run this command in your command prompt:

    conda install tensorflow keras opencv-python pillow numpy
    
## Creating a dataset
    import cv2
    import numpy as np

    # Load HAAR face classifier
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # Load functions
        def face_extractor(img):
            # Function detects faces and returns the cropped face
            # If no face detected, it returns the input image

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            if faces is ():
                return None

            # Crop all faces found
            for (x,y,w,h) in faces:
                cropped_face = img[y:y+h, x:x+w]

            return cropped_face

        # Initialize Webcam
        cap = cv2.VideoCapture(0)
        count = 0

        # Collect 100 samples of your face from webcam input
        while True:

            ret, frame = cap.read()
            if face_extractor(frame) is not None:
                count += 1
                face = cv2.resize(face_extractor(frame), (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                # Save file in specified directory with unique name
                file_name_path = 'C://Users//Bhayyasaheb//mlops_work//Face Rec//bhayyasaheb//image' + str(count) + '.jpg'
                cv2.imwrite(file_name_path, face)

                # Put count on images and display live count
                cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                cv2.imshow('Face Cropper', face)

            else:
                print("Face not found")
                pass

            if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
                break

        cap.release()
        cv2.destroyAllWindows()      
        print("Collecting Samples Complete")


## Loading the VGG16 Model
