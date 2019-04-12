# Face-Detection-with-AlexNet



there is a bug in inference code

change
>scale_img=cv2.resize(img,((int(img.shape[0]*scale)),(int(img.shape[1]*scale))))
to
>scale_img=cv2.resize(img,((int(img.shape[1]*scale)),(int(img.shape[0]*scale))))
to fix this bug

you can find more details in my blog below.

CSDN ：https://blog.csdn.net/Rrui7739/article/details/81261543
