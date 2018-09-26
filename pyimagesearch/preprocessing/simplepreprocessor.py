import cv2

class SimplePreproce
ssor:
    """
    width: The target width of our input image after resizing.
    height: The target height of our input image after resizing.
    inter: An optional parameter used to control which interpolation algorithm is used when
resizing.
    """
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter
        
    def preprocess(self, image):
        # resize the image to a fixed size, ignoring the aspect ratio
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)