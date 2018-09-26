import imutils
import cv2

class AspectAwarePreprocessor:
    """
    determine the shortest dimension and resize along it
    crop the image along the largest dimension to obtain the target width and height
    
    """
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation
        # method used when resizing
        
        self.width = width
        self.height = height
        self.inter = inter
        
    def preprocess(self, image):
        
        
        # grab the width and height of the input image
        (h, w) = image.shape[:2]
        # determine the delta offsets
        dW = 0
        dH = 0
        
        # if the width is smaller than the height, then resize along the smaller 
        # dimension (in this case, width) and then update the deltas
        # to crop the height to the desired dimension
        
        if w < h:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)
            
        else:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)
        
        # now that our images have been resized, we need to re-grab the width
        # and height, followed by performing the crop
        (h, w) = image.shape[:2]
        image = image[dH:h - dH, dW:w - dW]
        
        # finally, resize the image to the provided spatial dimensions to ensure
        # our output image is always a fixed size
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
        