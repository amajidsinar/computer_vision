import h5py
import os

class HDF5DatasetWriter:
    """
    This class takes 4 parameters:
    dims = 2-dimensional shapes which have the shape of (n,flattened_features) (REQUIRED)
    outputPath = the output path (REQUIRED)
    data = the name of the input data (e.g. images, features)
    buffSize = number of buffers before being flushed to disk
    """
    def __init__(self, dims, outputPath, data="images", buffSize=1000):
        if os.path.exists(outputPath):
            raise ValueError("The supplied output path already " 
                             "exists")
            
        # create databases
        self.db = h5py.File(outputPath, "w")
        # predictor is the input database
        self.predictor = self.db.create_dataset(data, dims, dtype="float")
        # response is the output database
        self.response = self.db.create_dataset("label", (dims[0],), dtype="int")
        
        self.buffSize = buffSize
        self.buffer = {"X":[], "y":[]}
        self.idx = 0
        
    def accumulate(self, feature, target):
        """
        fill the buffer
        """
        self.buffer["X"].extend(feature)
        self.buffer["y"].extend(target)
        # flush when buffer has reached its limit
        if len(self.buffer["X"]) >= self.buffSize:
            self.flush()
            
    def flush(self):
        """
        flush the buffer
        """
        
        # i is the current index that is available for filling the database
        i = self.idx + len(self.buffer["X"]) 
        
        # fill the database
        self.predictor[self.idx:i] = self.buffer["X"]
        self.response[self.idx:i] = self.buffer["y"]
        
        # reset index
        self.idx = i
        # reset buffer
        self.buffer = {"X":[], "y":[]}
    
    def close(self):
        if len(self.buffer["X"]) > 0:
            self.flush()
            
        self.db.close()
        
    
        

        
        
        