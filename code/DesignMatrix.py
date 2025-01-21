import numpy as np
import xarray as xr


class DesignMatrix(object):
    def __init__(self, timestamps, ophys_frame_rate):
        '''
        A toeplitz-matrix builder for running regression with multiple temporal kernels. 

        Args
            timestamps: The actual timestamps for each time bin that will be used in the regression model. 
            ophys_frame_rate: the number of ophys timestamps per second
        '''

        # Add some kernels
        self.X = None
        self.kernel_dict = {}
        self.running_stop = 0
        self.features = {'timestamps': timestamps}
        self.ophys_frame_rate = ophys_frame_rate
        self.unstd_features = {}


    def make_labels(self, label, num_weights,offset, length): 
        base = [label] * num_weights 
        numbers = [str(x) for x in np.array(range(0,length+1))+offset]
        return [x[0] + '_'+ x[1] for x in zip(base, numbers)]


    def get_mask(self, kernels=None):
        ''' 
            Args:
            kernels, a list of kernel string names
            Returns:
            mask ( a boolean vector), where these kernels have support
        '''
        if len(kernels) == 0:
            X = self.get_X() 
        else:
            X = self.get_X(kernels=kernels) 
        mask = np.any(~(X==0), axis=1)
        return mask.values

    
    def trim_X(self, boolean_mask):
        for kernel in self.kernel_dict.keys():
            self.kernel_dict[kernel]['kernel'] = self.kernel_dict[kernel]['kernel'][:,boolean_mask] 
        for feature in self.features.keys():  
            self.features[feature] = self.features[feature][boolean_mask]

    
    def get_X(self, kernels=None):
        '''
        Get the design matrix. 

        Args:
            kernels (optional list of kernel string names): which kernels to include (for model selection)
        Returns:
            X (np.array): The design matrix
        '''
        if kernels is None:
            kernels = self.kernel_dict.keys()

        kernels_to_use = []
        param_labels = []
        for kernel_name in kernels:
            kernels_to_use.append(self.kernel_dict[kernel_name]['kernel'])
            param_labels.append(self.make_labels(   kernel_name, 
                                                    np.shape(self.kernel_dict[kernel_name]['kernel'])[0], 
                                                    self.kernel_dict[kernel_name]['offset_samples'],
                                                    self.kernel_dict[kernel_name]['kernel_length_samples'] ))

        X = np.vstack(kernels_to_use) 
        x_labels = np.hstack(param_labels)

        assert np.shape(X)[0] == np.shape(x_labels)[0], 'Weight Matrix must have the same length as the weight labels'

        X_array = xr.DataArray(
            X, 
            dims =('weights','timestamps'), 
            coords = {  'weights':x_labels, 
                        'timestamps':self.features['timestamps']}
            )
        return X_array.T


    def add_kernel(self, features, kernel_length, label, offset=0, num_weights=None):
        '''
        Add a temporal kernel. 

        Args:
            features (np.array): The timestamps of each feature that the kernel will align to. 
            kernel_length (int): length of the kernel (in SECONDS). 
            label (string): Name of the kernel. 
            offset (int) :offset relative to the features. Negative offsets cause the kernel
                          to overhang before the feature (in SECONDS)
        '''
    
        #Enforce unique labels
        if label in self.kernel_dict.keys():
            raise ValueError('Labels must be unique')

        self.features[label] = features

        # CONVERT kernel_length to kernel_length_samples
        if num_weights is None:
            if kernel_length == 0:
                kernel_length_samples = 1
            else:
                kernel_length_samples = int(np.ceil(self.ophys_frame_rate * kernel_length)) 
        else:
            # Some kernels are hard-coded by number of weights
            kernel_length_samples = num_weights

        # CONVERT offset to offset_samples
        offset_samples = int(np.floor(self.ophys_frame_rate*offset))

        this_kernel = []
        for i in range(kernel_length_samples):
            this_kernel.append(np.roll(features, offset_samples + i))
        this_kernel = np.stack(this_kernel, axis=0)
    
        self.kernel_dict[label] = {
            'kernel': this_kernel,
            'kernel_length_samples': kernel_length_samples,
            'offset_samples': offset_samples,
            'kernel_length_seconds': kernel_length,
            'offset_seconds': offset,
            'ind_start': self.running_stop,
            'ind_stop': self.running_stop+kernel_length_samples
            }
        self.running_stop += kernel_length_samples
 
        
    def add_unstd_features(self, features, label):
        '''
        Add unstandardized features to the design matrix. 

        Args:
            features (np.array): The features to add. 
            label (string): Name of the features. 
        '''

        self.unstd_features[label] = features