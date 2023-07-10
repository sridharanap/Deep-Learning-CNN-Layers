import numpy as np

class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape= stride_shape
        self.pooling_shape= pooling_shape
        self.trainable = False
        self.input_tensor= None
        self.maximum_pos = None

    def forward(self,input_tensor):
        self.input_tensor=input_tensor
        batch_size, input_channel_size, inputspat_x, inputspat_y = input_tensor.shape
        stride_x, stride_y = self.stride_shape
        output_shape=(batch_size, input_channel_size , (inputspat_x-self.pooling_shape[0]) // stride_x+1,(inputspat_y-self.pooling_shape[1]) // stride_y+1)
        output_tensor=np.zeros(output_shape)
        self.maximum_pos=[]
        for b in range(output_shape[0]):
            for c in range(output_shape[1]):
                for i in range(output_shape[2]):
                    for j in range(output_shape[3]):
                        #maximum value estimation
                        output_tensor[b,c,i,j]=np.amax(input_tensor[b, c, i * stride_x:i * stride_x + self.pooling_shape[0],j * stride_y:j * stride_y + self.pooling_shape[1]])
                        #maximum value index estimation
                        maximum_index = np.argwhere(input_tensor[b, c, i * stride_x:i * stride_x + self.pooling_shape[0],j * stride_y:j * stride_y + self.pooling_shape[1]] == output_tensor[b, c, i, j]) + [i * stride_x,j * stride_y]
                        maximum_index=np.array([b,c,maximum_index[0][0],maximum_index[0][1]])
                        self.maximum_pos.append(maximum_index)
        return output_tensor

    def backward(self,error_tensor):
        output_error_tensor=np.zeros(self.input_tensor.shape)
        output_shape=output_error_tensor.shape
        error_tensor_shape=error_tensor.shape
        index=0
        for b in range(output_shape[0]):
            for c in range(output_shape[1]):
                for i in range(0,error_tensor_shape[2]):
                    for j in range(0,error_tensor_shape[3]):
                        #adding error values to the corner positions
                        output_error_tensor[b, c, self.maximum_pos[index][2], self.maximum_pos[index][3]] += error_tensor[b, c, i, j]
                        index+=1
        return output_error_tensor