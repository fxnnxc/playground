
import torch 

# Learning Deep Feature for Discriminative Localization

def class_activation_map(last_convolution_features, class_weights):
    # Last Conv    : [ 1 x K x W x H]
    # Last Weight  : [ K ]
    assert last_convolution_features.ndim ==4 and class_weights.ndim == 1 
    assert last_convolution_features.size(1) == class_weights.size(0) 
    
    batch_size = last_convolution_features.size(0)
    F_k = torch.nn.functional.adaptive_avg_pool2d(last_convolution_features, (1,1)).view(batch_size, -1)    # [1 x K]
    M_c  = torch.matmul(last_convolution_features.permute(0,2,3,1), class_weights.t())                      # [1 x W x H]

    result = {
        "global_average_pooling" : F_k,    # (1 x F_k)  global average
        "class_activation_map" : M_c       # (W x H)    position wise score
    }
    return result


if __name__ == "__main__":
    K = 32
    width, height = (64, 64)
    last_conv_feature = torch.rand(1, K, width, height)
    class_weights = torch.rand(K)

    cam = class_activation_map(last_conv_feature, class_weights)
    
    print("number of filters :", K)
    for k,v in cam.items(): 
        print(k, v.size())
