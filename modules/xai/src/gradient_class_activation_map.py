
import torch 

# Grad-CAM: visual Explanations from Deep Networks via Gradient-based Localization 

def gradient_class_activation_map(last_convolution_features, last_weights):
    
    assert last_convolution_features.ndim ==4 and last_weights.ndim == 2 
    assert last_convolution_features.size(1) == last_weights.size(1) 
    
    last_convolution_features = torch.autograd.Variable(last_convolution_features, requires_grad = True)

    batch_size = last_convolution_features.size(0)
    K = last_weights.size(1)
    F_k = torch.nn.functional.adaptive_avg_pool2d(last_convolution_features, (1,1)).view(batch_size, -1)
    S_c  = torch.matmul(F_k, last_weights.t())    
    S_c.sum().backward() 
    M_c  = torch.nn.functional.relu(last_convolution_features.grad)

    result = {
        "global_average_pooling" : F_k,    # (1 x F_k)  global average
        "class_score" : S_c,               # (1 x 1)    total score
        "class_activation_map" : M_c       # (W x H)    position wise score
    }

    return result


if __name__ == "__main__":
    K = 32
    width, height = (64, 64)
    last_conv_feature = torch.rand(1, K, width, height)
    last_weights = torch.rand(1, K)

    cam = gradient_class_activation_map(last_conv_feature, last_weights)
    
    print("number of filters :", K)
    for k,v in cam.items():
        print(k, v.size())