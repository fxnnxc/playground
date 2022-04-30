
import torch 

# Grad-CAM: visual Explanations from Deep Networks via Gradient-based Localization 

def gradient_class_activation_map(last_convolution_features, class_weights):
    # Last Conv    : [ 1 x K x W x H]
    # Last Weight  : [ K ]
    assert last_convolution_features.ndim ==4 and class_weights.ndim == 1 
    assert last_convolution_features.size(1) == class_weights.size(0)     
    batch_size = last_convolution_features.size(0)

    A = torch.autograd.Variable(last_convolution_features.clone(), requires_grad = True)
    F_k = torch.nn.functional.adaptive_avg_pool2d(A, (1,1)).view(batch_size, -1)
    S_c  = torch.matmul(F_k, class_weights.data)    
    print(S_c)
    S_c.sum().backward() 

    Z = A.size(2) * A.size(3)
    a_k = A.grad.sum(axis=-1).sum(axis=-1).squeeze(0) / Z  # [1 x Filter ] 
    A_k = A.permute(0,2,3,1)                           # [ 1 x W x H x  Filter] 
    M_c  = torch.nn.functional.relu(torch.matmul(A_k, a_k))                    # [ 1 x W x H] 

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
    class_weights = torch.rand(K)

    cam = gradient_class_activation_map(last_conv_feature, class_weights)
    
    print("number of filters :", K)
    for k,v in cam.items():
        print(k, v.size())