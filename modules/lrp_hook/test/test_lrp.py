import unittest
import torch 
import torch.nn as nn

class TestLRP(unittest.TestCase):

    def test_linear(self):
        from lrp import LrpModel
        a = torch.nn.Linear(10,20)
        b = torch.nn.Linear(20,30)

        lrp_model = LrpModel([a, nn.ReLU(), b], [{'epsilon':0.1}, None, {'gamma' : 0.2}])

        input = torch.rand(1, 10)
        result = lrp_model(input)
        result.sum().backward()

    def test_conv(self):
        from lrp import LrpModel
        a = torch.nn.Conv2d(3, 12, 3, 1)
        b = torch.nn.Conv2d(12, 6, 3, 1)

        lrp_model = LrpModel([a, nn.ReLU(), b], [{'epsilon':0.1}, None, {'gamma' : 0.2}])

        input = torch.rand(1, 3, 84, 84)
        result = lrp_model(input)
        result.sum().backward()
 
    def test_pool(self):
        from lrp import LrpModel
        a = torch.nn.Conv2d(3, 12, 3, 1)
        b = torch.nn.Conv2d(12, 6, 3, 1)

        lrp_model = LrpModel([a, nn.MaxPool2d(kernel_size=2), b, nn.AvgPool2d(kernel_size=2), nn.ReLU()], [{'epsilon':0.1}, None, {'gamma' : 0.2}, None, None])

        input = torch.rand(1, 3, 128, 128)
        result = lrp_model(input)
        result.sum().backward()
 
    def test_flataten(self):
        from lrp import LrpModel
        a = torch.nn.Conv2d(3, 12, 3, 1)
        b = torch.nn.Conv2d(12, 6, 3, 1)
        c = torch.nn.Linear(1014, 16)

        lrp_model = LrpModel(
                            [a, nn.MaxPool2d(kernel_size=2), b, nn.Flatten(1,-1), nn.ReLU()], 
                            [{'epsilon':0.1}, None, {'gamma' : 0.2}, None, None]
                    )

        input = torch.rand(1, 3, 32, 32)
        result = lrp_model(input)
        result.sum().backward()
 
    def test_kind_warning(self):
        from lrp import LrpModel
        a = torch.nn.Conv2d(3, 12, 3, 1)
        b = torch.nn.Conv2d(12, 6, 3, 1)
        c = torch.nn.Linear(1014, 16)

        lrp_model = LrpModel(
                            [a, nn.MaxPool2d(kernel_size=2), b, nn.Flatten(1,-1), c], 
                            [{'epsilon':0.1}, None, {'gamma' : 0.2}, None, {'gamma' : 0.2}]
                    )

        input = torch.rand(1, 3, 32, 32)
        result = lrp_model(input)
        result.sum().backward()

    def test_dropout(self):
        from lrp import LrpModel
        a = torch.nn.Conv2d(3, 12, 3, 1)
        b = torch.nn.Conv2d(12, 6, 3, 1)

        lrp_model = LrpModel(
                            [a, nn.Dropout(p=0.3), b, nn.Flatten(1,-1), nn.ReLU()], 
                            [{'epsilon':0.1}, None, {'gamma' : 0.2}, None, None]
                    )

        input = torch.rand(1, 3, 32, 32)
        result = lrp_model(input)
        result.sum().backward()


if __name__ == '__main__':
    unittest.main()
