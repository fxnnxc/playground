import unittest 
import torch 
import torch.nn as nn
from lrp_perfect import LRP

class TestSuit(unittest.TestCase):

    def test_model(self):

        layer1 = nn.Linear(30, 40)
        relu = nn.ReLU()
        layer2 = nn.Linear(40, 10)

        layers = [
            layer1,
            relu,
            layer2
        ]

        rule_descriptions = [
            {"gamma" : 0.2, "epsilon" : 0.1},
            {},
            {"epsilon" : 0.1},
        ] 

        model = LRP(layers, rule_descriptions, device="cpu")
        x = torch.rand(1, 30)
        R, A = model.forward(x)
        print(R)
        print(A)
        for r in R:
            print(r.size())
        print()
        for a in A:
            print(a.size())

