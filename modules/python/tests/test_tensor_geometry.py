import unittest
import torch
from modules.python.tensor_geometry import PoincareBall

class TestTensorGeometry(unittest.TestCase):
    def setUp(self):
        self.ball = PoincareBall(c=1.0)

    def test_mobius_add(self):
        x = torch.tensor([0.1, 0.2])
        y = torch.tensor([0.3, 0.1])
        res = self.ball.mobius_add(x, y)
        self.assertEqual(res.shape, x.shape)
        # Deve estar dentro da bola unitária
        self.assertLess(torch.norm(res), 1.0)

    def test_distance(self):
        x = torch.tensor([0.0, 0.0])
        y = torch.tensor([0.5, 0.0])
        dist = self.ball.distance(x, y)
        self.assertGreater(dist, 0)

    def test_vectorization(self):
        x = torch.randn(10, 512) * 0.1
        y = torch.randn(10, 512) * 0.1
        res = self.ball.mobius_add(x, y)
        self.assertEqual(res.shape, (10, 512))

if __name__ == "__main__":
    unittest.main()
