import random
import unittest
import torch
from controltac.diffusion import Diffusion
from controltac.prepare import stratified, grouped, FORCE_QUOTAS, POSE_QUOTAS
from controltac.runtime import normalize, denormalize

class CoreTests(unittest.TestCase):
    def test_exact_sampling_and_no_silent_duplicates(self):
        rows = [dict(object='cross7',pose=str(p),image=f'{p}-{i}') for p in range(3) for i in range(4)]
        first = stratified(rows,10,random.Random(42),poses=3)
        self.assertEqual(len(first),10)
        self.assertEqual(len(grouped(first)),3)
        self.assertEqual(len({r['image'] for r in first}),10)
        self.assertEqual(first,stratified(rows,10,random.Random(42),poses=3))
        with self.assertRaises(ValueError):
            stratified(rows,13,random.Random(42))
        self.assertEqual(len(stratified(rows,13,random.Random(42),allow_repeat=True)),13)
        self.assertEqual(sum(FORCE_QUOTAS),20000)
        self.assertEqual(sum(POSE_QUOTAS),7000)

    def test_normalization_roundtrip_residual(self):
        x = torch.rand(3,256,320)
        stats = dict(min=[-.4,-.3,-.2], max=[.2,.3,.4])
        self.assertTrue(torch.allclose(denormalize(normalize(x,stats),stats),x-127/255,atol=1e-6))

    def test_ddim_includes_clean_terminal_step(self):
        diffusion = Diffusion(1000)
        expected = torch.full((1,32,8,10),0.25)
        calls = []
        def perfect_epsilon(inputs,t,force):
            calls.append(t.item())
            a = diffusion.alpha[t].view(-1,1,1,1)
            return (inputs[:,:32] - a.sqrt()*expected)/(1-a).sqrt()
        result = diffusion.sample(perfect_epsilon,torch.zeros_like(expected),torch.zeros(1,3),steps=7)
        self.assertEqual(len(calls),7)
        self.assertEqual(calls[-1],0)
        self.assertTrue(torch.allclose(result,expected,atol=1e-5))
        with self.assertRaises(ValueError):
            diffusion.sample(perfect_epsilon,expected,torch.zeros(1,3),steps=1001)

if __name__ == '__main__':
    unittest.main()
