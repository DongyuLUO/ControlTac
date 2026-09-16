import unittest
import json
from pathlib import Path
import torch
import numpy as np
from controltac.diffusion import Diffusion
from controltac.data import read_rows
from controltac.runtime import normalize, denormalize

class CoreTests(unittest.TestCase):
    def test_examples_force_range_and_pose_conditions(self):
        examples = Path(__file__).resolve().parents[1]/'examples'
        for stage in ['force', 'force_pose']:
            config = json.loads((examples/f'{stage}.json').read_text())
            for key, source in [('initial_force','reference'),('target_force','target')]:
                self.assertTrue(-10 <= config[key][2] <= -1)
            self.assertNotEqual(config['initial_force'], config['target_force'])
            masks = [np.load(examples/f'assets/{stage}/{name}_mask.npy',allow_pickle=False) for name in ['reference','target']]
            self.assertEqual(np.array_equal(*masks), stage == 'force')
            if stage == 'force_pose':
                self.assertEqual(config['mask'], 'assets/force_pose/target_mask.npy')
            else:
                self.assertNotIn('mask', config)

    def test_distributed_manifests_and_normalization(self):
        root = Path(__file__).resolve().parents[1]
        expected = json.loads((root/'controltac/normalization.json').read_text())
        for name in ['examples/normalization.json','splits/normalization.json']:
            self.assertEqual(expected, json.loads((root/name).read_text()))
        force = read_rows(root/'splits/force_train.csv')
        pose = read_rows(root/'splits/force_pose_train.csv')
        self.assertEqual(len(force), 20000)
        self.assertEqual(len(pose), 7000)
        self.assertEqual(len({row['image'] for row in pose}), 7000)
        self.assertEqual(set(force[0]), {'object','image','force','reference_image'})
        self.assertEqual(set(pose[0]), {'object','image','force','mask'})
        image_objects = {row['image']:row['object'] for row in force}
        for row in force:
            self.assertEqual(image_objects[row['reference_image']],row['object'])

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
