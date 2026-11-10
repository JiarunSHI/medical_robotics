import unittest
import numpy as np
from PIL import Image
from compare_models import choose_cases,choose_frames,metrics,comparison_image

class CompareTests(unittest.TestCase):
    def test_case_sampling_preserves_split_and_is_repeatable(self):
        cases=[dict(case_id=i,split='test' if i<10 else 'train') for i in range(20)]
        a=choose_cases(cases,'test',8,123)
        self.assertEqual(a,choose_cases(cases,'test',8,123))
        self.assertEqual(len({r['case_id'] for r in a}),8)
        self.assertTrue(all(r['split']=='test' for r in a))
        with self.assertRaises(ValueError):
            choose_cases(cases,'test',8,123,[12])

    def test_positive_frames_only_and_repeatable(self):
        v=np.zeros((10,8,8),np.uint8);v[[1,3,5],2:4,2:4]=1
        frames=choose_frames(v,2,123,7,'positive')
        self.assertEqual(frames,choose_frames(v,2,123,7,'positive'))
        self.assertTrue(set(frames)<={1,3,5})

    def test_metrics_known_overlap(self):
        a=np.array([[1,1],[0,0]],bool);b=np.array([[1,0],[1,0]],bool)
        result=metrics(a,b)
        self.assertAlmostEqual(result['dice'],.5)
        self.assertAlmostEqual(result['iou'],1/3)

    def test_four_panel_original_unchanged(self):
        image=Image.new('RGB',(720,440),(80,80,80));mask=np.zeros((440,720),bool);mask[100:200,100:200]=1
        original=np.asarray(image).copy();result=metrics(mask,mask)
        canvas=comparison_image(image,mask,[mask,mask],[result,result],'Synthetic test')
        self.assertEqual(canvas.size,(2880,592))
        np.testing.assert_array_equal(np.asarray(canvas)[112:552,:720],original)
        np.testing.assert_array_equal(np.asarray(image),original)

if __name__=='__main__':
    unittest.main()
