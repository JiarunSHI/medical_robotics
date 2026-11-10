import unittest
import numpy as np
from PIL import Image
from src.cs_model import prepare_cs
from compare_cs import render_cs

class CSTests(unittest.TestCase):
    def test_original_rgb_normalization_and_padding(self):
        im=Image.new('RGB',(720,440),(255,128,64))
        x,geometry=prepare_cs(im)
        self.assertEqual(tuple(x.shape),(3,256,256))
        np.testing.assert_allclose(x[:,128,128].numpy(),[1,128/255,64/255],rtol=1e-6)
        np.testing.assert_array_equal(x[:,0,0].numpy(),[0,0,0])
        self.assertEqual(geometry,(0,50,256,156,720,440))

    def test_unknown_cs_panel(self):
        image=Image.new('RGB',(720,440),(70,70,70));mask=np.zeros((440,720),bool)
        result=render_cs(image,mask,mask,None,'CS unknown')
        self.assertEqual(result.size,(2160,592))
        np.testing.assert_array_equal(np.asarray(result)[112:552,:720],np.asarray(image))

if __name__=='__main__':
    unittest.main()
