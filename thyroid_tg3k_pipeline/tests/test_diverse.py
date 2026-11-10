import unittest
import numpy as np
from PIL import Image
from compare_diverse import extract_features,choose_diverse
from compare_models import comparison_image

class DiverseTests(unittest.TestCase):
    def test_mask_geometry_and_unknown_features(self):
        a=np.tile(np.arange(200,dtype=np.uint8),(120,1))
        image=Image.fromarray(a)
        mask=np.zeros((120,200),bool);mask[30:60,40:100]=True
        features=extract_features(image,mask)
        self.assertAlmostEqual(features['area_fraction'],.075)
        self.assertAlmostEqual(features['aspect_ratio'],2.)
        unknown=extract_features(image,np.zeros_like(mask))
        self.assertNotIn('area_fraction',unknown)
        self.assertIn('image_texture',unknown)

    def test_unknown_render_has_no_metric_requirement(self):
        image=Image.new('RGB',(720,440),(70,70,70))
        empty=np.zeros((440,720),bool)
        canvas=comparison_image(image,empty,[empty,empty],[None,None],'Unknown frame')
        self.assertEqual(canvas.size,(2880,592))

    def test_selection_never_requires_model_predictions(self):
        rows=[]
        for cid in range(1,15):
            for frame in range(0,300,25):
                row=dict(case_id=cid,frame=frame,annotation_status='provided',empty_region='')
                for key in ('area_fraction','aspect_ratio','center_x','center_y','gland_brightness','gland_texture','contrast','boundary_gradient','fill_ratio','image_brightness','image_texture'):
                    row[key]=float(cid+frame/100)
                rows.append(row)
            for i,region in enumerate(('before','inside','after')):
                rows.append(dict(case_id=cid,frame=400+i*30,annotation_status='unknown',empty_region=region,image_brightness=float(cid*10),image_texture=float(cid)))
        result=choose_diverse(rows)
        self.assertEqual(len(result),28)
        self.assertEqual(len({(r['case_id'],r['frame']) for r in result}),28)
        self.assertEqual(sum(r['annotation_status']=='unknown' for r in result),8)
        self.assertEqual(len({r['case_id'] for r in result}),14)
        self.assertEqual(result,choose_diverse(rows))

if __name__=='__main__':
    unittest.main()
