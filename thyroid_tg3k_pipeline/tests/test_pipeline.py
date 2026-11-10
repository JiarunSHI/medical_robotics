import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from src.predict import reconcile
from src.data import letterbox,restore
from src.model import MobileUNet
from src.nrrd_io import read_volume,write_volume,check_header
from src.common import ROOT,read_json,read_csv
from train import loss_fn,select_rows

class PipelineTests(unittest.TestCase):
    def test_keep_entire_original_frame_even_when_prediction_disagrees(self):
        original=np.zeros((17,23),np.uint8);original[3:7,4:9]=7
        result,status,_=reconcile(original,np.ones_like(original,dtype=float),7)
        np.testing.assert_array_equal(result,original)
        self.assertEqual(status,'provided')
        result,_,_=reconcile(original,np.zeros_like(original,dtype=float),7)
        np.testing.assert_array_equal(result,original)

    def test_only_empty_frame_gets_pseudo(self):
        original=np.zeros((7,9),np.uint8)
        prob=np.zeros((7,9));prob[2:4,3:6]=.8
        result,status,_=reconcile(original,prob,7)
        self.assertEqual(status,'pseudo')
        np.testing.assert_array_equal(result,(prob>=.5).astype(np.uint8)*7)
        _,status,_=reconcile(original,prob,7,min_area=7)
        self.assertEqual(status,'unknown')

    def test_unknown_is_never_assumed_negative(self):
        result,status,_=reconcile(np.zeros((4,5),np.uint8),np.zeros((4,5)),1)
        self.assertEqual(status,'unknown');self.assertFalse(result.any())

    def test_letterbox_inverse_coordinates(self):
        image=Image.new('RGB',(720,440))
        _,_,valid,geometry=letterbox(image,256)
        restored=restore(np.asarray(valid,dtype=np.float32)/255,geometry)
        self.assertEqual(restored.shape,(440,720))
        np.testing.assert_allclose(restored,1.)
        left,top,w,h,_,_=geometry
        probability=np.zeros((256,256),np.float32)
        probability[top:top+h,left:left+w//2]=1
        output=restore(probability,geometry)
        self.assertGreater(output[220,100],.99)
        self.assertLess(output[220,600],.01)

    def test_nrrd_roundtrip_preserves_voxels_and_rejects_offsets(self):
        h={'type':'unsigned char','dimension':'3','sizes':'9 7 5','encoding':'gzip',
           'space':'left-posterior-superior','space directions':'(1,0,0) (0,1,0) (0,0,1)',
           'space origin':'(0,0,0)','Segment0_LabelValue':'7','Segment0_Layer':'0',
           'Segment0_Name':'gland','Segment0_Extent':'1 2 1 2 1 2'}
        a=np.zeros((5,7,9),np.uint8);a[1,2:5,3:6]=7
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'sample.seg.nrrd';write_volume(path,a,h)
            b,_,label=read_volume(path)
            np.testing.assert_array_equal(a,b);self.assertEqual(label,7)
        h['Segmentation_ReferenceImageExtentOffset']='0 0 1'
        with self.assertRaises(ValueError):
            check_header(h)

    def test_saved_case_and_tg3k_splits(self):
        cases=read_json(ROOT/'configs/cases.json')
        old=read_json(ROOT/'provenance/manifest.json')
        self.assertEqual({r['case_id'] for r in cases},{r['case_id'] for r in old if r['target']=='gland'})
        splits=read_json(ROOT/'provenance/case_split.json')
        self.assertTrue(all(r['split']==splits[str(r['case_id'])] for r in cases))
        ids=read_json(ROOT/'provenance/tg3k/split_ids.json')
        self.assertFalse(set(ids['train'])&set(ids['val']))
        self.assertFalse(set(ids['unused'])&(set(ids['train'])|set(ids['val'])))
        rows=read_csv(ROOT/'provenance/tg3k/group_summary.csv')
        for r in rows:
            start,end=int(r['start']),int(r['end'])
            train=[x for x in ids['train'] if start<=x<=end]
            val=[x for x in ids['val'] if start<=x<=end]
            self.assertTrue(train and val)
            self.assertGreater(min(abs(t-v) for t in train for v in val),int(r['gap_each_side']))

    def test_original_labels_only_in_all_splits(self):
        rows=[dict(split=s,source=source) for s in ('train','val','test') for source in ('provided','pseudo','unknown')]
        self.assertEqual(len(select_rows(rows,'train','finetune')),1)
        self.assertEqual([r['source'] for r in select_rows(rows,'val','finetune')],['provided'])
        self.assertEqual([r['source'] for r in select_rows(rows,'test','finetune')],['provided'])
        self.assertEqual([r['source'] for r in select_rows(rows,'val','pretrain')],['provided'])

    def test_padding_does_not_change_loss(self):
        logits=torch.zeros(1,1,4,4);y=torch.zeros_like(logits);v=torch.zeros_like(logits)
        v[:,:,1:3,1:3]=1;y[:,:,1:3,1:3]=1
        a=loss_fn(logits,y,v,torch.ones(1))
        logits[:,:,0,:]=100
        b=loss_fn(logits,y,v,torch.ones(1))
        torch.testing.assert_close(a,b)

    def test_local_split_and_tg3k_test_only_policy(self):
        from src.policy import validate_rows
        case=read_json(ROOT/'configs/cases.json')[0]
        row=dict(dataset='local',case_id=str(case['case_id']),frame='0',
                 split=case['split'],source='provided',foreground_pixels='10')
        validate_rows([row])
        with self.assertRaises(ValueError):
            validate_rows([dict(row,foreground_pixels='0')])
        with self.assertRaises(ValueError):
            validate_rows([dict(row,source='pseudo')])
        with self.assertRaises(ValueError):
            validate_rows([dict(row,split='test' if case['split']!='test' else 'train')])
        ids=read_json(ROOT/'provenance/tg3k/split_ids.json')
        tg=dict(dataset='tg3k_unused',case_id='TG3K_G01',frame=str(ids['unused'][0]),
                split='test',source='provided',foreground_pixels='10')
        validate_rows([tg],allow_tg3k=True)
        for invalid in (dict(tg,split='train'),dict(tg,split='val'),dict(tg,frame=str(ids['train'][0]))):
            with self.assertRaises(ValueError):
                validate_rows([invalid],allow_tg3k=True)
        with self.assertRaises(ValueError):
            validate_rows([tg])

    def test_model_shape_forward_only(self):
        torch.set_num_threads(2)
        model=MobileUNet(imagenet=False).eval()
        with torch.inference_mode():
            result=model(torch.zeros(1,3,256,256))
        self.assertEqual(tuple(result.shape),(1,1,256,256))
        self.assertTrue(torch.isfinite(result).all())

if __name__=='__main__':
    unittest.main()
