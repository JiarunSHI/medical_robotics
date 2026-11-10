import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from .core import contributions, plan_grid, reconstruct, fuse_frame, fill_small_holes, read_case
from .nrrd_io import write_nrrd, read_nrrd
from .viewer import oblique_slice, plane_box_outline


class GeometryTests(unittest.TestCase):
    def test_kernel_conserves_weight_and_standard_linear(self):
        q = np.array([[2.2, 3.3, 4.4], [0., 1., 2.]])
        ix, w = contributions(q, 'linear', [1, 1, 1])
        np.testing.assert_allclose(w.sum(1), 1)
        self.assertAlmostEqual(w[0, 0], .8*.7*.6)
        np.testing.assert_allclose((ix*w[:,:,None]).sum(1),q)
        for gamma in [[.25,4,2],[1,1,1]]:
            _, w = contributions(q,'linear',gamma)
            np.testing.assert_allclose(w.sum(1),1)

    def test_nearest_is_round_not_floor(self):
        ix,_ = contributions(np.array([[.6,1.4,2.5]]),'nearest',[1]*3)
        np.testing.assert_array_equal(ix[0,0],[1,1,3])

    def test_physical_scaling_precedes_calibration_and_track(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            C=np.array([[0,-1,0,10],[1,0,0,20],[0,0,1,30],[0,0,0,1]],float)
            (p/'imageToProbeTransformMatrix_wld.xml').write_text('<Root><Transform From="Image" To="Probe" Matrix="'+ ' '.join(map(str,C.ravel()))+'" /></Root>')
            (p/'track.csv').write_text('0,1,0,0,100,0,1,0,200,0,0,1,300,\n',encoding='utf-8-sig')
            Image.new('L',(3,4),100).save(p/'0.bmp')
            frames,_=read_case(p,[.1,.2])
            np.testing.assert_allclose(frames[0]['transform']@np.array([2,3,0,1]),[109.4,220.2,330,1])

    def test_constant_frame_duplicate_hits_and_empty_zero(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'0.bmp';Image.new('L',(5,5),100).save(path)
            T=np.diag([.1,.1,1.,1.]);T[:3,3]=[.25,.25,.25]
            f={'id':0,'path':path,'width':5,'height':5,'transform':T}
            grid=plan_grid([f],[1,1,1])
            v,w=reconstruct([f],grid,chunk_pixels=3)
            np.testing.assert_allclose(v[w>0],100,atol=1e-4)
            self.assertAlmostEqual(float(w.sum()),25,places=4)
            self.assertTrue(np.all(v[w==0]==0))
            # Black is valid observed data, not a missing-pixel criterion.
            Image.new('L',(5,5),0).save(path)
            v,w=reconstruct([f],grid)
            self.assertGreater(int((w>0).sum()),0);self.assertEqual(float(v.max()),0)

    def test_decay_once_per_frame(self):
        total=np.array([[[100.]]],np.float32);w=np.array([[[2.]]],np.float32)
        fs=np.array([[[300.]]],np.float32);fw=np.array([[[3.]]],np.float32)
        fuse_frame(total,w,np.s_[:,:,:],fs,fw,'decayed-mean',.5)
        self.assertAlmostEqual(float((total/w)[0,0,0]),87.5)
        self.assertEqual(float(w[0,0,0]),4.)

    def test_nrrd_geometry_order_and_values(self):
        with tempfile.TemporaryDirectory() as d:
            v=np.arange(24,dtype=np.float32).reshape(2,3,4)
            p=Path(d)/'x.nrrd';write_nrrd(p,v,[-5,6,7],[.2,.3,.4])
            raw=p.read_bytes();self.assertIn(b'sizes: 4 3 2',raw.split(b'\n\n')[0])
            import gzip
            np.testing.assert_array_equal(np.frombuffer(gzip.decompress(raw.split(b'\n\n',1)[1]),dtype='<f4'),v.ravel())
            vv,o,s=read_nrrd(p)
            np.testing.assert_array_equal(vv,v);np.testing.assert_allclose(o,[-5,6,7]);np.testing.assert_allclose(s,[.2,.3,.4])

    def test_hole_fill_keeps_original_observations(self):
        v=np.ones((7,7,7),np.float32)*80;w=np.ones_like(v)
        w[3,3,3]=0;v[3,3,3]=0
        result,mask=fill_small_holes(v,w,[1,1,1],2)
        self.assertTrue(mask[3,3,3]);self.assertAlmostEqual(float(result[3,3,3]),80,places=4)
        np.testing.assert_array_equal(result[w>0],v[w>0]);self.assertEqual(v[3,3,3],0)

    def test_oblique_identity_matches_xy_plane(self):
        v=np.arange(125,dtype=np.float32).reshape(5,5,5)
        plane,_=oblique_slice(v,np.zeros(3),np.ones(3),np.array([2,2,2]),0,0,4,pixels=5)
        np.testing.assert_allclose(plane,v[2])

    def test_plane_outline_stays_in_box(self):
        poly=plane_box_outline(np.array([2,2,2]),np.array([0.,0.,1.]),np.zeros(3),np.array([4,4,4]))
        self.assertEqual(poly.shape,(5,3))
        np.testing.assert_allclose(poly[:,2],2)
        self.assertTrue(np.all((poly>=0)&(poly<=4)))


if __name__=='__main__':unittest.main()
