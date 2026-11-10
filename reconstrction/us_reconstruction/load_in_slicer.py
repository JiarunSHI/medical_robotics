"""Run inside 3D Slicer: exec(open(r'.../load_in_slicer.py', encoding='utf-8').read()); load_reconstruction(r'...')"""
from pathlib import Path


def load_reconstruction(folder, use_filled=None):
    import slicer
    folder = Path(folder)
    if use_filled is None: use_filled = (folder / 'filled_volume.nrrd').exists()
    path = folder / ('filled_volume.nrrd' if use_filled else 'volume.nrrd')
    node = slicer.util.loadVolume(str(path))
    node.SetName(folder.name + ' - abstract reference')
    node.SetAttribute('ReferenceFrame', 'Abstract tracking frame; no patient orientation registration')
    slicer.util.setSliceViewerLayers(background=node)
    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    slicer.util.resetSliceViews()
    logic = slicer.modules.volumerendering.logic()
    display = logic.CreateDefaultVolumeRenderingNodes(node)
    # A neutral ultrasound grayscale transfer function, editable in Volume Rendering.
    prop = display.GetVolumePropertyNode().GetVolumeProperty()
    prop.GetRGBTransferFunction().RemoveAllPoints()
    prop.GetRGBTransferFunction().AddRGBPoint(0, 0, 0, 0)
    prop.GetRGBTransferFunction().AddRGBPoint(255, 1, 1, 1)
    opacity = prop.GetScalarOpacity(); opacity.RemoveAllPoints()
    for value, alpha in [(0,0),(20,0),(80,.04),(160,.15),(255,.35)]: opacity.AddPoint(value,alpha)
    display.SetVisibility(True)
    for name in slicer.app.layoutManager().sliceViewNames():
        slicer.app.layoutManager().sliceWidget(name).mrmlSliceNode().SetSliceVisible(True)
    slicer.util.resetThreeDViews()
    print('Loaded abstract reference. RAS/LPS labels are display conventions, NOT verified patient orientation.')
    return node
