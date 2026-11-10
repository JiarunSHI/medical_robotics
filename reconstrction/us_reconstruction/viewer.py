"""Local Matplotlib viewer: three reference-axis slices, oblique MPR, 3D sampled points."""
import argparse
import json
from pathlib import Path
import numpy as np
from .nrrd_io import read_nrrd


def plane_box_outline(centre, normal, low, high):
    """Intersect an infinite plane with the volume box, ordered around its perimeter."""
    corners = np.array([[x,y,z] for x in (low[0],high[0]) for y in (low[1],high[1]) for z in (low[2],high[2])])
    distance = (corners-centre) @ normal
    points = []
    for i in range(8):
        for bit in (1,2,4):
            j=i^bit
            if j<=i: continue
            if abs(distance[i]) < 1e-8: points.append(corners[i])
            if abs(distance[j]) < 1e-8: points.append(corners[j])
            if distance[i]*distance[j] < 0:
                t=distance[i]/(distance[i]-distance[j])
                points.append(corners[i]+t*(corners[j]-corners[i]))
    if len(points)<3: return np.empty((0,3))
    points=np.unique(np.round(points,8),axis=0)
    c=points.mean(0);u=points[0]-c
    if np.linalg.norm(u)<1e-12:return np.empty((0,3))
    u/=np.linalg.norm(u);v=np.cross(normal,u);v/=np.linalg.norm(v)
    points=points[np.argsort(np.arctan2((points-c)@v,(points-c)@u))]
    return np.concatenate([points,points[:1]])


def oblique_slice(volume, origin, spacing, centre, yaw, pitch, width_mm, pixels=256):
    from scipy.ndimage import map_coordinates
    yaw, pitch = np.deg2rad([yaw, pitch])
    u = np.array([np.cos(yaw), np.sin(yaw), 0.])
    v = np.array([-np.sin(yaw)*np.cos(pitch), np.cos(yaw)*np.cos(pitch), np.sin(pitch)])
    t = np.linspace(-width_mm/2, width_mm/2, pixels)
    aa, bb = np.meshgrid(t, t)
    p = centre[:, None, None] + u[:, None, None]*aa + v[:, None, None]*bb
    q = (p-origin[:, None, None])/spacing[:, None, None]
    result = map_coordinates(volume, q[::-1], order=1, mode='constant', cval=0., prefilter=False)
    corners = np.array([centre + u*a + v*b for a, b in [(-1,-1),(1,-1),(1,1),(-1,1),(-1,-1)]])
    # Scale corners about the centre, not about the global origin.
    corners = centre + (corners-centre)*width_mm/2
    return result, corners


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('volume', type=Path, help='Output directory or volume.nrrd')
    p.add_argument('--snapshot', type=Path, help='Save a PNG without opening a window')
    p.add_argument('--points', type=int, default=12000)
    p.add_argument('--threshold', type=float, default=30.)
    a = p.parse_args(argv)
    if a.points < 1: p.error('--points must be positive')
    import matplotlib
    if a.snapshot: matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, CheckButtons
    if a.volume.is_dir():
        path = a.volume / ('filled_volume.nrrd' if (a.volume / 'filled_volume.nrrd').exists() else 'volume.nrrd')
    else:
        path = a.volume
    volume, origin, spacing = read_nrrd(path)
    metadata_path = path.parent / 'metadata.json'
    crop_note = 'Crop information unavailable.'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        crop = metadata.get('crop')
        if crop:
            if metadata.get('pixel_coordinate_convention') == 'original_full_image':
                crop_note = 'Lateral ROI: ' + crop['mode'] + '\nOriginal full-image pixel coordinates.'
            else:
                crop_note = 'Lateral crop: ' + crop['mode'] + '\nROI-local coordinates; calibrated offset.'
        else:
            crop_note = 'Legacy result: full image with borders.'
    mask_path = path.parent / 'observed_mask.nrrd'
    observed = read_nrrd(mask_path)[0] > 0 if mask_path.exists() else np.ones(volume.shape, bool)
    dims = np.array(volume.shape[::-1]); end = origin+(dims-1)*spacing
    valid = volume[observed]
    vmax = max(1., float(np.percentile(valid, 99.5))) if len(valid) else 255.
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle('Tracked ultrasound | ' + path.name + ' | abstract reference XYZ (mm)')
    axes = [fig.add_subplot(2, 3, i) for i in [1, 2, 3, 4]]
    ax3d = fig.add_subplot(2, 3, 5, projection='3d')
    textax = fig.add_subplot(2, 3, 6); textax.axis('off')
    fig.subplots_adjust(bottom=.28, top=.91, hspace=.35, wspace=.3)
    idx = (dims-1)//2
    images = []
    for ax, data, extent, labels in [
        (axes[0],volume[:,:,idx[0]],[origin[1],end[1],origin[2],end[2]],('Y mm','Z mm')),
        (axes[1],volume[:,idx[1],:],[origin[0],end[0],origin[2],end[2]],('X mm','Z mm')),
        (axes[2],volume[idx[2],:,:],[origin[0],end[0],origin[1],end[1]],('X mm','Y mm'))]:
        images.append(ax.imshow(data, origin='lower', extent=extent, cmap='gray', vmin=0, vmax=vmax, aspect='equal'))
        ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])
    width = float(np.linalg.norm(end-origin))
    centre = origin+idx*spacing
    ob, corners = oblique_slice(volume, origin, spacing, centre, 0, 0, width)
    images.append(axes[3].imshow(ob, origin='lower', extent=[-width/2,width/2,-width/2,width/2], cmap='gray', vmin=0, vmax=vmax))
    axes[3].set_xlabel('Plane U mm'); axes[3].set_ylabel('Plane V mm')
    # Subsample on a regular lattice before selecting bright points; bound viewer memory.
    stride = max(1, int(np.ceil((volume.size/300000.)**(1/3))))
    coarse = volume[::stride,::stride,::stride]
    keep = (coarse >= a.threshold) & observed[::stride,::stride,::stride]
    coords = np.argwhere(keep)
    if len(coords) > a.points: coords = coords[np.linspace(0,len(coords)-1,a.points,dtype=int)]
    xyz = origin + coords[:,::-1]*stride*spacing
    colors = coarse[tuple(coords.T)] if len(coords) else np.array([])
    cloud = ax3d.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=colors,cmap='gray',vmin=0,vmax=vmax,s=2,alpha=.3)
    line, = ax3d.plot([],[],[],color='cyan')
    for setter, lo, hi in [(ax3d.set_xlim,origin[0],end[0]),(ax3d.set_ylim,origin[1],end[1]),(ax3d.set_zlim,origin[2],end[2])]: setter(lo,hi)
    ax3d.set_xlabel('X mm'); ax3d.set_ylabel('Y mm'); ax3d.set_zlabel('Z mm')
    if hasattr(ax3d,'set_box_aspect'): ax3d.set_box_aspect(np.maximum(end-origin,1))
    ax3d.set_title('3D sampled intensity points + oblique plane')
    textax.text(0,1,'Drag XYZ: move slice intersection\nYaw/Pitch: rotate oblique plane\nWindow/Level: adjust grayscale\nDrag 3D view: rotate camera\nObserved only: hide unmeasured voxels\n\n3D points are a sampled preview,\nnot a segmentation or surface.\n\nFor volume rendering, open in Slicer.\n'+crop_note,va='top',fontsize=10)
    sliders = {}
    entries = [('X',origin[0],end[0],centre[0]),('Y',origin[1],end[1],centre[1]),('Z',origin[2],end[2],centre[2]),
               ('Yaw',-180,180,0),('Pitch',-90,90,0),('Window',1,510,vmax),('Level',0,255,vmax/2)]
    for i,(name,lo,hi,val) in enumerate(entries):
        column,row=divmod(i,4)
        sliders[name]=Slider(fig.add_axes([.10+column*.46,.20-row*.045,.32,.022]),name,lo,max(hi,lo+1e-6),valinit=val)
    checkbox=CheckButtons(fig.add_axes([.80,.035,.18,.07]),['Observed only'],[False])
    def update(_=None):
        c=np.array([sliders[k].val for k in 'XYZ']); ix=np.clip(np.rint((c-origin)/spacing).astype(int),0,dims-1)
        slices=[volume[:,:,ix[0]],volume[:,ix[1],:],volume[ix[2],:,:]]
        masks=[observed[:,:,ix[0]],observed[:,ix[1],:],observed[ix[2],:,:]]
        hide=checkbox.get_status()[0]
        for i,(data,m) in enumerate(zip(slices,masks)):
            images[i].set_data(np.ma.masked_where(~m,data) if hide else data)
            axes[i].set_title('%s = %.2f mm' % ('XYZ'[i],origin[i]+ix[i]*spacing[i]))
        ob, poly=oblique_slice(volume,origin,spacing,c,sliders['Yaw'].val,sliders['Pitch'].val,width)
        if hide:
            support,_=oblique_slice(observed.astype(np.float32),origin,spacing,c,sliders['Yaw'].val,sliders['Pitch'].val,width)
            ob=np.ma.masked_where(support<.999,ob)
        images[3].set_data(ob);axes[3].set_title('Oblique MPR: yaw %.0f / pitch %.0f' % (sliders['Yaw'].val,sliders['Pitch'].val))
        normal=np.cross(poly[1]-poly[0],poly[2]-poly[1]);normal/=np.linalg.norm(normal)
        outline=plane_box_outline(c,normal,origin,end)
        line.set_data(outline[:,0],outline[:,1]);line.set_3d_properties(outline[:,2])
        low=sliders['Level'].val-sliders['Window'].val/2;high=sliders['Level'].val+sliders['Window'].val/2
        for im in images: im.set_clim(low,high)
        cloud.set_clim(low,high);fig.canvas.draw_idle()
    for slider in sliders.values(): slider.on_changed(update)
    checkbox.on_clicked(update);update()
    if a.snapshot:
        a.snapshot.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.snapshot,dpi=130);plt.close(fig)
        print('Saved snapshot:',a.snapshot.resolve())
    else: plt.show()
    return fig, sliders, checkbox


if __name__ == '__main__': main()
