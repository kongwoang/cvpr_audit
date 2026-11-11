"""Registered action-independent stopping tubes on the frozen projection grid."""
import numpy as np
from vtf import speculative as SP


def tube(v, tau, cam, W, H, pp):
    assert tau in (1.,3.5)
    length = pp.standoff + v*pp.reaction_t + v*v/(2*tau)
    rr, yy = np.meshgrid(SP.R_GRID, SP.Y_GRID, indexing='ij')
    inside = (yy-SP.OBJ_W/2 < pp.corridor_half_w) & (yy+SP.OBJ_W/2 > -pp.corridor_half_w)
    mask = (rr <= min(length, pp.max_range)) & inside
    if not mask.any():
        return None, dict(analytic_length_m=float(length),farthest_grid_m=None,mask=mask)
    r,y = rr[mask],yy[mask];p=cam.calib.P2
    fx,fy,cx,cy = p[0,0],p[1,1],p[0,2],p[1,2]
    vb=cy+fy*cam.cam_h/r;uc=cx-fx*y/r;hw=fx*SP.OBJ_W/(2*r)
    box=np.array([max((uc-hw).min(),0),max((vb-fy*SP.OBJ_H/r).min(),0),min((uc+hw).max(),W),min(vb.max(),H)])
    return box, dict(analytic_length_m=float(length),farthest_grid_m=float(r.max()),mask=mask)
