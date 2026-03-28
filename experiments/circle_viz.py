#!/usr/bin/env python3
"""Visualize harmonic circle LDOS: gallery, top6, linecuts, spectrum."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, os
n=256; L=10.0; h=L/2; x=np.linspace(-h,h,n,endpoint=False); ext=[-h,h,-h,h]
E=np.linspace(0.5,5.5,300)
_r = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def ff(a,ax): return np.roll(np.flip(a,axis=ax),1,axis=ax)
def sc(r): return 0.25*(r+ff(r,1)+ff(r,0)+ff(ff(r,1),0))
ld=np.load(os.path.join(_r,'figures/ldos/harmonic_circle/ldos_harmonic_circle.npy'))
O=os.path.join(_r,'figures/ldos/harmonic_circle')
sp=ld.sum(axis=(1,2)); pk=np.sort(np.argsort(sp)[::-1][:6])
idx=np.linspace(0,len(E)-1,48,dtype=int)
fig,ax=plt.subplots(6,8,figsize=(28,22))
for k,i in enumerate(idx):
    a=ax[k//8,k%8]; r=sc(ld[i]); a.imshow(r,extent=ext,origin='lower',cmap='inferno',vmin=0,vmax=r.max()); a.set_title(f'E={E[i]:.2f}',fontsize=7); a.set_aspect('equal'); a.axis('off')
fig.suptitle('Harmonic circle LDOS gallery (r0=1.55, massless, M=48)',fontsize=13); plt.tight_layout()
fig.savefig(f'{O}/gallery.png',dpi=120,bbox_inches='tight'); plt.close()
fig,ax=plt.subplots(2,3,figsize=(15,10))
for j,p in enumerate(pk):
    a=ax.ravel()[j]; r=sc(ld[p]); im=a.imshow(r,extent=ext,origin='lower',cmap='inferno',vmin=0,vmax=r.max()); a.set_title(f'E={E[p]:.3f}',fontsize=11); a.set_aspect('equal'); plt.colorbar(im,ax=a,shrink=0.8)
fig.suptitle('Harmonic circle top 6 peaks',fontsize=13); plt.tight_layout()
fig.savefig(f'{O}/top6.png',dpi=150,bbox_inches='tight'); plt.close()
ls=np.array([sc(ld[j]) for j in range(len(E))])
fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5))
a1.pcolormesh(x,E,ls[:,n//2,:],cmap='inferno',shading='auto'); a1.set_xlabel('x'); a1.set_ylabel('E'); a1.set_title('LDOS(E,x) at y=0'); a1.set_xlim(-3,3)
a2.pcolormesh(x,E,ls[:,:,n//2],cmap='inferno',shading='auto'); a2.set_xlabel('y'); a2.set_ylabel('E'); a2.set_title('LDOS(E,y) at x=0'); a2.set_xlim(-2,2)
fig.suptitle('Harmonic circle 1D LDOS',fontsize=12); plt.tight_layout()
fig.savefig(f'{O}/linecuts.png',dpi=150,bbox_inches='tight'); plt.close()
fig,a=plt.subplots(figsize=(10,3.5)); a.plot(E,sp/sp.max(),'b-',lw=1.5); a.set_xlabel('E'); a.set_ylabel('LDOS'); a.set_title('Harmonic circle spectrum'); a.grid(alpha=0.3); plt.tight_layout()
fig.savefig(f'{O}/spectrum.png',dpi=150,bbox_inches='tight'); plt.close()
print('Done: gallery, top6, linecuts, spectrum')
