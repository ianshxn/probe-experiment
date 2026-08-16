import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
OUT="/home/cfeng/Projects/personal/probes/plots"
R="/home/cfeng/Projects/personal/probes/results"
SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; GRID="#e1e0d9"; BASE="#c3c2b7"
S8B="#2a78d6"; S70="#eb6834"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans","Segoe UI","Arial"],
    "figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,"text.color":INK,
    "axes.labelcolor":INK2,"xtick.color":MUTED,"ytick.color":MUTED,"axes.edgecolor":BASE,"font.size":10})
d8=json.load(open(f"{R}/layer_sweep_llama31_8b.json")); d70=json.load(open(f"{R}/layer_sweep_llama33_70b.json"))
def curves(d):
    x=np.array([r["layer"]/(len(d)-1)*100 for r in d])
    dec=np.array([r["decorrelated"]["auc"] for r in d])
    xf=np.array([(r["b2c"]["auc"]+r["c2b"]["auc"])/2 for r in d])
    return x,dec,xf
x8,dec8,xf8=curves(d8); x70,dec70,xf70=curves(d70)
def axchrome(ax):
    ax.grid(True,color=GRID,lw=0.8,zorder=0); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(BASE)

# --- 06 ---
fig,axes=plt.subplots(1,2,figsize=(11.4,5.7),sharey=True)
for ax,x,dec,xf,c,name in ((axes[0],x8,dec8,xf8,S8B,"Llama-3.1-8B"),
                           (axes[1],x70,dec70,xf70,S70,"Llama-3.3-70B")):
    ax.fill_between(x,xf,dec,color=c,alpha=0.16,zorder=2,lw=0)
    ax.plot(x,dec,color=c,lw=2,zorder=3)
    ax.plot(x,xf,color=c,lw=2,ls=(0,(4,3)),zorder=3)
    ax.axhline(0.5,color=BASE,lw=1.2,ls=":",zorder=1)
    ax.text(1,0.508,"chance",ha="left",va="bottom",color=MUTED,fontsize=9)
    ax.set_title(name,color=INK,loc="left",fontsize=11,fontweight="bold",pad=8)
    ax.set_xlabel("relative depth (%)"); ax.set_ylim(0.45,1.02)
    axchrome(ax)
axes[1].annotate("format-bound\nsignal",xy=(76,(dec70[60]+xf70[60])/2),ha="center",va="center",
                 color=S70,fontsize=9.5,fontweight="bold")
axes[0].annotate("format-bound\nsignal",xy=(62,(dec8[19]+xf8[19])/2),ha="center",va="center",
                 color=S8B,fontsize=9.5,fontweight="bold")
axes[0].set_ylabel("purpose AUC")
fig.text(0.007,0.98,"Purpose signal: transferable vs format-bound",
         fontsize=13,fontweight="bold",color=INK,va="top",ha="left")
fig.text(0.007,0.925,"Solid: all readable purpose signal (decorrelated).   Dashed: the part surviving a format switch (mean of both directions).   Shaded: the format-bound difference.",
         fontsize=9.5,color=INK2,va="top",ha="left")
fig.legend(handles=[Line2D([],[],color=MUTED,lw=2,label="all signal"),
                    Line2D([],[],color=MUTED,lw=2,ls=(0,(4,3)),label="transfers"),
                    Patch(facecolor=MUTED,alpha=0.25,label="format-bound")],
           loc="upper right",bbox_to_anchor=(0.995,0.995),ncol=3,frameon=False,fontsize=9.5,
           labelcolor=INK2,handlelength=1.8,columnspacing=1.2)
fig.tight_layout(rect=[0,0,1,0.87]); fig.savefig(f"{OUT}/06_signal_split_by_depth.png",dpi=200); plt.close(fig)

# --- 07 ---
fig,ax=plt.subplots(figsize=(9.8,5.7))
g8,g70=dec8-xf8,dec70-xf70
ax.plot(x8,g8,color=S8B,lw=2,zorder=3)
ax.plot(x70,g70,color=S70,lw=2,zorder=3)
ax.axhline(0,color=BASE,lw=1.2,ls=":",zorder=1)
ax.text(99,0.004,"fully format-invariant",ha="right",va="bottom",color=MUTED,fontsize=9)
i19=19
ax.plot([x70[i19]],[g70[i19]],"o",ms=9,color=S70,mec=SURF,mew=2,zorder=4)
ax.annotate(f"70B layer 19 · gap {g70[i19]:+.3f}\nnearly all signal transfers",
            (x70[i19],g70[i19]),textcoords="offset points",xytext=(12,-38),
            color=S70,fontsize=9.5,fontweight="bold")
ax.set_xlabel("relative depth through the network (%)")
ax.set_ylabel("format-bound gap  (decorrelated − transfer AUC)")
axchrome(ax); ax.set_ylim(-0.02,0.32)
fig.legend(handles=[Line2D([],[],color=S8B,lw=2,label="Llama-3.1-8B"),
                    Line2D([],[],color=S70,lw=2,label="Llama-3.3-70B")],
           loc="upper right",bbox_to_anchor=(0.995,0.995),ncol=1,frameon=False,fontsize=9.5,
           labelcolor=INK2,handlelength=1.8)
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(f"{OUT}/07_format_bound_gap.png",dpi=200); plt.close(fig)
print("fixed 06, 07")
