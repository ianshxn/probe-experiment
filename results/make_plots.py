import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SP="/tmp/claude-1000/-home-cfeng-Projects-personal-probes/4dab0446-2b30-487f-a03a-7f437886ffd0/scratchpad"
OUT="/home/cfeng/Projects/personal/probes/plots"
SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; GRID="#e1e0d9"; BASE="#c3c2b7"
S8B="#2a78d6"; S70="#eb6834"; S3="#1baf7a"
plt.rcParams.update({"font.family":"sans-serif",
    "font.sans-serif":["DejaVu Sans","Segoe UI","Helvetica","Arial"],
    "figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
    "text.color":INK,"axes.labelcolor":INK2,"xtick.color":MUTED,"ytick.color":MUTED,
    "axes.edgecolor":BASE,"font.size":10})

d8=json.load(open(f"{SP}/full_8b.json")); d70=json.load(open(f"{SP}/full_70b.json"))
def depth(d): n=len(d); return np.array([r["layer"]/(n-1)*100 for r in d])
def series(d,reg,key): return np.array([r[reg][key] for r in d])
x8,x70=depth(d8),depth(d70)
p8=max(d8,key=lambda r:r["decorrelated"]["auc"]); p70=max(d70,key=lambda r:r["decorrelated"]["auc"])

def header(fig,title,sub,handles,ncol):
    fig.text(0.008,0.975,title,fontsize=13,fontweight="bold",color=INK,va="top",ha="left")
    fig.text(0.008,0.917,sub,fontsize=9.5,color=INK2,va="top",ha="left")
    if handles: fig.legend(handles=handles,loc="upper right",bbox_to_anchor=(0.995,0.995),
                           ncol=ncol,frameon=False,fontsize=9.5,labelcolor=INK2,
                           handlelength=1.8,columnspacing=1.4)
def axchrome(ax,xlab,ylab):
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.grid(True,color=GRID,lw=0.8,zorder=0); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(BASE)

# --- 1 ---
fig,ax=plt.subplots(figsize=(9.4,5.8))
ax.plot(x8,series(d8,"decorrelated","auc"),color=S8B,lw=2,zorder=3)
ax.plot(x70,series(d70,"decorrelated","auc"),color=S70,lw=2,zorder=3)
ax.plot(x8,series(d8,"confounded","auc"),color=S8B,lw=2,ls=(0,(4,3)),alpha=.85,zorder=2)
ax.plot(x70,series(d70,"confounded","auc"),color=S70,lw=2,ls=(0,(4,3)),alpha=.85,zorder=2)
ax.axhline(0.5,color=BASE,lw=1.2,ls=":",zorder=1)
ax.text(1,0.508,"chance",ha="left",va="bottom",color=MUTED,fontsize=9)
for p,n,c,xa in ((p8,"8B",S8B,x8),(p70,"70B",S70,x70)):
    xv=p["layer"]/(len(xa)-1)*100; yv=p["decorrelated"]["auc"]
    ax.plot([xv],[yv],"o",ms=9,color=c,mec=SURF,mew=2,zorder=4)
    dy = -22 if n=="8B" else 12
    ax.annotate(f"{n} peak · L{p['layer']} · AUC {yv:.3f}",(xv,yv),textcoords="offset points",
                xytext=(10,dy),color=c,fontsize=9.5,fontweight="bold",zorder=5)
axchrome(ax,"relative depth through the network (%)","purpose AUC (out-of-fold)")
ax.set_ylim(0.42,1.04)
header(fig,"Purpose signal peaks early — and the confounded probe never rises",
       "Solid: decorrelated training.   Dashed: standard confounded training.",
       [Line2D([],[],color=S8B,lw=2,label="Llama-3.1-8B"),
        Line2D([],[],color=S70,lw=2,label="Llama-3.3-70B"),
        Line2D([],[],color=MUTED,lw=2,ls=(0,(4,3)),label="confounded")],3)
fig.tight_layout(rect=[0,0,1,0.87]); fig.savefig(f"{OUT}/01_purpose_signal_by_depth.png",dpi=200); plt.close(fig)

# --- 2 ---
fig,axes=plt.subplots(1,2,figsize=(11.4,5.6),sharey=True)
cells=["evaluation::benchmark","evaluation::casual","deployment::benchmark","deployment::casual"]
lbl=["eval\nquiz","eval\ncasual","deploy\nquiz","deploy\ncasual"]
for ax,p,name,col in ((axes[0],p8,"Llama-3.1-8B  ·  layer 8",S8B),(axes[1],p70,"Llama-3.3-70B  ·  layer 31",S70)):
    i=np.arange(4); w=0.38
    cv=[p["confounded"]["cells"][c] for c in cells]; dv=[p["decorrelated"]["cells"][c] for c in cells]
    ax.bar(i-w/2-0.012,cv,w,color=MUTED,zorder=3)
    ax.bar(i+w/2+0.012,dv,w,color=col,zorder=3)
    for xi,v in zip(i-w/2-0.012,cv): ax.text(xi,v+2.5,f"{v:.0f}",ha="center",color=INK2,fontsize=9,fontweight="bold")
    for xi,v in zip(i+w/2+0.012,dv): ax.text(xi,v+2.5,f"{v:.0f}",ha="center",color=INK,fontsize=9,fontweight="bold")
    ax.set_xticks(i); ax.set_xticklabels(lbl,color=INK2,fontsize=9.5)
    ax.set_title(name,color=INK,loc="left",fontsize=11,fontweight="bold",pad=10)
    ax.set_ylim(0,110); axchrome(ax,"",""); ax.grid(axis="x",visible=False)
axes[0].set_ylabel("% of prompts called “evaluation”")
header(fig,"The confounded probe answers by format, not purpose","",
       [Line2D([],[],color=MUTED,lw=9,label="confounded training (only eval-quiz + deploy-casual)"),
        Line2D([],[],color=S8B,lw=9,label="decorrelated training, Llama-3.1-8B (all four cells)"),
        Line2D([],[],color=S70,lw=9,label="decorrelated training, Llama-3.3-70B (all four cells)")],1)
fig.tight_layout(rect=[0,0,1,0.86]); fig.savefig(f"{OUT}/02_confound_signature.png",dpi=200); plt.close(fig)

# --- 3 ---
fig,axes=plt.subplots(1,2,figsize=(11.4,5.2),sharey=True)
for ax,reg,title in ((axes[0],"b2c","trained on quiz  →  tested on casual"),
                     (axes[1],"c2b","trained on casual  →  tested on quiz")):
    ax.plot(x8,series(d8,reg,"auc"),color=S8B,lw=2,zorder=3)
    ax.plot(x70,series(d70,reg,"auc"),color=S70,lw=2,zorder=3)
    ax.axhline(0.5,color=BASE,lw=1.2,ls=":",zorder=1)
    ax.text(1,0.508,"chance",ha="left",va="bottom",color=MUTED,fontsize=9)
    ax.set_title(title,color=INK,loc="left",fontsize=11,fontweight="bold",pad=10)
    axchrome(ax,"relative depth (%)",""); ax.set_ylim(0.42,1.0)
axes[0].set_ylabel("purpose AUC on the unseen format")
header(fig,"Transfer to an unseen prompt format improves with scale","",
       [Line2D([],[],color=S8B,lw=2,label="Llama-3.1-8B"),Line2D([],[],color=S70,lw=2,label="Llama-3.3-70B")],2)
fig.tight_layout(rect=[0,0,1,0.86]); fig.savefig(f"{OUT}/03_transfer_by_depth.png",dpi=200); plt.close(fig)

# --- 4 ---
fig,ax=plt.subplots(figsize=(9.6,5.6))
groups=["decorrelated","quiz → casual","casual → quiz"]; regs=["decorrelated","b2c","c2b"]
i=np.arange(3); w=0.2
for k,(p,name,col) in enumerate(((p8,"8B",S8B),(p70,"70B",S70))):
    au=[p[r]["auc"] for r in regs]; ba=[p[r]["bal_acc"] for r in regs]
    off=(k-0.5)*2*w
    ax.bar(i+off-w*0.26,au,w*0.5,color=col,zorder=3)
    ax.bar(i+off+w*0.26,ba,w*0.5,color=col,alpha=0.40,zorder=3)
    for xi,v in zip(i+off-w*0.26,au): ax.text(xi,v+.012,f"{v:.2f}",ha="center",fontsize=8.5,color=INK,fontweight="bold")
    for xi,v in zip(i+off+w*0.26,ba): ax.text(xi,v+.012,f"{v:.2f}",ha="center",fontsize=8.5,color=INK2)
    for g in range(3): ax.text(i[g]+off,-0.062,name,ha="center",color=INK2,fontsize=9.5,fontweight="bold")
ax.axhline(0.5,color=BASE,lw=1.2,ls=":",zorder=1)
ax.set_xticks(i); ax.set_xticklabels(groups,color=INK,fontsize=11,fontweight="bold")
ax.tick_params(axis="x",pad=24,length=0); ax.set_ylim(0,1.06)
axchrome(ax,"","score"); ax.grid(axis="x",visible=False)
header(fig,"Ranking survives a format change; the decision threshold does not",
       "Solid bar: AUC (ranking quality).   Pale bar: balanced accuracy (fixed threshold at 0).",
       [Line2D([],[],color=MUTED,lw=9,label="AUC"),
        Line2D([],[],color=MUTED,lw=9,alpha=0.40,label="balanced accuracy")],2)
fig.tight_layout(rect=[0,0,1,0.86]); fig.savefig(f"{OUT}/04_ranking_vs_threshold.png",dpi=200); plt.close(fig)

# --- 5 ---
fig,ax=plt.subplots(figsize=(9.4,5.6))
ax.axhline(1.0,color=S3,lw=1.4,ls=(0,(4,3)),zorder=1)
ax.plot(x8,series(d8,"decorrelated","paired_pos"),color=S8B,lw=2,zorder=3)
ax.plot(x70,series(d70,"decorrelated","paired_pos"),color=S70,lw=2,zorder=3)
ax.axhline(0.5,color=BASE,lw=1.2,ls=":",zorder=1)
ax.text(1,0.508,"chance",ha="left",va="bottom",color=MUTED,fontsize=9)
ax.text(50,1.012,"72 / 72 payload blocks",ha="center",va="bottom",color=S3,fontsize=9.5,fontweight="bold")
axchrome(ax,"relative depth through the network (%)","fraction of payloads scored eval > deploy")
ax.set_ylim(0.45,1.05)
header(fig,"Within a matched pair, the probe is almost always right",
       "Same question, same wrapper, one sentence changed. Out-of-fold on held-out purpose families.",
       [Line2D([],[],color=S8B,lw=2,label="Llama-3.1-8B"),Line2D([],[],color=S70,lw=2,label="Llama-3.3-70B")],2)
fig.tight_layout(rect=[0,0,1,0.87]); fig.savefig(f"{OUT}/05_paired_consistency_by_depth.png",dpi=200); plt.close(fig)
print("wrote 5 plots")
