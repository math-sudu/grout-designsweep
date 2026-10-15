"""Rebuild the four paper figures from the public-paste support calculations."""

from pathlib import Path
import argparse
import json
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle, Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from tools import origin_figures as fs
from grouting_pino.paste_reference import load_paste_references
from grouting_pino.support_response import SectionResponse

DATA = ROOT / "results/support_design"
OUT = ROOT / "paper/figures"
PREVIEW = None
COLORS = [fs.BLACK, fs.RED, fs.BLUE]
MIXES = ["W32", "W40", "W48"]
MARKERS = ["o", "s", "^"]
REFS = load_paste_references(ROOT / "data/public_paste_benchmark")
SECTION = SectionResponse.load(DATA / "reference_response.npz")
RECORDS = []


def panel(ax, label, title):
    fs.style_axes(ax)
    ax.set_title(f"({label}) {title}", loc="left", fontsize=8.5, pad=7)


def save(fig, number, height, label, thesis, inputs):
    fs.save(fig, OUT / f"fig{number}.pdf", size_mm=(136, height),
            preview_path=PREVIEW / f"fig{number}.png" if PREVIEW else None)
    RECORDS.append(dict(id=f"F{number}", status="active", label=label,
                        source="scripts/plot_support_paper.py",
                        outputs=f"paper/figures/fig{number}.pdf",
                        editable=f"paper/figures/fig{number}.svg",
                        size_mm=[136, height], visual_thesis=thesis,
                        data_sources=inputs,
                        verification={"receipt": f"paper/figures/fig{number}.qa.json"}))
    plt.close(fig)


def figure1():
    fig = fs.figure(136, 112)
    ax = fig.add_axes([.035, .39, .44, .56])
    ax.set(xlim=(-32, 32), ylim=(-61, 8), aspect="equal")
    ax.set_axis_off()
    ax.add_patch(Rectangle((-30, -60), 60, 60, facecolor="#f4efe6", edgecolor="black", lw=.9))
    ax.plot([-30, 30], [0, 0], color="black", lw=1.1)
    for r, c in [(6.02, "#cc9563"), (5.8, "#c9c9c9"), (5.25, "white")]:
        ax.add_patch(Circle((0, -25.52), r, facecolor=c, edgecolor="black", lw=.65))
    ax.annotate("", (15, -19.5), (15, 0),
                arrowprops=dict(arrowstyle="<->", lw=.7))
    ax.text(17, -9.5, "19.5 m", rotation=90, fontsize=8, va="center")
    ax.text(0, 4, "Ground surface", ha="center", fontsize=8)
    ax.text(0, -47, "Plane strain\n60 m × 60 m", ha="center", fontsize=8)
    ax.text(0, -57, "Fixed base", ha="center", fontsize=7.5)
    ax.text(-28, -33, "Roller", rotation=90, fontsize=7.5)
    ax.text(26, -33, "Roller", rotation=90, fontsize=7.5)
    for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
        a = np.array([np.cos(theta), np.sin(theta)])
        end = np.array([0, -25.52]) + 6.8*a
        start = np.array([0, -25.52]) + 10.5*a
        ax.annotate("", end, start, arrowprops=dict(arrowstyle="->", color=fs.RED, lw=.7))
    ax.text(-23, -13, "Inward traction", color=fs.RED, fontsize=7.5)
    fig.text(.04, .966, "(a) Section response", fontsize=8.5)

    ax = fig.add_axes([.55, .48, .42, .44])
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.set_axis_off()
    for x, w, color, name in [(.10, .41, "#c9c9c9", "Lining"), (.51, .25, "#e8c49d", "Paste")]:
        ax.add_patch(Rectangle((x, .27), w, .40, facecolor=color, edgecolor="black", lw=.9))
        ax.text(x+w/2, .54, name, ha="center", fontsize=8.5)
    ax.text(.305, .38, "$T(r,t)$", ha="center", fontsize=9)
    ax.text(.635, .42, "$q(r,t)$\n$T(r,t)$", ha="center", va="center", fontsize=8)
    ax.text(.10, .96, "Tunnel air", fontsize=8, ha="center")
    ax.text(.76, .96, "Soil", fontsize=8, ha="center")
    for x, name in [(.10, "$h_a$"), (.76, "$h_s$")]:
        ax.annotate("", (x, .72), (x, .90), arrowprops=dict(arrowstyle="->", lw=.7))
        ax.text(x+.035, .82, name, fontsize=8)
    for x, txt in [(.10,"5.25"),(.51,"5.80"),(.76,"6.02")]:
        ax.plot([x,x],[.22,.27],color="black",lw=.7)
        ax.text(x,.16,txt,ha="center",fontsize=8)
    ax.text(.43,.04,"Radius r (m)",ha="center",fontsize=8)
    fig.text(.55, .966, "(b) Coupled radial heat transfer", fontsize=8.5)
    fig.text(.55,.43,r"Heat release: $c_m\,\dot q$"+"\nStiffness: $E_m(q)$",fontsize=8.5)

    ax=fig.add_axes([.04,.045,.92,.30]); ax.set(xlim=(0,1),ylim=(0,1)); ax.set_axis_off()
    ax.text(0,.96,"(c) Material state to common support requirement",fontsize=8.5)
    centers=[.10,.365,.64,.90]
    text=["Published curves\nHeat flow + modulus", "Continue q and T\nInitial state + boundaries",
          "Radial section response\n"+r"$C_g[E_m(q(r,t))]$", "Choose material\nLoad by deadline"]
    for x,txt in zip(centers,text):
        ax.text(x,.57,txt,ha="center",va="center",fontsize=8)
    for a,b in zip(centers[:-1],centers[1:]):
        ax.annotate("",(b-.115,.57),(a+.115,.57),arrowprops=dict(arrowstyle="->",lw=.8))
    ax.annotate("",(.87,.25),(.35,.25),arrowprops=dict(arrowstyle="->",lw=.8))
    ax.text(.61,.05,"Retain construction time when updating the material state",ha="center",fontsize=8)
    save(fig,1,112,"fig:method",
         "Connect the actual section and radial heat-transfer locations to the state-continuation and common-demand query.",
         ["configs/support_design.yaml","src/grouting_pino/paste_thermal.py","scripts/run_radial_support.py","src/grouting_pino/support_response.py"])


def figure2():
    fig=fs.figure(136,116)
    axes=[fig.add_axes(r) for r in [[.115,.59,.345,.32],[.635,.59,.345,.32],
                                  [.115,.12,.345,.32],[.635,.12,.345,.32]]]
    requirements=pd.read_csv(DATA/"uniform_reaction_requirements.csv")
    required_modulus=SECTION.required_modulus_pa(2.2e-9)/1e9
    for i,(mix,ref) in enumerate(REFS.items()):
        knots=np.r_[ref.heat_age_h,ref.modulus_age_h]
        age=np.unique(np.r_[ref.start_age_h,knots[(knots>ref.start_age_h)&(knots<ref.end_age_h)],ref.end_age_h])
        q=ref.heat_at_age(age)
        style=dict(color=COLORS[i],marker=MARKERS[i],ms=3,markevery=max(1,len(age)//14),lw=.8,mfc="white",mew=.6)
        axes[0].plot(age,ref.rate_at_heat(q),label=mix,**style)
        ma=np.r_[ref.start_age_h,ref.modulus_age_h[(ref.modulus_age_h>ref.start_age_h)&(ref.modulus_age_h<ref.end_age_h)],ref.end_age_h]
        axes[1].plot(ma,ref.modulus_at_heat(ref.heat_at_age(ma))/1e9,
                     color=COLORS[i],marker=MARKERS[i],ms=3,markevery=max(1,len(ma)//14),lw=.8,mfc="white",mew=.6)
        early_q=np.unique(np.r_[q[q<=18000],np.linspace(0,18000,361)])
        axes[2].plot(early_q/1000,ref.modulus_at_heat(early_q)/1e9,
                     color=COLORS[i],lw=1.3,ls=["-","--","-."][i])
        requirement=requirements[(requirements.mix_id==mix)&np.isclose(requirements.target_um_kpa,2.2)].iloc[0]
        qr=requirement.heat_increment_j_kg_cement/1000
        axes[2].plot(qr,required_modulus,marker=MARKERS[i],color=COLORS[i],ms=4,mfc="white",mew=.8)
        axes[2].plot([qr,qr],[0,required_modulus],color=COLORS[i],ls=":",lw=.7)
    for a in axes[:2]:a.set(xlim=(0,55),xlabel="Reference age (h)")
    axes[0].set(ylabel="Heat flow (W/kg cement)")
    axes[1].set(ylabel="Elastic modulus (GPa)",ylim=(0,18))
    axes[2].set(xlabel="Heat increment (kJ/kg cement)",ylabel="Elastic modulus (GPa)",
                xlim=(0,18),ylim=(0,.8),xticks=[0,5,10,15],yticks=[0,.2,.4,.6,.8])
    axes[2].axhline(required_modulus,color=".5",ls="--",lw=.7,zorder=0)
    axes[2].text(17.4,.32,"0.376 GPa",ha="right",fontsize=7.5,color=".35")
    axes[0].legend(frameon=False,fontsize=7.5,loc="upper right")
    e=SECTION.modulus_pa/1e9;c=SECTION.compliance_m_per_pa*1e9
    axes[3].semilogx(e,c,color=fs.BLACK,lw=1.3)
    er=SECTION.required_modulus_pa(2.2e-9)/1e9
    axes[3].plot([e[0],er,er],[2.2,2.2,.0],ls="--",color=fs.RED,lw=.8)
    axes[3].plot(er,2.2,"o",color=fs.RED,ms=4)
    axes[3].text(.055,2.35,"2.2 μm/kPa",color=fs.RED,fontsize=7.5)
    axes[3].text(.48,3.3,"0.376 GPa",color=fs.RED,fontsize=7.5)
    axes[3].set(xlabel="Uniform annulus modulus (GPa)",ylabel="Compliance (μm/kPa)",ylim=(1.2,4.5),xlim=(.05,20),
                xticks=[.1,1,10],xticklabels=["0.1","1","10"])
    for ax,letter,title in zip(axes,"abcd",["20°C calorimetry","20°C EMM-ARM","Heat required for stiffness","Common demand conversion"]):panel(ax,letter,title)
    save(fig,2,116,"fig:material",
         "A common section demand maps to different reaction requirements in the paired material curves.",
         ["data/public_paste_benchmark/heat_flow.csv","data/public_paste_benchmark/modulus.csv",
          "results/support_design/reference_response.npz","results/support_design/uniform_reaction_requirements.csv"])


def figure3():
    fig=fs.figure(136,128)
    axes=[fig.add_axes(r) for r in [[.115,.63,.345,.29],[.635,.63,.345,.29],
                                  [.115,.12,.345,.34],[.635,.12,.345,.34]]]
    history=pd.read_csv(DATA/"radial_histories.csv")
    for i,mix in enumerate(MIXES):
        d=history[(history['case']=="reference")&(history.scenario=="nominal")&(history.mix_id==mix)].sort_values("time_h")
        for ax,quantity in zip(axes[:2],["paste_max_temperature_c","response_um_kpa"]):
            ax.plot(d.time_h,d[quantity],color=COLORS[i],ls=["-","--","-."][i],lw=1.3,label=mix)
            ax.plot(d.time_h.iloc[-1],d[quantity].iloc[-1],marker=MARKERS[i],color=COLORS[i],ms=4,mfc="white")
    axes[0].set(xlim=(0,6.3),ylim=(19,41),xlabel="Construction time (h)",ylabel="Maximum paste temperature (°C)")
    axes[1].set(xlim=(0,6.3),ylim=(1.75,3.5),xlabel="Construction time (h)",ylabel="Compliance (μm/kPa)")
    axes[1].axhline(2.2,color=".5",ls=":",lw=.8)
    axes[1].text(3.5,2.25,"Common demand",fontsize=7.5,color=".35")
    axes[0].legend(loc="upper left",frameon=False,fontsize=7.5)
    updates=pd.read_csv(DATA/"state_updates.csv")
    for i,mix in enumerate(MIXES):
        d=updates[updates.mix_id==mix]
        axes[2].plot(d.state_time_h,d.remaining_time_h,ls="none",marker=MARKERS[i],
                     color=COLORS[i],ms=4,mfc="white",mew=.8)
    axes[2].set(xlim=(-.1,2.1),ylim=(-.15,3.8),xticks=[0,1,2],yticks=[0,1,2,3],
                xlabel="Checkpoint on construction clock (h)",ylabel="Remaining waiting time (h)")
    fs.style_axes(axes[2])
    cold=history[(history['case']=="cold_paste_warm_surroundings")&(history.mix_id=="W48")]
    time=np.linspace(3.8,4.1,301)
    for column,color,style in [("uniform_response_um_kpa",fs.BLACK,"--"),("response_um_kpa",fs.RED,"-")]:
        curves=[]
        for _,group in cold.groupby("scenario"):
            group=group.sort_values("time_h")
            if group.time_h.min()>time[0] or group.time_h.max()<time[-1]:
                raise ValueError("The W48 comparison needs every scenario over the displayed interval")
            curves.append(np.interp(time,group.time_h,group[column]))
        axes[3].plot(time,np.max(curves,axis=0),color=color,ls=style,lw=1.2)
    axes[3].axhline(2.2,color=".5",lw=.8,ls=":")
    axes[3].axvline(4,color=".5",lw=.8,ls=":")
    axes[3].text(3.815,2.177,"Paste 16°C\nLining / soil 30°C",fontsize=7.3)
    axes[3].set(xlim=(3.8,4.1),ylim=(2.165,2.26),xlabel="Construction time (h)",
                ylabel="Largest scenario response (μm/kPa)",xticks=[3.8,3.9,4,4.1],yticks=[2.18,2.2,2.22,2.24,2.26])
    for ax,letter,title in zip(axes,"abcd",["Self-heating at 20°C","Radial support response","Fixed-material updates","W48 near the 4 h deadline"]):
        ax.set_title(f"({letter}) {title}",loc="left",fontsize=8.5,pad=8)
    for ax in [axes[0],axes[1],axes[3]]:fs.style_axes(ax)
    fig.legend(handles=[Line2D([],[],color=fs.BLACK,ls="--",label="Uniform table"),
                        Line2D([],[],color=fs.RED,ls="-",label="Radial FE")],
               loc="lower right",bbox_to_anchor=(.98,.015),ncol=2,frameon=False,fontsize=7.5)
    save(fig,3,128,"fig:histories",
         "State updates retain elapsed construction time; radial stiffness can change a near-boundary loading choice.",
         ["results/support_design/radial_histories.csv","results/support_design/state_updates.csv"])


def decision_code(row):
    best=row.least_cement_all_evaluated_mix
    if row.least_cement_choice_resolved:
        return MIXES.index(best) if best in MIXES else 3
    return 4


def figure4():
    fig=fs.figure(136,136)
    data=pd.read_csv(DATA/"cement_time_frontier.csv").fillna("")
    colors=[fs.BLACK,fs.RED,fs.BLUE,"white","#cccccc"]
    cases=["reference","cold_paste_warm_surroundings"]
    titles=["20°C initial state and soil","Paste 16°C; lining / soil 30°C"]
    for column,(case,title) in enumerate(zip(cases,titles)):
        ax=fig.add_axes([.12+column*.48,.63,.36,.29])
        for assumptions,color,style in [("nominal",fs.BLACK,"-"),("thermal",fs.RED,"--")]:
            d=data[(data['case']==case)&(data.assumption_set==assumptions)
                   &np.isclose(data.target_um_kpa,2.2)]
            last=None
            for row in d.itertuples():
                if row.cement_kg_m3=="" or not row.least_cement_choice_resolved:
                    last=None
                    continue
                y=float(row.cement_kg_m3)
                ax.plot([row.start_h,row.end_h],[y,y],color=color,ls=style,lw=1.2)
                if last is None or last!=y:
                    if last is not None:
                        ax.plot([row.start_h,row.start_h],[last,y],color=color,ls=style,lw=.9)
                    ax.plot(row.start_h,y,marker="o" if assumptions=="nominal" else "s",
                            ms=3.5,mfc="white",mec=color,mew=.7)
                last=y
        masses=data[data.cement_kg_m3!=""].groupby("least_cement_all_evaluated_mix").cement_kg_m3.first()
        for mix,mass in masses.items():
            ax.text(5.65,float(mass)+14,mix,ha="right",fontsize=7.3)
        ax.text(.06,.07,"Demand 2.2 μm/kPa",transform=ax.transAxes,fontsize=7.3)
        ax.set(xlim=(0,6),ylim=(1200,1630),xticks=[0,2,4,6],yticks=[1250,1400,1550],
               xlabel="Latest loading time (h)")
        if column==0:ax.set_ylabel("Minimum cement content (kg/m³)")
        panel(ax,chr(97+column),title)

        ax=fig.add_axes([.12+column*.48,.155,.36,.29])
        d=data[(data['case']==case)&(data.assumption_set=="thermal")]
        targets=np.sort(d.target_um_kpa.unique())
        edges=np.r_[targets[0],(targets[:-1]+targets[1:])/2,targets[-1]]
        polygons={i:[] for i in range(5)}
        for iy,target in enumerate(targets):
            for row in d[d.target_um_kpa==target].itertuples():
                if row.start_h==row.end_h:continue
                polygons[decision_code(row)].append(
                    [(row.start_h,edges[iy]),(row.end_h,edges[iy]),
                     (row.end_h,edges[iy+1]),(row.start_h,edges[iy+1])])
        for category,vertices in polygons.items():
            ax.add_collection(PolyCollection(vertices,facecolors=colors[category],
                              edgecolors="none",linewidths=0,antialiased=False))
        ax.set(xlim=(0,6),ylim=(1.5,3.5),xticks=[0,2,4,6],yticks=[1.5,2,2.5,3,3.5],
               xlabel="Latest loading time (h)")
        if column==0:ax.set_ylabel("Demand (μm/kPa)")
        ax.plot([2,3,4] if column==0 else [4], [2.2]*3 if column==0 else [2.2],
                ls="none",marker="o" if column==0 else "s",ms=4,mfc="white",mec="black",mew=.8)
        panel(ax,chr(99+column),"Choice across all 11 scenarios")
    fig.legend(handles=[Line2D([],[],color=fs.BLACK,ls="-",label="Nominal"),
                        Line2D([],[],color=fs.RED,ls="--",label="All 11 scenarios")],
               loc="center",bbox_to_anchor=(.53,.53),ncol=2,frameon=False,fontsize=8)
    names=["W32","W40","W48","No feasible candidate","Unresolved minimum"]
    fig.legend(handles=[Patch(facecolor=c,edgecolor=".45",lw=.5,label=l) for c,l in zip(colors,names)],
               loc="lower center",bbox_to_anchor=(.53,.012),ncol=3,frameon=False,fontsize=7.5,
               columnspacing=1.6,handlelength=1.4)
    save(fig,4,136,"fig:selection",
         "The cement-time frontier quantifies material savings at radial attainment times; demand and thermal conditions move the material switches.",
         ["results/support_design/cement_time_frontier.csv"])


def main():
    global PREVIEW
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path,
                        help="Optional temporary PNG directory under H:/view-cache/")
    PREVIEW = parser.parse_args().preview_dir
    for draw in [figure1,figure2,figure3,figure4]:draw()
    manifest=dict(project_id="grout-designsweep",manifest_role="Current manuscript figures",
                  style="Origin",template_basis={"figure_ids":[31,19,17],
                    "style_source":"workflows/figures/origin-style.json",
                    "adopted":"Dimensioned sections, aligned response panels and common-axis comparisons; original artwork from project data."},
                  figures=RECORDS)
    (ROOT/"figures").mkdir(exist_ok=True)
    (ROOT/"figures/manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print("Rebuilt four figures and their editable SVGs.")


if __name__=="__main__":main()
