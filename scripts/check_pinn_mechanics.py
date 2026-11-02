"""Independent physical checks of the trained mixed section neural field."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from grouting_pino.pinn_mechanics import (SectionPINN, DOMAINS, RADIUS, PRESSURE,
    DISPLACEMENT, LENGTH, CENTER_Y, NU, annulus_points, sample_domains)
from grouting_pino.pinn_thermal import load_thermal
from grouting_pino.paste_reference import load_paste_references


def load_section(directory):
    config = json.loads((directory/"training.json").read_text())["config"]
    reference = load_paste_references(ROOT/"data/public_paste_benchmark")[config["mix_id"]]
    if config["time_h"] == 0:
        modulus = lambda r: np.full_like(r, reference.modulus_at_heat(0))
    else:
        thermal_path = config.get("thermal_model", f"results/pinn_support/{config['mix_id']}/thermal/model.pt")
        thermal = load_thermal(ROOT/thermal_path,
                               ROOT/"data/public_paste_benchmark")
        modulus = lambda r: thermal.query("paste",np.clip(r,5.8,6.02),config["time_h"])["modulus_pa"]
    return SectionPINN.load(directory/"model.npz",modulus)


def quadrature(domain, order=80):
    nodes, weights = leggauss(order)
    angle = (nodes+1)*np.pi/2
    angle_weights = weights*np.pi/2
    if domain == "soil":
        # Split at the rectangular corners so quadrature never integrates
        # across a kink in the radial extent of the soil domain.
        cuts=[0.,np.arctan2(30.,-CENTER_Y),np.pi-np.arctan2(30.,60+CENTER_Y),np.pi]
        angle=np.concatenate([a+(nodes+1)*(b-a)/2 for a,b in zip(cuts[:-1],cuts[1:])])
        angle_weights=np.concatenate([weights*(b-a)/2 for a,b in zip(cuts[:-1],cuts[1:])])
        upper = np.minimum(30/np.sin(angle),np.where(np.cos(angle)>0,
            -CENTER_Y/np.maximum(np.cos(angle),1e-15),
            (60+CENTER_Y)/np.maximum(-np.cos(angle),1e-15)))
        lower = 6.02
    else:
        lower, upper = RADIUS[domain]
    radius = lower + (np.asarray(upper)[...,None]-lower)*(nodes+1)/2
    radius = np.broadcast_to(radius,(len(angle),order))
    theta = np.broadcast_to(angle[:,None],radius.shape)
    weight = radius*(np.asarray(upper)[...,None]-lower)/2*weights[None,:]*angle_weights[:,None]*2
    return annulus_points(radius,theta),weight.ravel()


def constitutive(model,domain,xy,f):
    E = model.modulus(domain,xy)
    mu = E/(2*(1+NU))
    lam = E*NU/((1+NU)*(1-2*NU))
    tr = f["ux_x"]+f["uy_y"]
    return dict(sxx=lam*tr+2*mu*f["ux_x"],syy=lam*tr+2*mu*f["uy_y"],
                sxy=mu*(f["ux_y"]+f["uy_x"]),szz=lam*tr)


def assess(model,order=80):
    stats={"domain":{},"interfaces":{},"quadrature_order":order}
    points=sample_domains(model.config,model.config["seed"]+8901)
    for d,xy in points.items():
        f=model.query(d,xy,True)
        calc=constitutive(model,d,xy,f)
        equilibrium=np.c_[f["sxx_x"]+f["sxy_y"],f["sxy_x"]+f["syy_y"]]
        con=np.c_[f["sxx"]-calc["sxx"],f["syy"]-calc["syy"],
                  np.sqrt(2)*(f["sxy"]-calc["sxy"])]
        stats["domain"][d]={"equilibrium_rms_pa_m":float(np.sqrt(np.mean(equilibrium**2))),
              "constitutive_rms_pa":float(np.sqrt(np.mean(con**2))),
              "stress_rms_pa":float(np.sqrt(np.mean(np.c_[f["sxx"],f["syy"],f["sxy"]]**2)))}
        for name, residual, scale in [("equilibrium", equilibrium, PRESSURE/LENGTH),
                                      ("constitutive", con, PRESSURE)]:
            absolute = np.linalg.norm(residual, axis=1) / scale
            index = int(absolute.argmax())
            stats["domain"][d].update({
                name+"_normalized_p99": float(np.quantile(absolute, .99)),
                name+"_normalized_max": float(absolute[index]),
                name+"_maximum_xy_m": xy[index].tolist()})
    theta=np.pi*(np.arange(601)+0.37)/601
    nx,ny=np.sin(theta),np.cos(theta)
    for r,inner,outer in [(5.8,"lining","paste"),(6.02,"paste","soil")]:
        xy=annulus_points(r,theta)
        a,b=model.query(inner,xy),model.query(outer,xy)
        ux,uy=a["ux"]-b["ux"],a["uy"]-b["uy"]
        tx=(a["sxx"]-b["sxx"])*nx+(a["sxy"]-b["sxy"])*ny
        ty=(a["sxy"]-b["sxy"])*nx+(a["syy"]-b["syy"])*ny
        target=-PRESSURE if inner=="paste" else 0
        stats["interfaces"][inner+"_"+outer]={
            "displacement_jump_max_m":float(np.max(np.hypot(ux,uy))),
            "traction_error_rms_pa":float(np.sqrt(np.mean((tx-target*nx)**2+(ty-target*ny)**2))),
            "traction_error_max_pa":float(np.max(np.hypot(tx-target*nx,ty-target*ny))),
            "radial_traction_jump_mean_pa":float(np.mean(tx*nx+ty*ny)),
            "tangential_traction_jump_rms_pa":float(np.sqrt(np.mean((tx*ny-ty*nx)**2)))}
    inner=model.query("lining",annulus_points(5.25,theta))
    stats["intrados_traction_rms_pa"]=float(np.sqrt(np.mean(
        (inner["sxx"]*nx+inner["sxy"]*ny)**2+(inner["sxy"]*nx+inner["syy"]*ny)**2)))
    x=np.linspace(-30,30,1201)
    surface=model.query("soil",np.c_[x,np.zeros_like(x)])
    stats["surface_traction_rms_pa"]=float(np.sqrt(np.mean(surface["syy"]**2+surface["sxy"]**2)))
    stats["surface_settlement_um_kpa"]={"min":float(np.min(-surface["uy"])*1e9/PRESSURE),
        "max":float(np.max(-surface["uy"])*1e9/PRESSURE),
        "peak_x_m":float(x[np.argmax(-surface["uy"])])}
    peak=int(np.argmax(-surface["uy"]))
    bracket=(x[max(0,peak-1)],x[min(len(x)-1,peak+1)])
    refined=minimize_scalar(lambda at:float(model.query("soil",np.array([[at,0.]]))["uy"][0]),
                            bounds=bracket,method="bounded",options={"xatol":1e-5})
    stats["surface_settlement_um_kpa"].update(
        refined_peak_x_m=float(refined.x),refined_max=float(-refined.fun*1e9/PRESSURE))
    side=model.query("soil",np.c_[np.full(601,30),np.linspace(-60,0,601)])
    base=model.query("soil",np.c_[np.linspace(0,30,601),np.full(601,-60)])
    stats["side_displacement_max_m"]=float(np.max(np.abs(side["ux"])))
    stats["base_displacement_max_m"]=float(np.max(np.hypot(base["ux"],base["uy"])))
    stats["side_tangential_traction_rms_pa"]=float(np.sqrt(np.mean(side["sxy"]**2)))
    strain_energy,mixed_work=0.,0.
    for d in DOMAINS:
        xy,weight=quadrature(d,order)
        f=model.query(d,xy,True)
        calc=constitutive(model,d,xy,f)
        strain_energy+=float(weight@(calc["sxx"]*f["ux_x"]+calc["syy"]*f["uy_y"]
                                     +calc["sxy"]*(f["ux_y"]+f["uy_x"])))
        mixed_work+=float(weight@(f["sxx"]*f["ux_x"]+f["syy"]*f["uy_y"]
                                   +f["sxy"]*(f["ux_y"]+f["uy_x"])))
    node,weight=leggauss(order*2)
    angle=(node+1)*np.pi/2
    xy=annulus_points(6.02,angle)
    f=model.query("soil",xy)
    work=float(-PRESSURE*6.02*np.pi*(weight@(np.sin(angle)*f["ux"]+np.cos(angle)*f["uy"])))
    stats["virtual_work"]={"elastic_j_m":strain_energy,"mixed_j_m":mixed_work,"load_j_m":work,
                           "elastic_relative_defect":(strain_energy-work)/work,
                           "mixed_relative_defect":(mixed_work-work)/work}
    # Resultant of each closed annular sector is a local force-balance probe.
    local={}
    for d in DOMAINS:
        lo,hi=RADIUS[d]
        if d=="soil": hi=15.0
        lo,hi=lo+.1*(hi-lo),hi-.1*(hi-lo)
        for label,aa,bb in [("crown",0.07,0.8),("flank",1.1,1.9),("invert",2.3,3.07)]:
            angle=aa+(node+1)*(bb-aa)/2
            resultant=np.zeros(2)
            absolute=0.
            for radius,sign in [(lo,-1),(hi,1)]:
                f=model.query(d,annulus_points(radius,angle))
                tx=sign*(f["sxx"]*np.sin(angle)+f["sxy"]*np.cos(angle))
                ty=sign*(f["sxy"]*np.sin(angle)+f["syy"]*np.cos(angle))
                ds=weight*radius*(bb-aa)/2
                resultant+=np.array([ds@tx,ds@ty])
                absolute+=float(ds@np.hypot(tx,ty))
            r=lo+(node+1)*(hi-lo)/2
            for ang,sign in [(aa,-1),(bb,1)]:
                f=model.query(d,annulus_points(r,ang))
                tx=sign*(f["sxx"]*np.cos(ang)-f["sxy"]*np.sin(ang))
                ty=sign*(f["sxy"]*np.cos(ang)-f["syy"]*np.sin(ang))
                ds=weight*(hi-lo)/2
                resultant+=np.array([ds@tx,ds@ty])
                absolute+=float(ds@np.hypot(tx,ty))
            local[d+"_"+label]={"force_defect_n_m":resultant.tolist(),
                "defect_over_absolute_boundary_force":float(np.linalg.norm(resultant)/absolute)}
    stats["local_force"]=local
    stats["evaluation"] = {"domain_points": model.config["domain_points"],
        "domain_seed": model.config["seed"]+8901, "interface_points": len(theta),
        "equilibrium_scale_pa_m": PRESSURE/LENGTH, "constitutive_scale_pa": PRESSURE,
        "displacement_scale_m": DISPLACEMENT, "independent_of_training": True}
    return stats


def main():
    p=argparse.ArgumentParser()
    p.add_argument("directory",type=Path)
    p.add_argument("--order",type=int,default=80)
    args=p.parse_args()
    with threadpool_limits(limits=4):
        model=load_section(args.directory)
        stats=assess(model,args.order)
    (args.directory/"validation.json").write_text(json.dumps(stats,indent=2)+"\n")
    print(json.dumps(stats,indent=2))


if __name__=="__main__":
    main()
