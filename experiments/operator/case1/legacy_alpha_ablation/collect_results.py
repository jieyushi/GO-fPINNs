"""Aggregate completed non-smoke runs by budget/variant, without selecting by test errors."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

METRICS=["relative_l2","amplitude_relative_l2","phase_mae_radians",
         "mean_absolute_complex_error","max_absolute_complex_error"]
DIAGNOSTIC_METRICS=['phase_mae_all_radians']
FORMATS={'R2_alpha_ablation_softmax_noise_local_xt','R2_alpha_full_softmax_noise_local_xt'}
RESOURCE_KEYS=["total_updates","residual_point_evaluations","optimization_seconds",
               "wall_seconds_total_checkpointed","peak_training_allocated_MiB",
               "peak_training_reserved_MiB","process_peak_rss_MiB",
               "prediction_parameters","loss_weight_parameters","total_optimizer_parameters"]
DISPLAY={"full":"Full Softmax+noise","no_enrichment":"No enrichment","direct_operator":"Direct operator",
    "no_grid_enrichment":"No grid enrichment","no_parameter_enrichment":"No alpha enrichment",
    "no_consistency":"No consistency","no_dual_tanh":"No dual-tanh","no_gradient":"No gradient",
    "no_soe":"w/o SOE","no_adaptive_weights":"Uniform weights","no_fourier":"No Fourier","plain_operator":"Plain operator"}

def read_rows(path):
    with Path(path).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))

def write_rows(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def stats(values):
    values=[float(x) for x in values if x is not None and math.isfinite(float(x))]
    return (statistics.mean(values),statistics.stdev(values) if len(values)>1 else None) if values else (None,None)

def cell(mean,std,scale=1.,integer=False):
    if mean is None:return "--"
    if integer and (std is None or std==0):return str(round(mean))
    return "$%.2f%s$"%(mean/scale,("\\pm %.2f"%(std/scale)) if std is not None else "")

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--expected-seeds",default="1234,2024,3407")
    p.add_argument('--expected-variants',default=None,help='Optional coverage list; default checks all eleven variants.')
    p.add_argument('--include-smoke',action='store_true',help='Functional verification only; never use these results as accuracy data.')
    a=p.parse_args(argv);root=a.root.resolve()
    expected={int(x) for x in a.expected_seeds.split(",")}
    expected_names=a.expected_variants.split(',') if a.expected_variants else list(DISPLAY)
    if not expected or min(expected)<0 or len(set(expected_names))!=len(expected_names) or any(n not in DISPLAY for n in expected_names):
        p.error('Invalid expected variants/seeds.')
    specs={};represented=set()
    for manifest in sorted(root.rglob("manifest_*.json")):
        if "_screening" in manifest.parts:continue
        data=json.loads(manifest.read_text())
        for item in data["entries"]:
            key=(item["budget_mode"],item["variant"],item["seed"])
            specs[key]=Path(item["path"]);represented.add(Path(item["path"]).resolve())
    for path in sorted(root.rglob("summary.json")):
        if "_screening" in path.parts or path.parent.resolve() in represented:continue
        summary=json.loads(path.read_text())
        if summary.get("format") not in FORMATS:continue
        key=(summary.get("budget_mode","matched_steps"),summary.get("variant","full"),summary["seed"])
        if key in specs and specs[key].resolve()!=path.parent.resolve():
            raise ValueError("Duplicate run identity; use separate roots for alternative configurations: "+str(key))
        specs[key]=path.parent
    runs=[];alpha_rows=[];warnings=[];hardware_rows=[]
    for (mode,name,seed),path in sorted(specs.items()):
        if seed not in expected:continue
        if not (path/"summary.json").exists():continue
        summary=json.loads((path/"summary.json").read_text())
        allowed={'TRAINING_COMPLETED_ACCURACY_UNVERIFIED','SMOKE_ONLY'} if a.include_smoke else {'TRAINING_COMPLETED_ACCURACY_UNVERIFIED'}
        if summary.get('format') not in FORMATS:raise ValueError('A manifest references incompatible old results: '+str(path))
        if summary.get("status") not in allowed:continue
        config=json.loads((path/"config.json").read_text())
        if config.get('variant','full')!=name or summary['seed']!=seed:
            raise ValueError('Run identity differs from manifest/directory: '+str(path))
        if name not in ('no_adaptive_weights','plain_operator') and config.get('weighting',{}).get('all_stages')!='softmax_noise':
            raise ValueError('Active weight rule mismatch: '+str(path))
        values=summary.get("test_macro_over_unseen_alphas")
        if not values:continue
        per_alpha=read_rows(path/"test_per_alpha.csv")
        r=summary["resources"];arch=config["architecture"]
        mib=lambda key:None if r.get(key) is None else float(r[key])/2**20
        row=dict(budget_mode=mode,variant=name,seed=seed,path=str(path),
            **values,total_updates=summary["total_updates"],
            residual_point_evaluations=summary["residual_point_evaluations"],
            global_max_absolute_complex_error=max(float(x["max_absolute_complex_error"]) for x in per_alpha),
            optimization_seconds=r["optimization_seconds"],
            wall_seconds_total_checkpointed=r["wall_seconds_total_checkpointed"],
            peak_training_allocated_MiB=mib("peak_training_allocated_bytes"),
            peak_training_reserved_MiB=mib("peak_training_reserved_bytes"),
            process_peak_rss_MiB=mib("process_peak_rss_bytes"),
            prediction_parameters=arch["prediction_parameters"],
            loss_weight_parameters=arch["loss_weight_parameters"],
            total_optimizer_parameters=arch["total_optimizer_parameters"],
            validation_best=summary["validation_best"],selected_stage=summary["selected_stage"],
            selected_epoch=summary["selected_epoch"],initial_lr=config["args"]["initial_lr"])
        runs.append(row)
        for item in per_alpha:
            alpha_rows.append(dict(budget_mode=mode,variant=name,seed=seed,
                                  **{k:float(value) for k,value in item.items()}))
        hardware_rows.append(dict(budget_mode=mode,variant=name,seed=seed,hardware=config["hardware"],
                                  architecture=arch,optimization=config["optimization"],
                                  history_inventory=[h["history_state_inventory"] for h in summary["stages"]],
                                  phase_updates=[h["updates"] for h in summary["stages"]]))
    grouped={}
    for row in runs:grouped.setdefault((row["budget_mode"],row["variant"]),[]).append(row)
    aggregates=[]
    for (mode,name),rows in sorted(grouped.items()):
        actual={r["seed"] for r in rows};entry=dict(budget_mode=mode,variant=name,n_runs=len(rows),
            seeds=",".join(str(x) for x in sorted(actual)),complete_seed_set=actual==expected)
        if actual!=expected:warnings.append("%s/%s missing seeds: %s"%(mode,name,sorted(expected-actual)))
        for key in METRICS+DIAGNOSTIC_METRICS+["global_max_absolute_complex_error"]+RESOURCE_KEYS:
            mean,std=stats([r[key] for r in rows]);entry[key+"_mean"]=mean;entry[key+"_std"]=std
        aggregates.append(entry)
    grouped_alpha={}
    for row in alpha_rows:grouped_alpha.setdefault((row["budget_mode"],row["variant"],row["alpha"]),[]).append(row)
    alpha_aggregates=[]
    for (mode,name,alpha),rows in sorted(grouped_alpha.items()):
        entry=dict(budget_mode=mode,variant=name,alpha=alpha,n_runs=len(rows))
        for key in METRICS+DIAGNOSTIC_METRICS:
            mean,std=stats([r[key] for r in rows]);entry[key+"_mean"]=mean;entry[key+"_std"]=std
        alpha_aggregates.append(entry)
    for mode in sorted({r["budget_mode"] for r in runs}):
        infos=[x for x in hardware_rows if x["budget_mode"]==mode]
        fingerprints={json.dumps({k:x["hardware"].get(k) for k in ["gpu","torch","cuda_runtime","torch_threads","tf32_matmul_allowed"]},sort_keys=True) for x in infos}
        if len(fingerprints)>1:warnings.append(mode+": hardware/software differ; do not pool speed or memory claims without stratifying.")
        paths=[Path(r["path"])/"config.json" for r in runs if r["budget_mode"]==mode]
        configs=[json.loads(p.read_text()) for p in paths]
        protocols={json.dumps([c["validation_alphas"],c["test_alphas"],c["args"]["eval_points"],
                    c["args"]["alpha_order"],c["args"]["grids"],c["args"]["boundary_points"],
                    c["args"].get('noise_amplitude'),c["args"].get('weight_lower_bound'),
                    c["source_hashes"]["operator_alpha_core.py"],c["source_hashes"]["training_support.py"]],sort_keys=True) for c in configs}
        if len(protocols)>1:warnings.append(mode+": source/evaluation/reference-space conditions differ; inspect before comparison.")
        actual={(r['variant'],r['seed']) for r in runs if r['budget_mode']==mode}
        for name in expected_names:
            missing=sorted(seed for seed in expected if (name,seed) not in actual)
            if missing:warnings.append('%s/%s missing requested runs: %s'%(mode,name,missing))
    output=root/"aggregated";output.mkdir(parents=True,exist_ok=True)
    write_rows(output/"per_run.csv",runs);write_rows(output/"mean_std.csv",aggregates)
    write_rows(output/"per_alpha_mean_std.csv",alpha_aggregates)
    (output/"hardware_architecture_resources.json").write_text(json.dumps(hardware_rows,indent=2),encoding="utf-8")
    (output/"checks.json").write_text(json.dumps(dict(expected_seeds=sorted(expected),warnings=warnings,
        metric_aggregation="Mean of the 20 per-alpha metrics within each run; sample SD across independent seeds (ddof=1).",
        maxae_note="max_absolute_complex_error is the alpha-macro of per-alpha maxima. global_max_absolute_complex_error is the maximum over all test points/alphas.",
        phase_main='phase_mae_radians: masked wrapped MAE',
        phase_diagnostic='phase_mae_all_radians: wrapped MAE on all points',
        expected_variants=expected_names,complete_requested_coverage=bool(runs) and not any('missing requested runs' in w for w in warnings),
        functional_smoke_included=a.include_smoke,
        scope="GPU PyTorch allocator peaks during loss/autograd/backward/Adam; CPU RSS is separate. SMOKE and tuning-only runs excluded."),indent=2))
    tex=["% Generated from completed runs. Requires booktabs; no fabricated missing results."]
    for mode in sorted({r["budget_mode"] for r in aggregates}):
        group=[r for r in aggregates if r["budget_mode"]==mode]
        tex+=["\\begin{table*}[t]","\\centering\\small\\setlength{\\tabcolsep}{3pt}",
            "\\caption{Operator comparisons under "+mode.replace("_"," ")+
            ". Errors are parameter-macro means, followed by sample standard deviations across seeds; $n$ gives the number of completed runs. MaxAE denotes the mean of per-parameter maxima.}",
            "\\begin{tabular}{lcccccc}","\\toprule",
            "Variant & $n$ & \\shortstack{Rel. $L_2$\\\\$(\\times10^{-2})$} & \\shortstack{Amp. Rel. $L_2$\\\\$(\\times10^{-2})$} & \\shortstack{Phase MAE\\\\$(\\times10^{-2})$ rad} & \\shortstack{MAE\\\\$(\\times10^{-3})$} & \\shortstack{MaxAE\\\\$(\\times10^{-3})$} \\\\","\\midrule"]
        for row in group:
            tex.append(DISPLAY[row["variant"]]+" & "+str(row["n_runs"])+" & "+
                " & ".join(cell(row[k+"_mean"],row[k+"_std"],scale) for k,scale in zip(METRICS,[1e-2,1e-2,1e-2,1e-3,1e-3]))+" \\\\")
        tex+=["\\bottomrule","\\end{tabular}","\\end{table*}",""]
        tex+=["\\begin{table*}[t]","\\centering\\small\\setlength{\\tabcolsep}{3pt}",
            "\\caption{Measured training resources under "+mode.replace("_"," ")+
            ". GPU peaks refer to optimization steps including automatic differentiation and backward propagation. CPU runs have no GPU-memory measurement.}",
            "\\begin{tabular}{lccccc}","\\toprule",
            "Variant & \\shortstack{Predictor\\\\params} & Updates & \\shortstack{Optimization\\\\time (s)} & \\shortstack{Peak allocated\\\\(MiB)} & \\shortstack{Peak reserved\\\\(MiB)} \\\\","\\midrule"]
        for row in group:
            keys=["prediction_parameters","total_updates","optimization_seconds","peak_training_allocated_MiB","peak_training_reserved_MiB"]
            tex.append(DISPLAY[row["variant"]]+" & "+
                " & ".join(cell(row[k+"_mean"],row[k+"_std"],integer=k in ['prediction_parameters','total_updates']) for k in keys)+" \\\\")
        tex+=["\\bottomrule","\\end{tabular}","\\end{table*}",""]
    (output/"results_tables.tex").write_text("\n".join(tex),encoding="utf-8")
    print(json.dumps(dict(completed_runs=len(runs),groups=len(aggregates),output=str(output),warnings=warnings),indent=2))

if __name__=="__main__":main()
