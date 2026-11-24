"""Eleven Softmax+noise operator comparisons. Old Softplus results are rejected."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import ablation_support as v
import train_ablation as t
import train_alpha_operator as original
import training_support as s

ROOT=Path(__file__).resolve().parent
DONE={"TRAINING_COMPLETED_ACCURACY_UNVERIFIED","SMOKE_ONLY","VALIDATION_ONLY_COMPLETED"}

def selected_variants(preset,value):
    result=value.split(",") if value else (v.PRIMARY if preset=="primary" else v.COMPONENTS if preset=="components" else list(v.VARIANTS))
    if len(set(result))!=len(result) or any(x not in v.VARIANTS for x in result):
        raise ValueError("Unknown or duplicate variant.")
    return result

def reuse_original(path,expected):
    path=Path(path)
    if not (path/"summary.json").exists():raise FileNotFoundError("Completed V6 reference missing: "+str(path))
    config=json.loads((path/"config.json").read_text());summary=json.loads((path/"summary.json").read_text())
    if config.get("format") not in {original.FORMAT,t.FORMAT} or summary.get("status")!="TRAINING_COMPLETED_ACCURACY_UNVERIFIED":
        raise ValueError("Reuse requires a completed NEW Softmax+noise Full, not an old Softplus run.")
    is_new=config["format"]==t.FORMAT
    if is_new and (config.get("variant")!="full" or config["args"]["budget_mode"]=="matched_time"):
        raise ValueError("Only the fixed-budget full variant may be reused as reference.")
    hashes=t.source_hashes() if is_new else original.source_hashes()
    for name,digest in hashes.items():
        if config["source_hashes"].get(name)!=digest:raise ValueError("V6 source mismatch: "+name)
    for key,value in vars(expected).items():
        if key not in {"output","resume"} and config["args"].get(key)!=value:
            raise ValueError("V6 reference configuration mismatch: "+key)
    if summary["total_updates"]!=sum(expected.stage_epochs):raise ValueError("V6 reference is incomplete.")
    if config.get('weighting',{}).get('all_stages')!='softmax_noise':
        raise ValueError('Full reference does not use Softmax+noise in every stage.')
    required=["test_per_alpha.csv","resources.json","final.pt"]
    if any(not (path/name).exists() for name in required):raise ValueError("Reference outputs are incomplete.")
    return path.resolve()

def run_job(args,seed,variant,mode,output,extra,phase_seconds=None):
    command=[sys.executable,"-u",str(ROOT/"train_ablation.py"),"--device",args.device,
        "--seed",str(seed),"--variant",variant,"--budget-mode",mode,
        "--epochs-per-stage",str(args.epochs_per_stage),"--initial-lr",str(args.initial_lr),
        "--output",str(output)]
    if args.resume:command+=["--resume"]
    if args.smoke:command+=["--smoke"]
    if args.validation_only:command+=["--validation-only"]
    if phase_seconds:command+=["--phase-seconds",",".join(format(x,".17g") for x in phase_seconds)]
    command+=extra
    log=args.output/"logs"/mode/variant/("seed_%d.log"%seed);log.parent.mkdir(parents=True,exist_ok=True)
    print("Running %s/%s/seed=%d -> %s"%(mode,variant,seed,output),flush=True)
    with log.open("a",encoding="utf-8") as handle:
        handle.write("\nCOMMAND "+json.dumps(command)+"\n");handle.flush()
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end="",flush=True);handle.write(line);handle.flush()
        code=process.wait()
    if code:raise RuntimeError("Run failed; log=%s. Use the same command with --resume after resolving the cause."%log)
    data=json.loads((output/"summary.json").read_text())
    if data["status"] not in DONE:raise RuntimeError("Training did not finish: "+str(output))
    with log.open("a",encoding="utf-8") as handle:
        handle.write("\nFINISHED "+json.dumps(dict(status=data["status"],total_updates=data["total_updates"],
            summary=str(output/"summary.json"),resources=str(output/"resources.json")))+"\n")
    return output.resolve()

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset",choices=["primary","components","all"],default="primary")
    p.add_argument("--variants",default=None)
    p.add_argument("--seeds",default="1234,2024,3407")
    p.add_argument("--device",default="cuda")
    p.add_argument("--epochs-per-stage",type=int,default=5000)
    p.add_argument("--initial-lr",type=float,default=1e-3)
    p.add_argument("--budget-mode",choices=["matched_steps","matched_point_work","matched_time"],default="matched_steps")
    p.add_argument("--output",type=Path,default=Path("runs/R2_alpha_softmax_noise_grid6810"))
    p.add_argument("--reuse-full-from",type=Path,help="Completed NEW Softmax+noise standalone Full root containing seed_N/. Old Softplus results are rejected.")
    p.add_argument("--require-full",action="store_true",help="Verify all requested completed Softmax Full seeds before any ablation; use after the separate Full queue.")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--validation-only",action="store_true")
    a,extra=p.parse_known_args(argv)
    protected={"--seed","--variant","--output","--budget-mode","--phase-seconds"}
    if any(x.split("=")[0] in protected for x in extra):p.error("Cannot override suite-owned run arguments.")
    seeds=[int(x) for x in a.seeds.split(",")]
    if not seeds or min(seeds)<0 or len(set(seeds))!=len(seeds):p.error("Seeds must be distinct and nonnegative.")
    names=selected_variants(a.preset,a.variants);a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    references={}
    if a.require_full:
        if a.smoke or a.validation_only:p.error('--require-full is for completed accuracy-budget Full runs.')
        if a.reuse_full_from:p.error('Use either --require-full or --reuse-full-from, not both.')
        for seed in seeds:
            expected=original.parse_args(['--device',a.device,'--seed',str(seed),
                '--epochs-per-stage',str(a.epochs_per_stage),'--initial-lr',str(a.initial_lr)]+extra)
            candidate=a.output/'matched_steps'/'full'/('seed_%d'%seed)
            references[seed]=reuse_original(candidate,expected)
    entries=[]
    for seed in seeds:
        reference=references.get(seed)
        if a.reuse_full_from:
            if a.smoke or a.validation_only:p.error("Do not reuse accuracy runs for smoke or validation-only tuning.")
            expected=original.parse_args(["--device",a.device,"--seed",str(seed),
                "--epochs-per-stage",str(a.epochs_per_stage),"--initial-lr",str(a.initial_lr)]+extra)
            reference=reuse_original(a.reuse_full_from/("seed_%d"%seed),expected)
        if a.budget_mode=="matched_time" and reference is None:
            reference=run_job(a,seed,"full","matched_steps",
                a.output/"matched_steps"/"full"/("seed_%d"%seed),extra)
        if a.budget_mode=="matched_point_work" and reference is None:
            candidate=a.output/"matched_steps"/"full"/("seed_%d"%seed)
            if (candidate/"summary.json").exists():
                expected=original.parse_args(["--device",a.device,"--seed",str(seed),
                    "--epochs-per-stage",str(a.epochs_per_stage),"--initial-lr",str(a.initial_lr)]+extra)
                reference=reuse_original(candidate,expected)
        times=None
        if a.budget_mode=="matched_time":
            summary=json.loads((reference/"summary.json").read_text())
            times=[float(x["optimization_seconds"]) for x in summary["stages"]]
        for name in names:
            if name=="full" and reference is not None:
                destination=reference
                print("Validated full reference:",destination,flush=True)
            else:
                destination=run_job(a,seed,name,a.budget_mode,
                    a.output/a.budget_mode/name/("seed_%d"%seed),extra,times)
            entries.append(dict(variant=name,seed=seed,budget_mode=a.budget_mode,path=str(destination),
                                reference_full_path=str(reference) if reference else None))
        manifest=a.output/("manifest_%s_seed_%d.json"%(a.budget_mode,seed))
        previous=json.loads(manifest.read_text()).get("entries",[]) if manifest.exists() else []
        merged={(x["budget_mode"],x["variant"],x["seed"]):x for x in previous}
        merged.update({(x["budget_mode"],x["variant"],x["seed"]):x for x in entries if x["seed"]==seed})
        s.atomic_json(manifest,dict(entries=list(merged.values()),
                                   common_command_args=dict(vars(a),output=str(a.output),reuse_full_from=str(a.reuse_full_from) if a.reuse_full_from else None),
                                   extra_training_args=extra))
    if not a.validation_only:
        subprocess.run([sys.executable,str(ROOT/"collect_results.py"),"--root",str(a.output),
                        "--expected-seeds",a.seeds,"--expected-variants",','.join(names)],check=True)

if __name__=="__main__":main()
