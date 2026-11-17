"""Numerical fixtures and strict, code-free patch application."""
import copy
import json
import math
import numpy as np

TASKS=("action_gain","add_drag","semantic_damping")


def forward(weights,x):
    return np.maximum(x@np.asarray(weights["W1"]).T+weights["b1"],0)@np.asarray(weights["W2"])+weights["b2"][0]


def quantize(weights):
    return {k:np.round(np.asarray(v,dtype=float),8).tolist() for k,v in weights.items()}


def gauge(weights,seed,transformed):
    w={k:np.asarray(v,dtype=float).copy() for k,v in weights.items()}
    if transformed:
        rng=np.random.default_rng(700000+seed)
        p=rng.permutation(len(w["b1"]))
        scale=np.exp(rng.uniform(-np.log(3),np.log(3),len(p)))
        w["W1"]=w["W1"][p]*scale[:,None]
        w["b1"]=w["b1"][p]*scale
        w["W2"]=w["W2"][p]/scale
    return quantize(w)


def apply_patch(weights,patches):
    arrays={k:np.asarray(v,dtype=float).copy() for k,v in weights.items()}
    seen=set()
    if not isinstance(patches,list):
        raise ValueError("patches must be a list")
    for patch in patches:
        if set(patch)!={"tensor","index","value"}:
            raise ValueError("Unexpected patch fields")
        name,index,value=patch["tensor"],patch["index"],patch["value"]
        if name not in arrays or not isinstance(index,list) or len(index)!=arrays[name].ndim:
            raise ValueError("Invalid tensor/index rank")
        if any(type(i) is not int or i<0 or i>=s for i,s in zip(index,arrays[name].shape)):
            raise ValueError("Invalid index")
        if type(value) not in (int,float) or not math.isfinite(value) or abs(value)>1000:
            raise ValueError("Invalid parameter value")
        key=(name,tuple(index))
        if key in seen:
            raise ValueError("Duplicate parameter replacement")
        seen.add(key);arrays[name][tuple(index)]=value
    return {k:v.tolist() for k,v in arrays.items()}


def compile_edit(weights,task,coefficient=None):
    result=copy.deepcopy(weights)
    if task=="action_gain":
        for row in result["W1"]:
            row[2]*=1.5
        return result
    dormant=[i for i,v in enumerate(result["W2"]) if v==0]
    if len(dormant)<2:
        return None
    coefficient=.2 if task=="add_drag" else coefficient
    if coefficient is None:
        raise ValueError("Missing drag coefficient")
    i,j=dormant[:2]
    result["W1"][i]=[0.,1.,0.];result["b1"][i]=0.;result["W2"][i]=-coefficient
    result["W1"][j]=[0.,-1.,0.];result["b1"][j]=0.;result["W2"][j]=coefficient
    return result


def response_schema():
    return {"type":"object","properties":{
        "summary":{"type":"string"},
        "estimated_c":{"anyOf":[{"type":"number"},{"type":"null"}]},
        "patches":{"type":"array","items":{"type":"object","properties":{
            "tensor":{"type":"string","enum":["W1","b1","W2","b2"]},
            "index":{"type":"array","items":{"type":"integer"}},
            "value":{"type":"number"}},"required":["tensor","index","value"],"additionalProperties":False}}},
        "required":["summary","estimated_c","patches"],"additionalProperties":False}


def make_prompt(weights,task):
    request={
        "action_gain":"Implement f_new(q,v,u)=f_old(q,v,1.5*u), preserving the entire original function apart from this input transformation.",
        "add_drag":"Implement f_new(q,v,u)=f_old(q,v,u)-0.2*v. Preserve all other original behavior. Architecture and hidden width must stay fixed.",
        "semantic_damping":"Infer the unknown positive linear viscous damping coefficient c from these weights, then double that coefficient ONLY: f_new(x)=f_old(x)-c*v. Preserve the quadratic drag, restoring response, action response, and other original behavior. Report your estimated_c."}[task]
    return ("You are performing direct weight-space programming. Reason from the raw numerical weights "
        "below; you have no tools, execution, examples, probes, or feedback. Produce explicit numerical "
        "parameter replacements, not code or a training recipe. You may change existing tensor entries "
        "but cannot change architecture. Unlisted entries remain identical.\n"
        "Inputs x=[q,v,u] are physical position, velocity, applied force. Output is acceleration. "
        "This network was trained to approximate the family -k*sin(q)-c*v-a*v*abs(v)+g*u, "
        "with positive unknown coefficients. Coefficient values are NOT supplied. "
        "The intended domain is [-1,1]^3.\n"
        "Architecture: z=ReLU(W1*x+b1); f(x)=sum_i W2[i]*z[i]+b2[0]. "
        f"W1 shape [{len(weights['b1'])},3]; b1 and W2 length {len(weights['b1'])}; b2 length 1. "
        "All indices are zero-based. Some representations may contain unused hidden capacity.\n"
        "Request: "+request+"\nWeights:\n"+json.dumps(weights,separators=(",",":"))+"\n"
        "Return JSON with summary (brief public explanation, not detailed chain of thought), "
        "estimated_c (number for semantic damping, otherwise null), and patches. "
        "Each patch is {tensor: W1/b1/W2/b2, index: [row,column] for W1 or [i] for vectors, "
        "value: a numerical replacement}. Do not return an executable expression. "
        "Use enough decimal precision. If you cannot find an edit, return an empty patches list.")
