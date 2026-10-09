(function(root){
'use strict';
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const norm=a=>Math.sqrt(dot(a,a));
const copy=a=>a.map(r=>r.slice());
function trace(history,decays){let state=decays.map(()=>0);for(const u of history)state=state.map((v,i)=>decays[i]*v+(1-decays[i])*u);return state;}
function posteriorUpdate(p,predictions,observation,sigma){
 if(!Number.isFinite(sigma)||sigma<=0)throw new Error('Noise must be positive');
 const logs=p.map((v,i)=>Math.log(Math.max(v,2.2250738585072014e-308))-.5*((predictions[i]-observation)/sigma)**2);
 const top=Math.max(...logs),weights=logs.map(v=>Math.exp(v-top)),total=weights.reduce((a,b)=>a+b,0);
 return weights.map(v=>v/total);
}
function chooseGate(p,predictions,available){
 if(!available.length)throw new Error('No unused gate remains');
 let best=available[0],score=-Infinity;
 for(const index of available){const pred=predictions[index],mean=dot(pred,p),v=pred.reduce((s,x,i)=>s+p[i]*(x-mean)**2,0);if(v>score){score=v;best=index;}}
 return best;
}
function readState(state,gate,eta){const value=dot(state,gate);return {value,state:state.map((v,i)=>v-eta*gate[i]*value)};}
function begin(scene,config){
 if(!Number.isFinite(config.noise)||config.noise<=0||!Number.isFinite(config.read_damage)||config.read_damage<0||config.read_damage>1)throw new Error('Invalid read parameters');
 const originals=scene.histories.map(h=>trace(h,config.decays));
 const candidates=config.method==='erased'?originals.map(r=>r.map(()=>0)):copy(originals);
 const truth=config.truth===undefined?scene.truth:config.truth;
 if(!Number.isInteger(truth)||truth<0||truth>=candidates.length)throw new Error('Invalid history');
 const actual=candidates[truth].slice(),prior=candidates.map(()=>1/candidates.length);
 const initial=dot(actual,config.uniform)+config.noise*scene.initial_noise;
 return {scene,config,truth,originals,candidates,actual,originalTrue:originals[truth].slice(),prior,
         posterior:posteriorUpdate(prior,candidates.map(r=>dot(r,config.uniform)),initial,config.noise),
         actions:[],observations:[initial],records:[]};
}
function nextGate(run){
 const {config,candidates,posterior,prior,actions,scene}=run;
 const available=config.gates.map((_,i)=>i).filter(i=>!actions.includes(i));
 const predictions=config.gates.map(g=>candidates.map(s=>dot(g,s)));
 if(config.method==='active'||config.method==='erased')return chooseGate(posterior,predictions,available);
 if(config.method==='open_loop')return chooseGate(prior,predictions,available);
 if(config.method==='random')return scene.random_order.find(i=>available.includes(i));
 const fixed=Array.from({length:config.branches},(_,i)=>i).sort((a,b)=>(a%4-b%4)||Math.floor(a/4)-Math.floor(b/4));
 return [...fixed,...config.gates.map((_,i)=>i).slice(config.branches)].find(i=>available.includes(i));
}
function step(run,manualGate=null){
 if(run.actions.length>=run.config.budget)throw new Error('Reading budget reached');
 const index=manualGate===null?nextGate(run):manualGate;
 if(!Number.isInteger(index)||index<0||index>=run.config.gates.length||run.actions.includes(index))throw new Error('Choose an unused gate');
 const gate=run.config.method==='post_mix'?run.config.uniform:run.config.gates[index];
 const predictions=run.candidates.map(s=>dot(s,gate));
 const answer=readState(run.actual,gate,run.config.read_damage);
 const received=answer.value+run.config.noise*run.scene.noise[run.actions.length][index];
 run.posterior=posteriorUpdate(run.posterior,predictions,received,run.config.noise);
 run.candidates=run.candidates.map((s,j)=>s.map((v,i)=>v-run.config.read_damage*gate[i]*predictions[j]));
 run.actual=answer.state;
 run.actions.push(index);run.observations.push(received);
 run.records.push({index,gate:gate.slice(),predictions,received});
 return run;
}
function disturbance(run){return norm(run.actual.map((v,i)=>v-run.originalTrue[i]))/Math.max(norm(run.originalTrue),1e-12);}
function entropy(p){return -p.reduce((s,v)=>s+(v>0?v*Math.log2(v):0),0);}
const api={dot,norm,trace,posteriorUpdate,chooseGate,readState,begin,nextGate,step,disturbance,entropy};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.MemoryGateCore=api;
})(globalThis);
