(function(root){
'use strict';
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const norm=a=>Math.sqrt(dot(a,a));
function mat(a,w){return w[0].map((_,j)=>a.reduce((s,v,i)=>s+v*w[i][j],0));}
function initialReceiver(model,answer){const p=model.params;return p.initial_w.map((v,i)=>Math.tanh(answer/.1*v+p.initial_b[i]));}
function selectGate(model,h,step,randomGate){
 if(!Number.isInteger(step)||step<0||step>=3)throw new Error('Reading budget reached');
 if(model.method==='random'){if(!randomGate)throw new Error('Independent random gate required');return randomGate.slice();}
 const p=model.params,extra=model.method==='adaptive'?mat(h,p.policy):p.gate_logits[step].map(()=>0);
 const logits=p.gate_logits[step].map((v,i)=>v+extra[i]),top=Math.max(...logits),weights=logits.map(v=>Math.exp(v-top)),length=norm(weights);
 return weights.map(v=>v/length);
}
function receive(model,h,gate,answer){const p=model.params,a=mat(h,p.recurrent),b=mat(gate,p.gate_input);return a.map((v,i)=>Math.tanh(v+b[i]+answer/.1*p.answer_input[i]+p.bias[i]));}
function predictState(model,h){const p=model.params,a=mat(h,p.decoder);return a.map((v,i)=>.1*(v+p.decoder_b[i]));}
function begin(world,model,mode='normal'){
 if(!['normal','no_context','erased','post_mix'].includes(mode))throw new Error('Unknown ablation');
 const actual=world.states.map(v=>mode==='erased'?0:v),initial=actual.reduce((a,b)=>a+b,0)/Math.sqrt(12)+world.noise[0];
 return {world,model,mode,original:world.states.slice(),actual,h:initialReceiver(model,initial),gates:[],observations:[initial]};
}
function nextGate(run){const i=run.gates.length;
 const chosen=selectGate(run.model,run.mode==='no_context'?run.h.map(()=>0):run.h,i,run.world.random_gates[i]);
 return run.mode==='post_mix'?chosen.map(()=>1/Math.sqrt(12)):chosen;
}
function step(run){
 if(run.gates.length>=3)throw new Error('Reading budget reached');
 const gate=nextGate(run),value=dot(run.actual,gate),answer=value+run.world.noise[run.gates.length+1];
 run.h=receive(run.model,run.h,gate,answer);run.actual=run.actual.map((v,i)=>v-.25*gate[i]*value);
 run.gates.push(gate);run.observations.push(answer);return run;
}
function answer(run,query){return {prediction:dot(predictState(run.model,run.h),query),truth:dot(run.original,query)};}
function disturbance(run){return norm(run.actual.map((v,i)=>v-run.original[i]))/Math.max(norm(run.original),1e-12);}
const api={dot,norm,mat,initialReceiver,selectGate,receive,predictState,begin,nextGate,step,answer,disturbance};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.LearnedQueryCore=api;
})(globalThis);
