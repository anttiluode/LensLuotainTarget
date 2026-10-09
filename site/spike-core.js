'use strict';
const SpikeCore=(()=>{
  const BUDGET=8,SCALE=.1,ETA=.25,THRESHOLD_SCALE=.15;
  const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
  const mv=(a,w)=>w[0].map((_,j)=>a.reduce((s,v,i)=>s+v*w[i][j],0));
  const positive=x=>{const top=Math.max(...x),v=x.map(z=>Math.exp(z-top)),n=Math.sqrt(dot(v,v));return v.map(z=>z/n);};
  const graded=m=>m.method.startsWith('graded');
  function initialReceiver(model,initialAnswer){const p=model.params,a=initialAnswer/(graded(model)?SCALE:1);return p.initial_w.map((v,i)=>Math.tanh(a*v+p.initial_b[i]));}
  function selectCommand(model,h,step,randomGate,randomThreshold){
    if(!Number.isInteger(step)||step<0||step>=BUDGET)throw Error('Eight-read budget exhausted');
    const p=model.params,method=model.method;
    const contextGate=positive(mv(h,p.gate_policy).map((v,i)=>v+p.gate_logits[step][i]));
    const contextThreshold=THRESHOLD_SCALE*Math.tanh(dot(h,p.threshold_policy)+p.threshold_logits[step]);
    let g=['adaptive','gate_only','zero_threshold','graded_adaptive'].includes(method)?contextGate.slice():positive(p.gate_logits[step]);
    let theta=['adaptive','threshold_only','graded_adaptive'].includes(method)?contextThreshold:THRESHOLD_SCALE*Math.tanh(p.threshold_logits[step]);
    if(method==='zero_threshold')theta=0;
    if(method==='random'){if(!randomGate||!Number.isFinite(randomThreshold))throw Error('Random commands missing');g=randomGate.slice();theta=randomThreshold;}
    return {g,theta,context_gate:contextGate,context_threshold:contextThreshold};
  }
  function receive(model,h,command,answer){
    const p=model.params,r=answer/(graded(model)?SCALE:1),a=mv(h,p.recurrent),ga=mv(command.g,p.actual_gate_input),gc=mv(command.context_gate,p.context_gate_input);
    return a.map((v,i)=>Math.tanh(v+ga[i]+gc[i]+command.theta/SCALE*p.actual_threshold_input[i]+command.context_threshold/SCALE*p.context_threshold_input[i]+r*p.answer_input[i]+p.bias[i]));
  }
  function predictState(model,h){const p=model.params;return mv(h,p.decoder).map((v,i)=>SCALE*(v+p.decoder_b[i]));}
  function startEpisode(model,world,mode='normal'){
    if(!['normal','erased','post_mix'].includes(mode))throw Error('Unknown control');
    const original=world.states.slice(),memory=mode==='erased'?original.map(()=>0):original.slice();
    const a=memory.reduce((s,v)=>s+v/Math.sqrt(12),0)+world.noise[0];
    const initial=graded(model)?a:(a>0?1:-1),h=initialReceiver(model,initial);
    return {model,world,mode,original,memory,h,step:0,observations:[initial],gates:[],thresholds:[],hidden:[h.slice()]};
  }
  function nextCommand(run){
    const c=selectCommand(run.model,run.h,run.step,run.world.random_gates[run.step],run.world.random_thresholds[run.step]);
    if(run.mode==='post_mix')c.g=Array(12).fill(1/Math.sqrt(12));
    return c;
  }
  function stepEpisode(run){
    const c=nextCommand(run),value=dot(c.g,run.memory),noisy=value+run.world.noise[run.step+1];
    const reply=graded(run.model)?noisy:(noisy>c.theta?1:-1);
    run.h=receive(run.model,run.h,c,reply);
    run.memory=run.memory.map((v,i)=>v-ETA*c.g[i]*value);
    run.gates.push(c.g.slice());run.thresholds.push(c.theta);run.observations.push(reply);run.hidden.push(run.h.slice());run.step++;
    return run;
  }
  function answerQuestion(run,question){return {prediction:dot(question,predictState(run.model,run.h)),target:dot(question,run.original)};}
  return {BUDGET,dot,initialReceiver,selectCommand,receive,predictState,startEpisode,nextCommand,stepEpisode,answerQuestion};
})();
if(typeof module!=='undefined'&&module.exports)module.exports=SpikeCore;else globalThis.SpikeCore=SpikeCore;
