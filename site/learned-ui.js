(function(){
'use strict';
const C=globalThis.LearnedQueryCore,D=globalThis.LearnedQueryData,$=id=>document.getElementById(id);
const names={adaptive:'Learned adaptive',fixed:'Optimized fixed',random:'Random',recurrent:'Matched recurrent',no_context:'Selection context removed',erased:'Branch memory erased',post_mix:'Gate after mixing'};
let run,worldIndex=0,queryIndex=0;
D.worlds.forEach((world,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent=world.name+(world.transfer?' · unseen family':'');$('learned-world').appendChild(option);});
function reset(){const model=D.models.find(m=>m.method===$('learned-model').value);run=C.begin(D.worlds[worldIndex],model,$('learned-control').value);render();}
function plot(canvas,lines,range){const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);ctx.strokeStyle='#324c62';ctx.lineWidth=1;for(let j=0;j<5;j++){const y=14+(h-30)*j/4;ctx.beginPath();ctx.moveTo(24,y);ctx.lineTo(w-12,y);ctx.stroke();}lines.forEach((values,i)=>{ctx.strokeStyle=['#eeb273','#75d5bc'][i];ctx.lineWidth=2.3;ctx.beginPath();values.forEach((v,k)=>{const x=24+(w-40)*k/Math.max(values.length-1,1),y=14+(range[1]-v)/(range[1]-range[0])*(h-30);if(k)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.stroke();});}
function render(){const finished=run.gates.length===3,gate=finished?run.gates.at(-1):C.nextGate(run),world=run.world;
 $('learned-count').textContent=`${run.gates.length} / 3`;$('learned-damage').textContent=(100*C.disturbance(run)).toFixed(1)+'%';
 const r=D.receipt.resources[run.model.method];$('learned-storage').textContent=String(r.persistent_values);
 $('learned-resources').textContent=`${r.parameters} trained parameters · ${r.receiver_values} receiver values + 12 branch values · ${r.neural_macs} neural MACs per complete episode. Prediction workspace and gate-command values are additional.`;
 $('learned-world-note').textContent=world.transfer?'A switching-block history family was absent from training. This demonstration is separate from the held-out transfer set.':'A new example from one of the four training families. This specific history is separate from training, validation and held-out tests.';
 plot($('learned-history'),[world.histories],[-1,1]);
 $('learned-gate-label').textContent=finished?'Last issued gate · current retained state':`Next query · read ${run.gates.length+1}`;
 $('learned-gates').innerHTML='';gate.forEach((value,i)=>{const col=document.createElement('div');col.className='gate-col';col.setAttribute('aria-label',`Branch ${i+1}, coefficient ${value.toFixed(3)}`);col.innerHTML=`<div class="gate-track"><div class="gate-fill" style="height:${100*value}%"></div></div>${i+1}<strong>${value.toFixed(3)}</strong>`;$('learned-gates').appendChild(col);});
 $('learned-current-sum').textContent=C.dot(run.actual,gate).toFixed(4);$('learned-last-answer').textContent=run.observations.at(-1).toFixed(4);
 $('learned-last-label').textContent=run.gates.length?`Completed read ${run.gates.length}`:'Initial shared reading';
 $('learned-read').disabled=finished;
 const notes={normal:'Each received answer changes the receiver state. Adaptive selection uses that state for the next gate; the other strategies use a trained schedule or independent random commands.',no_context:'The receiver still retains answers for prediction, but selection is given zero context. This is an inference ablation of the adaptive model, not a separately trained competitor.',erased:'The sender memory is zeroed. Scoring still asks about the original history; predicting the erased state cannot count as recall.',post_mix:'The actual reading uses the common uniform sum after mixing. The receiver cannot expose branch differences through that scalar.'};
 $('learned-mode-note').textContent=notes[run.mode];$('learned-mode-note').classList.toggle('warning',run.mode!=='normal');
 $('learned-observations').textContent=run.observations.map((v,i)=>`${i?`Read ${i}`:'Initial'}: ${v.toFixed(4)}`).join(' · ');
 const estimate=C.predictState(run.model,run.h),range=Math.max(.03,...run.original.map(Math.abs),...estimate.map(Math.abs))*1.25;
 plot($('learned-inferred'),[run.original,estimate],[-range,range]);
 const q=world.queries[queryIndex],answer=C.answer(run,q);$('learned-prediction').textContent=answer.prediction.toFixed(4);$('learned-truth').textContent=answer.truth.toFixed(4);$('learned-error').textContent=Math.abs(answer.prediction-answer.truth).toFixed(4);
 $('learned-query-vector').textContent='Question coefficients: '+q.map(v=>v.toFixed(2)).join(', ');
 results();
}
function results(){const receipt=D.receipt,c=receipt.conditions.find(c=>c.name===$('learned-condition').value);
 $('learned-results').innerHTML='';Object.entries(c.summary).forEach(([method,s])=>{const row=document.createElement('tr');if(method==='adaptive')row.className='active';row.innerHTML=`<td>${names[method]}</td><td>${s.nrmse.toFixed(4)}</td><td>${s.objective.toFixed(5)}</td><td>${(100*s.relative_disturbance).toFixed(1)}%</td>`;$('learned-results').appendChild(row);});
 const fixed=c.comparisons.fixed,recurrent=c.comparisons.recurrent;
 $('learned-comparison').textContent=`Adaptive future-answer MSE improves by ${(100*fixed.mse.relative_gain).toFixed(1)}% against optimized fixed and ${(100*recurrent.mse.relative_gain).toFixed(1)}% against matched recurrent. Seed wins: ${fixed.positive_seed_wins}/3 and ${recurrent.positive_seed_wins}/3.`;
 $('learned-intervals').textContent=`Paired 95% intervals for baseline minus adaptive MSE: fixed [${fixed.mse.low.toExponential(2)}, ${fixed.mse.high.toExponential(2)}]; recurrent [${recurrent.mse.low.toExponential(2)}, ${recurrent.mse.high.toExponential(2)}].`;
 const labels={L1_contract_checks:'Boundary / gradients / parity',L2_random:'Beyond random',L3_fixed_and_recurrent:'Beyond trained controls',L4_unseen_history_transfer:'Unseen-family transfer',L5_information_loss_controls:'Information-loss controls'};
 $('learned-frozen-gates').innerHTML='';Object.entries(receipt.gates).forEach(([key,value])=>{const tag=document.createElement('span');tag.className=value===true?'':value===false?'fail':'pending';tag.textContent=`${value===true?'PASS':value===false?'FAIL':'PENDING'} · ${labels[key]}`;$('learned-frozen-gates').appendChild(tag);});
 const earned=receipt.adaptive_advantage_earned===true;$('learned-verdict').textContent=earned?'The learned interface passed this bounded test.':'The learned interface has not earned every frozen gate.';$('learned-verdict').classList.toggle('fail',!earned);
 $('learned-claim').textContent=earned?'Without a supplied history dictionary, this trained interface predicts new linear answers better than the optimized fixed schedule and the specified recurrent receiver, including the held-out switching family. The result supports this particular learned interface and benchmark.':'The complete receipt preserves what worked and which frozen requirements failed. A win against random alone does not establish an adaptive advantage against the trained controls.';
}
$('learned-read').onclick=()=>{C.step(run);render();};$('learned-reset').onclick=reset;
$('learned-world').onchange=()=>{worldIndex=Number($('learned-world').value);reset();};
$('learned-model').onchange=()=>{$('learned-control').value='normal';reset();};
$('learned-control').onchange=()=>{if($('learned-control').value!=='normal')$('learned-model').value='adaptive';reset();};
$('learned-query').onchange=()=>{queryIndex=Number($('learned-query').value);render();};
$('learned-condition').onchange=results;reset();
})();
