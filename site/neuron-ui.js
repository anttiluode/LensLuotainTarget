(function(){
'use strict';
const C=globalThis.MemoryGateCore,D=globalThis.MemoryGateData,$=id=>document.getElementById(id),letters='ABCDEFGH';
let sceneIndex=0,run,pending=null;
const labels={active:'Memory-guided gates',random:'Random gates',fixed:'Fixed branches',open_loop:'Gates without retained context',post_mix:'Gate after mixing',erased:'Branch memory erased'};
function tabs(){const neural=location.hash==='#neuron';$('neuron-view').hidden=!neural;$('shadow-view').hidden=neural;
 $('tab-neuron').setAttribute('aria-selected',String(neural));$('tab-shadow').setAttribute('aria-selected',String(!neural));
 $('tab-neuron').tabIndex=neural?0:-1;$('tab-shadow').tabIndex=neural?-1:0;}
$('tab-neuron').onclick=()=>{location.hash='neuron';};$('tab-shadow').onclick=()=>{location.hash='shadow';};
document.querySelector('.view-tabs').addEventListener('keydown',event=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(event.key)){event.preventDefault();const neural=event.key==='End'||(event.key!=='Home'&&location.hash!=='#neuron');location.hash=neural?'neuron':'shadow';(neural?$('tab-neuron'):$('tab-shadow')).focus();}});
window.addEventListener('hashchange',tabs);tabs();
function config(){let method=$('neuron-policy').value;if(!$('neuron-retain').checked)method='erased';else if($('neuron-location').value==='after')method='post_mix';
 return {...D.config,method,truth:Number($('neuron-history').value),noise:Number($('neuron-noise').value),read_damage:$('neuron-damage').checked?.25:0};}
function reset(){pending=null;run=C.begin(D.scenes[sceneIndex],config());render();}
function plot(canvas,lines,range){const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);
 ctx.strokeStyle='#324c62';ctx.lineWidth=1;for(let j=0;j<5;j++){const y=14+(h-30)*j/4;ctx.beginPath();ctx.moveTo(24,y);ctx.lineTo(w-12,y);ctx.stroke();}
 lines.forEach((values,i)=>{ctx.strokeStyle=['#74d5bd','#eeb273','#92baff'][i];ctx.lineWidth=2.4;ctx.beginPath();values.forEach((v,k)=>{const x=24+(w-40)*k/Math.max(values.length-1,1),y=14+(range[1]-v)/(range[1]-range[0])*(h-30);if(!k)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();});}
function diagram(gate){const state=run.actual,svg=$('neuron-cell');let content='<text x="22" y="17" fill="#91abc0" font-size="12">Retained branch traces</text><text x="263" y="17" fill="#91abc0" font-size="12">Gate</text><text x="612" y="102" fill="#91abc0" font-size="12">Current sum</text><text x="612" y="118" fill="#91abc0" font-size="10">Before sensor noise</text>';
 const scale=Math.max(...state.map(Math.abs),.001);
 state.forEach((v,i)=>{const y=34+i*21,on=gate[i]>0;content+=`<text x="20" y="${y+4}" fill="#839db5" font-size="11">${i+1}</text><line x1="128" y1="${y}" x2="${128+v/scale*66}" y2="${y}" stroke="${v>=0?'#73cdb7':'#e1aa73'}" stroke-width="5"/><path d="M206 ${y} C332 ${y} 382 155 441 155" fill="none" stroke="${on?'#78d1b7':'#385066'}" stroke-width="${on?2.4:1}" opacity="${on?1:.45}"/><rect x="267" y="${y-5}" width="20" height="10" rx="3" fill="${on?'#6cc3a9':'#23394b'}"/><text x="211" y="${y+3}" fill="#92acbf" font-size="9">${v.toFixed(3)}</text>`;});
 content+='<line x1="128" y1="26" x2="128" y2="280" stroke="#365269"/><circle cx="480" cy="155" r="39" fill="#193747" stroke="#73c7af" stroke-width="2"/><text x="480" y="159" text-anchor="middle" fill="#d1eae2" font-size="15">Soma</text><path d="M520 155 H603 M595 148 L603 155 L595 162" fill="none" stroke="#78c9b3" stroke-width="2"/>';
 content+=`<rect x="612" y="129" width="106" height="54" rx="8" fill="#193343" stroke="#436878"/><text x="665" y="161" text-anchor="middle" fill="#a0e7d2" font-size="22">${C.dot(state,gate).toFixed(3)}</text><text x="390" y="290" fill="#7f99ad" font-size="11">Current traces → selected weighted sum; received readings below</text>`;
 svg.innerHTML=content;
}
function render(){const finished=run.actions.length>=run.config.budget,chosen=finished?run.actions.at(-1):(pending===null?C.nextGate(run):pending);
 const gate=run.config.method==='post_mix'?run.config.uniform:run.config.gates[chosen];
 const other=run.truth===0?1:0,a=run.candidates[run.truth],b=run.candidates[other];
 const ya=C.dot(a,gate),yb=C.dot(b,gate),gap=Math.abs(ya-yb)/run.config.noise;
 plot($('neuron-histories'),[run.scene.histories[run.truth],run.scene.histories[other]],[-1,1]);
 $('neuron-history-label').textContent=`Histories ${letters[run.truth]} and ${letters[other]}`;
 $('neuron-same').textContent=Math.max(...run.originals.map(s=>Math.abs(C.dot(s,run.config.uniform)))).toFixed(5);
 $('neuron-next').textContent=finished?'Three readings used':`Next gate ${chosen+1}${pending!==null?' · chosen by you':''}`;
 $('neuron-predictions').textContent=finished?'Reset to ask again.':`${letters[run.truth]} predicts ${ya.toFixed(3)}; ${letters[other]} predicts ${yb.toFixed(3)}. Separation: ${gap.toFixed(2)} × noise.`;
 $('neuron-next-branches').textContent=run.config.method==='post_mix'?'The gate acts after the branch values have already been combined.':`Open branches: ${gate.map((v,i)=>v>0?i+1:null).filter(v=>v!==null).join(', ')}.`;
 $('neuron-reading-count').textContent=`${run.actions.length} / 3`;
 $('neuron-uncertainty').textContent=C.entropy(run.posterior).toFixed(2)+' bits';
 $('neuron-disturbance').textContent=(100*C.disturbance(run)).toFixed(1)+'%';
 $('neuron-noise-value').textContent=run.config.noise.toFixed(3);
 $('neuron-probe').disabled=finished;
 $('neuron-mode-note').textContent=run.config.method==='erased'?'The branch traces have been erased. The observer still knows the candidate histories, but its readings cannot reveal which one happened.':run.config.method==='post_mix'?'After-mixing gates only see the common scalar. They cannot expose the retained differences between branches.':run.config.method==='open_loop'?'Gate selection discards the previous readings. The decoder still keeps them: this isolates the value of context when choosing a gate.':run.config.read_damage?'Each read also changes the branch traces. The meter shows how far this abstract memory has moved from its original state.':'Reads expose the traces without changing them in this condition.';
 $('neuron-mode-note').classList.toggle('warning',['erased','post_mix'].includes(run.config.method));
 $('neuron-branches').innerHTML='';for(let i=0;i<12;i++){const button=document.createElement('button');button.textContent=String(i+1);button.className=gate[i]>0?'selected':'';button.setAttribute('aria-label',`Choose branch ${i+1} as the next gate`);button.setAttribute('aria-pressed',String(pending===i));button.disabled=finished||run.actions.includes(i)||run.config.method==='post_mix'||run.config.method==='erased';button.onclick=()=>{pending=pending===i?null:i;render();};$('neuron-branches').appendChild(button);}
 diagram(gate);
 const range=Math.max(.08,...run.observations.map(Math.abs))*1.3;plot($('neuron-readings'),[run.observations],[-range,range]);
 $('neuron-reading-list').textContent=run.observations.map((v,i)=>`${i===0?'Initial':`Read ${i}`}: ${v.toFixed(3)}`).join(' · ');
 const winner=run.posterior.indexOf(Math.max(...run.posterior));$('neuron-beliefs').innerHTML='';run.posterior.forEach((p,i)=>{const col=document.createElement('div');col.className='belief-col'+(winner===i?' winner':'');col.innerHTML=`<div class="belief-track"><div class="belief-fill" style="height:${100*p}%"></div></div><span>${letters[i]}</span><strong>${(100*p).toFixed(0)}%</strong>`;col.setAttribute('aria-label',`History ${letters[i]} probability ${(100*p).toFixed(1)} percent`);$('neuron-beliefs').appendChild(col);});
 $('neuron-belief-note').textContent=`${labels[run.config.method]}. The receiver has ${run.actions.length+1} scalar readings and the known eight-history dictionary.`;
 resultTable();
}
function resultTable(){const c=D.receipt.conditions.find(x=>x.config.setup.read_damage===run.config.read_damage),body=$('neuron-result-rows');body.innerHTML='';
 Object.entries(c.summary).forEach(([method,s])=>{const row=document.createElement('tr');if(method==='active')row.className='active';row.innerHTML=`<td>${labels[method]}</td><td>${s.correct} / ${s.total} <small>(${(100*s.accuracy).toFixed(1)}%)</small></td><td>${s.mean_logloss.toFixed(3)}</td><td>${s.forecast_nrmse.toFixed(3)}</td><td>${(100*s.mean_disturbance).toFixed(1)}%</td>`;body.appendChild(row);});
 $('neuron-result-condition').textContent=run.config.read_damage?'Frozen condition: reads change memory (η = 0.25).':'Frozen condition: non-disturbing reads (η = 0).';
 $('neuron-result-noise-note').textContent='The benchmark uses noise 0.030 and three new readings. The noise slider changes only the explainer above.';
 const titles={G1_initial_ambiguity:'Same initial soma',G2_active_vs_random:'Guided > random',G3_retained_context_vs_open_loop:'Context earns a role',G4_information_loss_controls:'Erasure / post-mix controls',G5_read_disturbance:'Read-write check'};
 $('neuron-gates').innerHTML='';Object.entries(c.gates).forEach(([key,pass])=>{const tag=document.createElement('span');tag.className=pass?'':'fail';tag.textContent=`${pass?'PASS':'FAIL'} · ${titles[key]}`;$('neuron-gates').appendChild(tag);});
 $('neuron-result-verdict').textContent=`Guided gates gain ${(100*c.comparisons.active_minus_random_accuracy).toFixed(1)} percentage points over random. Their log loss improves by ${c.comparisons.open_loop_minus_active_logloss.toFixed(3)} nats over the stronger comparison that discards selection context.`;
}
$('neuron-probe').onclick=()=>{C.step(run,pending);pending=null;render();};$('neuron-reset').onclick=reset;
$('neuron-world').onclick=()=>{sceneIndex=(sceneIndex+1)%D.scenes.length;$('neuron-history').value=String(D.scenes[sceneIndex].truth);reset();};
for(const id of ['neuron-policy','neuron-history','neuron-location','neuron-retain','neuron-damage'])$(id).addEventListener('change',reset);
$('neuron-noise').addEventListener('input',reset);
$('neuron-history').value=String(D.scenes[0].truth);reset();
})();
