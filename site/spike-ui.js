'use strict';
(()=>{
  const C=globalThis.SpikeCore,D=globalThis.SpikeData,$=id=>document.getElementById('spike-'+id);
  const names={adaptive:'Adaptive gate + threshold',fixed:'Trained fixed schedule',gate_only:'Adaptive gate · fixed thresholds',threshold_only:'Fixed gates · adaptive threshold',zero_threshold:'Adaptive gate · threshold zero',random:'Independent random commands',graded_adaptive:'Graded adaptive reference',graded_fixed:'Graded fixed reference'};
  const f=(x,n=4)=>Number(x).toFixed(n),pct=x=>(100*x).toFixed(1)+'%';
  const methods=Object.keys(names);
  for(const id of ['method','reference'])$(id).innerHTML=methods.map(m=>`<option value="${m}">${names[m]}</option>`).join('');
  $('reference').value='fixed';
  $('world').innerHTML=D.worlds.map((w,i)=>`<option value="${i}">${w.name}</option>`).join('');
  let selected,reference;
  const world=()=>D.worlds[Number($('world').value)];
  const model=method=>D.models.find(m=>m.method===method&&m.seed===Number($('seed').value));
  function reset(){selected=C.startEpisode(model($('method').value),world(),$('mode').value);reference=C.startEpisode(model($('reference').value),world(),$('mode').value);draw();}
  function history(){
    const h=world().histories,scale=28/Math.max(...h.map(Math.abs),.001);
    const path=h.map((v,i)=>`${i?'L':'M'}${12+i*936/31},${40-v*scale}`).join(' ');
    $('history').innerHTML=`<line x1="12" y1="40" x2="948" y2="40" stroke="#33454d"/><path d="${path}" fill="none" stroke="#efc083" stroke-width="2"/>`;
    $('world-note').textContent=`${world().transfer?'Withheld switching family':'Familiar history family'} · 32 inputs → 12 leaky branches · initial uniform sum zero · ${$('mode').selectedOptions[0].textContent}`;
  }
  function card(run,prefix){
    const get=id=>$(prefix+id),done=run.step===8;
    const command=done?{g:run.gates[7],theta:run.thresholds[7]}:C.nextCommand(run);
    get('title').textContent=names[run.model.method];
    const isGraded=run.model.method.startsWith('graded'),reply=run.observations.at(-1);
    get('reply').textContent=isGraded?f(reply,3):(reply>0?'1':'0');
    get('command-label').textContent=done?'Last applied gate · read 8':`Next physical gate · read ${run.step+1}`;
    get('gates').innerHTML=command.g.map((v,i)=>`<div class="gate" style="height:${Math.max(2,85*v)}px" title="Branch ${i+1}: ${f(v)}"><small>${i+1}</small></div>`).join('');
    get('threshold').textContent=(isGraded?'θ unused: ':'θ ')+f(command.theta);
    get('threshold-marker').style.left=((command.theta/.15+1)*50)+'%';
    get('bits').innerHTML=run.observations.map((v,i)=>`<span class="bit ${v>0?'one':''} ${i===0?'initial':''}" title="${i===0?'Initial read':'Read '+i}: ${isGraded?f(v):v>0?'one':'zero'}">${isGraded?f(v,2):v>0?'1':'0'}</span>`).join('');
    const delta=Math.sqrt(run.memory.reduce((s,v,i)=>s+(v-run.original[i])**2,0)),norm=Math.sqrt(C.dot(run.original,run.original));
    get('damage').textContent=`Original-state displacement: ${pct(delta/Math.max(norm,1e-12))}${isGraded?' · scalar answers, more than one bit per read':''}`;
  }
  function forecasts(){
    const done=selected.step===8;
    $('query').disabled=!done;
    if(!done){for(const id of ['prediction','ref-prediction','truth','errors'])$(id).textContent='—';$('state').innerHTML='<text x="480" y="86" text-anchor="middle" fill="#a4b7bd" font-size="13">Final forecasts become available after eight reads.</text>';$('query-note').textContent='The decoder is trained on the final state after read eight.';return;}
    const q=world().queries[Number($('query').value)],a=C.answerQuestion(selected,q),b=C.answerQuestion(reference,q);
    $('prediction').textContent=f(a.prediction);$('ref-prediction').textContent=f(b.prediction);$('truth').textContent=f(a.target);$('errors').textContent=f(Math.abs(a.prediction-a.target),3)+' / '+f(Math.abs(b.prediction-b.target),3);
    const states=[selected.original,C.predictState(selected.model,selected.h),C.predictState(reference.model,reference.h)],colors=['#efc083','#6ce3cb','#8399bc'];
    const max=Math.max(...states.flat().map(Math.abs),.01),scale=63/max;
    let svg='<line x1="12" y1="78" x2="948" y2="78" stroke="#40515a"/>';
    for(let i=0;i<12;i++){
      for(let k=0;k<3;k++){const v=states[k][i],height=Math.max(1,Math.abs(v)*scale);svg+=`<rect x="${30+i*78+k*15}" y="${v>=0?78-height:78}" width="11" height="${height}" rx="2" fill="${colors[k]}"/>`;}
      svg+=`<text x="${50+i*78}" y="154" fill="#a4b7bd" font-size="10" text-anchor="middle">${i+1}</text>`;
    }
    $('state').innerHTML=svg;
    $('query-note').textContent='This question reads the final inferred state; it is never sent to the gate or threshold selector. Original values are scoring targets only.';
  }
  function evidence(){
    const receipt=D.receipt,condition=$('condition').value,table=receipt.tables[condition],comparison=condition==='transfer'?receipt.gates.S5:receipt.gates.S2;
    $('results').innerHTML=methods.map(m=>{const r=table[m];return `<tr class="${m==='adaptive'?'selected':m==='fixed'?'reference':''}"><td>${names[m]}</td><td>${f(Math.sqrt(r.normalized_mse),3)}</td><td>${f(r.objective,4)}</td><td>${f(r.mean_squared_displacement,6)}</td></tr>`;}).join('');
    const gain=(table.fixed.mse-table.adaptive.mse)/table.fixed.mse;
    $('comparison').textContent=`Adaptive versus trained fixed: ${pct(gain)} lower answer MSE. `+(condition==='familiar'?'Familiar questions are a descriptive condition.':`Paired MSE difference 95% interval [${f(comparison.interval.low,7)}, ${f(comparison.interval.high,7)}]; ${comparison.seed_wins}/3 initializations win. Full loss J ${comparison.objective_better?'improves':'does not improve'}.`);
  }
  function draw(){
    $('count').innerHTML=selected.step+' <small>/ 8</small>';
    card(selected,'');card(reference,'ref-');history();forecasts();
    const done=selected.step===8;$('read').disabled=done;$('finish').disabled=done;
    $('status').textContent=done?'Read budget complete. Try another final question.':`${8-selected.step} questions remain`;
  }
  function claims(){
    const r=D.receipt,s1=D.verification?.S1_passed===true,binary=s1&&r.statistical_binary_interface_passed,joint=binary&&r.statistical_joint_gate_threshold_passed;
    $('gain').textContent=pct(r.gates.S2.relative_gain);
    $('verdict').textContent=joint?'Both learned commands earned their advantage.':binary?'Adaptive thresholds earned an advantage.':'The stronger claim remains unearned.';
    const labels={S1:'Integrity + inference',S2:'Adaptive vs fixed',S3:'Threshold contribution',S4:'Branch contribution',S5:'Unseen histories',S6:'Hidden-memory controls'};
    $('criteria').innerHTML=Object.entries(labels).map(([key,label])=>{const pass=key==='S1'?s1:r.gates[key].passed;return `<span class="criterion ${pass?'pass':'fail'}">${key} · ${label} · ${pass?'PASS':'NOT EARNED'}</span>`;}).join('');
    $('claim-title').textContent=joint?'A learned gate-and-threshold benefit.':binary?'Learning where to put the threshold matters.':'A result with its limits left visible.';
    $('claim').textContent=joint?'A receiver trained on binary replies earns gains from both gate and threshold adaptation on this structured history problem, including withheld histories.':binary?'The learned adaptive binary interface beats a trained fixed schedule and transfers to an unseen history family. The branch-only and threshold-only controls determine which command earns the gain: a joint gate-and-threshold advantage is not established unless both component gates pass.':'This run does not earn the predeclared learned adaptive binary-interface claim. Every control and failed criterion remains visible; the experiment was not retuned on its test outcomes.';
  }
  for(const id of ['world','method','reference','seed','mode'])$(id).addEventListener('change',reset);
  $('read').addEventListener('click',()=>{C.stepEpisode(selected);C.stepEpisode(reference);draw();});
  $('finish').addEventListener('click',()=>{while(selected.step<8){C.stepEpisode(selected);C.stepEpisode(reference);}draw();});
  $('reset').addEventListener('click',reset);$('query').addEventListener('change',forecasts);$('condition').addEventListener('change',evidence);
  claims();evidence();reset();
})();
