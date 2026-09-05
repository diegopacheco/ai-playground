const ground = 510;
function path(ctx, points, color, width = 2, fill = false) {
  ctx.beginPath();
  points.forEach(([x, y], index) => index ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  if (fill) { ctx.closePath(); ctx.fillStyle = color; ctx.fill(); }
  else ctx.stroke();
}
function rect(ctx, x, y, w, h, color) { ctx.fillStyle = color; ctx.fillRect(x, y, w, h); }
function ellipse(ctx, x, y, rx, ry, color) { ctx.fillStyle = color; ctx.beginPath(); ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2); ctx.fill(); }
function text(ctx, value, x, y, font, color, align = 'center') { ctx.fillStyle = color; ctx.font = font; ctx.textAlign = align; ctx.fillText(value, x, y); }
function cloud(ctx, x, y, scale) {
  ctx.save();ctx.translate(x,y);ctx.scale(scale,scale);
  ellipse(ctx,0,0,48,11,'#f8f7e9');ellipse(ctx,-13,-9,22,15,'#f8f7e9');ellipse(ctx,17,-4,26,13,'#f8f7e9');ctx.restore();
}
function bridge(ctx) {
  const color = '#94aea0';
  path(ctx, [[520,310],[1190,310]], color, 6);
  for (const x of [730, 1080]) {
    rect(ctx,x,154,10,193,color);rect(ctx,x+30,154,10,193,color);
    rect(ctx,x,173,40,7,color);rect(ctx,x,225,40,7,color);rect(ctx,x,285,40,7,color);
    path(ctx,[[x,152],[x+40,152]],color,4);
  }
  ctx.beginPath();ctx.moveTo(510,304);ctx.quadraticCurveTo(630,287,735,160);ctx.quadraticCurveTo(905,420,1085,160);ctx.quadraticCurveTo(1150,260,1240,285);ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();
  for(let x=530;x<1200;x+=24){let y=x<735?304-(x-510)/225*143:x<1085?160+145*Math.sin((x-735)/350*Math.PI):160+(x-1085)*.9;path(ctx,[[x,y],[x,309]],color,1);}
}
function ferry(ctx, x = 765) {
  rect(ctx,x-180,264,390,112,'#cfceb6');rect(ctx,x-193,260,418,12,'#b6bda4');
  for(let i=0;i<18;i++){rect(ctx,x-169+i*21,288,10,24,'#9da997');rect(ctx,x-169+i*21,331,10,36,'#9aa592');}
  rect(ctx,x-25,158,60,217,'#d7d3b8');rect(ctx,x-31,214,72,9,'#b1b59e');rect(ctx,x-30,156,70,9,'#a7b299');
  rect(ctx,x-19,113,48,47,'#d7d3b8');path(ctx,[[x-27,113],[x+5,83],[x+36,113]],'#a7b299',1,true);
  rect(ctx,x-1,61,10,28,'#9ca98f');ellipse(ctx,x+5,187,17,17,'#f7f2d9');ellipse(ctx,x+5,187,2,2,'#73866d');path(ctx,[[x+5,174],[x+5,187],[x+15,192]],'#73866d',2);
  for(let y=239;y<340;y+=35)rect(ctx,x-6,y,20,21,'#9eab94');
  text(ctx,'PORT OF SAN FRANCISCO',x+8,282,'9px sans-serif','#788c74');
}
function wharf(ctx, pier39) {
  for(let i=0;i<6;i++){
    const x=450+i*135;const y=285+(i%2)*15;
    rect(ctx,x,y,127,97,i%2?'#d6cab1':'#cccbb3');
    path(ctx,[[x-7,y],[x+64,y-39],[x+133,y]],i%2?'#b49b83':'#9bab91',1,true);
    for(let j=0;j<4;j++){rect(ctx,x+12+j*28,y+20,16,24,'#8d9d8a');rect(ctx,x+12+j*28,y+58,16,39,'#8d9d8a');}
  }
  rect(ctx,630,227,11,163,'#8f9a7b');rect(ctx,840,227,11,163,'#8f9a7b');rect(ctx,621,220,240,45,'#f6eed8');
  text(ctx,pier39?'P I E R  3 9':'FISHERMAN’S WHARF',740,248,'bold 19px serif','#8b7359');
  if(pier39){
    for(let i=0;i<5;i++){ellipse(ctx,690+i*69,420,31,12,'#8c9480');ellipse(ctx,711+i*69,409,10,15,'#8c9480');}
  }else{
    path(ctx,[[860,408],[897,434],[1020,434],[1045,408]],'#faf3de',1,true);rect(ctx,919,376,68,32,'#e6d5b8');rect(ctx,953,339,4,40,'#7f9886');path(ctx,[[958,341],[958,374],[993,374]],'#f6ecd4',1,true);
  }
}
function tree(ctx,x,y,s=1){ctx.save();ctx.translate(x,y);ctx.scale(s,s);path(ctx,[[0,0],[0,-113]],'#879174',9);ellipse(ctx,-9,-110,35,44,'#a8b793');ellipse(ctx,17,-130,30,39,'#b7c69e');ellipse(ctx,-27,-135,25,34,'#b7c69e');ctx.restore();}
function lamp(ctx,x){path(ctx,[[x,455],[x,287],[x+10,277],[x+24,277]],'#788b77',3);ellipse(ctx,x+25,280,9,5,'#718571');ellipse(ctx,x+25,283,6,3,'#f5f1d7');}
export function drawSkater(ctx, x, y, rider, time = 0, jump = 0, trick = '', trickAge = 1, scale = 1) {
  ctx.save();ctx.translate(x,y);ctx.scale(scale,scale);ctx.lineCap='round';ctx.lineJoin='round';
  const girl=rider==='girl';const active=trick && trickAge<.48;const phase=active?trickAge/.48*Math.PI*2:0;
  if(trick==='spin'&&active)ctx.scale(Math.cos(phase),1);
  const lean=jump>0?-7:Math.sin(time*3)*2;
  const pants=girl?'#516f66':'#49606a';const shirt=girl?'#d99573':'#efad55';const skin='#bd8463';
  path(ctx,[[-4,-44],[-18+lean,-24],[-26,-6]],pants,13);path(ctx,[[8,-46],[21+lean,-26],[30,-6]],pants,13);
  path(ctx,[[-32,-5],[-17,-5]],'#f7f3df',8);path(ctx,[[24,-5],[39,-5]],'#f7f3df',8);
  path(ctx,[[0,-82],[-4,-46],[13,-43],[16,-78]],shirt,1,true);
  path(ctx,[[3,-73],[-16,-59],[-32,-64]],skin,7);path(ctx,[[15,-74],[31,-63],[43,-71]],skin,7);
  path(ctx,[[0,-77],[-11,-65]],shirt,12);path(ctx,[[16,-77],[25,-68]],shirt,12);
  path(ctx,[[9,-86],[10,-94]],skin,8);
  if(girl){ellipse(ctx,2,-102,14,18,'#584e3f');path(ctx,[[-7,-101],[-17,-83],[-23,-87]],'#584e3f',9);}
  ellipse(ctx,11,-102,11,13,skin);path(ctx,[[20,-105],[24,-100],[20,-99]],skin,2,true);
  if(girl){path(ctx,[[0,-105],[4,-116],[17,-113]],'#584e3f',8);path(ctx,[[-15,-91],[-20,-92]],'#e6a858',3);}
  else{path(ctx,[[-1,-110],[5,-119],[17,-118],[25,-109]],'#688573',1,true);path(ctx,[[-3,-109],[28,-109]],'#688573',4);}
  ellipse(ctx,18,-103,1,1,'#423e34');
  ctx.save();ctx.translate(3,5);if(active&&trick!=='spin')ctx.scale(1,Math.cos(phase));
  path(ctx,[[-40,-4],[-32,1],[30,1],[40,-5]],'#626d4d',5);path(ctx,[[-29,4],[29,4]],'#d59b62',2);ellipse(ctx,-24,9,5,5,'#faf4df');ellipse(ctx,25,9,5,5,'#faf4df');ellipse(ctx,-24,9,2,2,'#879078');ellipse(ctx,25,9,2,2,'#879078');ctx.restore();ctx.restore();
}
export function drawScene(ctx, game, elapsed) {
  const w=1200,h=650;ctx.clearRect(0,0,w,h);
  const sky=ctx.createLinearGradient(0,0,0,450);sky.addColorStop(0,'#e7efdf');sky.addColorStop(1,'#f6f2db');rect(ctx,0,0,w,h,sky);
  ellipse(ctx,930,111,45,45,'#f9f2cc');cloud(ctx,452,105,1.2);cloud(ctx,1078,182,.9);cloud(ctx,136,84,.7);
  path(ctx,[[0,297],[80,284],[140,291],[230,271],[360,292],[445,284],[550,302],[720,286],[915,299],[1100,280],[1200,299],[1200,382],[0,382]],'#ccd8c1',1,true);
  for(let i=0;i<22;i++){const x=30+i*31;rect(ctx,x,286-(i*23%50),17+(i%3)*8,80,'#c3d0b9');}
  if(game.spot.id==='embarcadero')bridge(ctx);
  rect(ctx,0,353,1200,102,'#b6cfc1');
  if(game.spot.id==='ferry')ferry(ctx);
  if(['wharf','pier39'].includes(game.spot.id))wharf(ctx,game.spot.id==='pier39');
  for(let i=0;i<36;i++){const x=(i*113+Math.sin(elapsed*.4+i)*8)%1200;path(ctx,[[x,366+(i*17)%82],[x+20+(i%4)*9,366+(i*17)%82]],'#d7e2cb',1.5);}
  if(game.spot.id==='embarcadero'){
    path(ctx,[[874,394],[895,409],[959,409],[973,394]],'#f8f3df',1,true);rect(ctx,911,375,39,19,'#f8f3df');rect(ctx,925,367,5,8,'#e0b994');
  }
  const offset=game.distance*.22;
  ctx.save();
  for(let i=-1;i<6;i++){
    const x=i*290-(offset%290);lamp(ctx,x+140);
  }
  path(ctx,[[0,424],[1200,424]],'#899c85',3);path(ctx,[[0,439],[1200,439]],'#a1ae93',2);
  for(let i=-1;i<26;i++){const x=i*55-(offset%55);path(ctx,[[x,424],[x,460]],'#8c9d84',3);}
  ctx.restore();
  rect(ctx,0,457,1200,193,'#e7e4cf');rect(ctx,0,457,1200,9,'#c9cfb6');rect(ctx,0,466,1200,4,'#f5f1dc');
  for(let i=-2;i<9;i++){const x=i*210-(game.distance%210);path(ctx,[[x,472],[x-100,650]],'#d5d5bf',1);}
  path(ctx,[[0,544],[1200,544]],'#d5d5bf',1);path(ctx,[[0,614],[1200,614]],'#d5d5bf',1);
  for(let i=-1;i<3;i++){const x=i*750-((game.distance*.5)%750)+40;tree(ctx,x,463,.8);rect(ctx,x-30,453,60,17,'#acb393');}
  ellipse(ctx,287,ground+14,49-game.y*.12,8-game.y*.02,'#b4b89b66');
  for(const obstacle of game.obstacles){const x=obstacle.x;
    if(obstacle.type==='cone'){path(ctx,[[x,ground],[x+17,ground-43],[x+35,ground]],'#db9465',1,true);path(ctx,[[x+8,ground-17],[x+27,ground-17]],'#faf0da',6);rect(ctx,x-4,ground-3,43,6,'#b38262');}
    else if(obstacle.type==='bench'){rect(ctx,x+9,ground-27,7,29,'#8c9680');rect(ctx,x+82,ground-27,7,29,'#8c9680');rect(ctx,x,ground-35,100,9,'#aa9675');rect(ctx,x,ground-48,100,8,'#b5a180');}
    else{rect(ctx,x,ground-37,65,39,'#b6ac8d');ellipse(ctx,x+31,ground-39,34,17,'#8f9f77');ellipse(ctx,x+18,ground-47,19,12,'#a2b489');}
  }
  if(game.invincible===0||Math.floor(elapsed*12)%2===0)drawSkater(ctx,285,ground-game.y,game.rider,elapsed,game.y,game.trick,game.trickAge,1.15);
  if(game.status==='playing')for(let i=0;i<3;i++)path(ctx,[[212-i*12,ground-15+i*9],[232-i*9,ground-15+i*9]],'#a9b095',2);
  for(let i=0;i<5;i++){let x=600+i*103,y=90+(i*31)%80;path(ctx,[[x-5,y+2],[x,y],[x+5,y+2]],'#98ab93',1.3);}
}
export function drawPreview(canvas, rider) {
  const ctx=canvas.getContext('2d');ctx.clearRect(0,0,150,132);ellipse(ctx,73,124,42,5,'#a9ad922a');drawSkater(ctx,72,114,rider,0,0,'',1,.95);
}
