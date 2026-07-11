const canvas=document.querySelector("#life")
const ctx=canvas.getContext("2d")
let width=0
let height=0
let current=1
let pointer={x:0,y:0,active:false}
let wakes=[]
const palette=["#f5cc59","#dce9e3","#56cbd0","#f28c72","#163a4d"]
const fish=Array.from({length:34},(_,i)=>({x:Math.random(),y:.2+Math.random()*.58,size:5+Math.random()*17,speed:.00015+Math.random()*.00038,direction:Math.random()>.18?1:-1,color:palette[i%palette.length],phase:Math.random()*6.28}))
const bubbles=Array.from({length:38},()=>({x:Math.random(),y:Math.random(),size:1+Math.random()*3,speed:.0001+Math.random()*.0003}))
function resize(){width=canvas.width=innerWidth*devicePixelRatio;height=canvas.height=innerHeight*devicePixelRatio;ctx.scale(devicePixelRatio,devicePixelRatio);width=innerWidth;height=innerHeight}
function drawFish(f,t){const x=f.x*width,y=f.y*height+Math.sin(t*.001+f.phase)*5,s=f.size,d=f.direction;ctx.save();ctx.translate(x,y);ctx.scale(d,1);ctx.globalAlpha=.42;ctx.fillStyle=f.color;ctx.beginPath();ctx.ellipse(0,0,s*1.4,s*.5,0,0,Math.PI*2);ctx.fill();ctx.beginPath();ctx.moveTo(-s*1.25,0);ctx.lineTo(-s*2,-s*.8);ctx.lineTo(-s*1.85,s*.75);ctx.closePath();ctx.fill();ctx.restore()}
function render(t){ctx.clearRect(0,0,width,height);bubbles.forEach(b=>{b.y-=b.speed*(current?1:.35);if(b.y<-.02){b.y=1.02;b.x=Math.random()}ctx.strokeStyle="rgba(220,255,250,.32)";ctx.lineWidth=.6;ctx.beginPath();ctx.arc(b.x*width,b.y*height,b.size,0,Math.PI*2);ctx.stroke()});wakes=wakes.filter(w=>w.life>0);wakes.forEach(w=>{w.life-=.025;w.radius+=1.8;ctx.strokeStyle=`rgba(190,255,248,${w.life*.42})`;ctx.lineWidth=1;ctx.beginPath();ctx.ellipse(w.x,w.y,w.radius,w.radius*.42,0,0,Math.PI*2);ctx.stroke()});fish.forEach(f=>{f.x+=f.speed*f.direction*(current?1:.2);if(f.x>1.12)f.x=-.12;if(f.x<-.12)f.x=1.12;if(pointer.active){const dx=f.x*width-pointer.x,dy=f.y*height-pointer.y,dist=Math.hypot(dx,dy);if(dist<180&&dist>0){const force=(180-dist)/180;f.x+=dx/dist*force*.006;f.y+=dy/dist*force*.004;f.direction=dx>0?1:-1}}drawFish(f,t)});requestAnimationFrame(render)}
addEventListener("resize",resize)
addEventListener("pointermove",e=>{pointer={x:e.clientX,y:e.clientY,active:true};if(!wakes.length||Math.hypot(e.clientX-wakes[wakes.length-1].x,e.clientY-wakes[wakes.length-1].y)>28)wakes.push({x:e.clientX,y:e.clientY,radius:4,life:1})})
addEventListener("pointerleave",()=>pointer.active=false)
function descend(){const range=document.documentElement.scrollHeight-innerHeight,progress=range?Math.min(1,scrollY/range):0,depth=12.4+progress*167.6,aquarium=document.querySelector("#aquarium"),backdrop=document.querySelector(".backdrop"),hero=document.querySelector(".hero"),card=document.querySelector("#species-card"),zone=document.querySelector(".coordinates span");document.querySelector("#depth-value").textContent=`${depth.toFixed(1)} M`;aquarium.style.setProperty("--descent",String(progress));backdrop.style.transform=`scale(${1+progress*.55}) translateY(${-progress*18}%)`;backdrop.style.filter=`brightness(${1-progress*.68}) saturate(${1-progress*.25})`;hero.style.opacity=String(Math.max(0,1-progress*2.2));hero.style.transform=`translateY(${-progress*140}px)`;document.querySelector(".caustics").style.opacity=String(Math.max(0,.14-progress*.18));card.classList.toggle("open",progress>.24&&progress<.72);zone.textContent=progress<.25?"SUNLIT REEF · LIVE CURRENT":progress<.6?"TWILIGHT WATER · DESCENDING":"DEEP PACIFIC · LOW LIGHT"}
addEventListener("scroll",descend,{passive:true})
addEventListener("resize",descend)
document.querySelector("#current").addEventListener("click",e=>{current=current?0:1;e.currentTarget.setAttribute("aria-pressed",String(Boolean(current)));e.currentTarget.querySelector("b").textContent=current?"GENTLE":"STILL"})
document.querySelector("#dive").addEventListener("click",()=>scrollTo({top:document.body.scrollHeight,behavior:"smooth"}))
document.querySelector("#close-card").addEventListener("click",()=>document.querySelector("#species-card").classList.remove("open"))
let audio
document.querySelector("#sound").addEventListener("click",e=>{const on=e.currentTarget.classList.toggle("on");e.currentTarget.setAttribute("aria-pressed",String(on));e.currentTarget.lastElementChild.textContent=on?"SOUND ON":"SOUND OFF";if(!audio){audio=new AudioContext();const gain=audio.createGain();gain.gain.value=.018;const filter=audio.createBiquadFilter();filter.type="lowpass";filter.frequency.value=380;const buffer=audio.createBuffer(1,audio.sampleRate*2,audio.sampleRate);const data=buffer.getChannelData(0);for(let i=0;i<data.length;i++)data[i]=Math.random()*2-1;const source=audio.createBufferSource();source.buffer=buffer;source.loop=true;source.connect(filter).connect(gain).connect(audio.destination);source.start();e.currentTarget.audioGain=gain}else e.currentTarget.audioGain.gain.setTargetAtTime(on?.018:0,audio.currentTime,.3)})
resize()
requestAnimationFrame(render)
if(location.hash==="#species")document.querySelector("#species-card").classList.add("open")
