const canvas=document.querySelector("#life")
const ctx=canvas.getContext("2d")
const species=[
  {name:"Moorish idol",latin:"Zanclus cornutus",image:"moorish-idol.png",depth:"3–180 m",status:"STABLE",description:"A reef wanderer recognized by its long, trailing dorsal filament."},
  {name:"Yellow tang",latin:"Zebrasoma flavescens",image:"yellow-tang.png",depth:"2–46 m",status:"STABLE",description:"A bright reef grazer that helps keep coral surfaces clear of algae."},
  {name:"Clown triggerfish",latin:"Balistoides conspicillum",image:"clown-triggerfish.png",depth:"1–75 m",status:"STABLE",description:"A bold solitary hunter with one of the reef's most unmistakable patterns."},
  {name:"Emperor angelfish",latin:"Pomacanthus imperator",image:"emperor-angelfish.png",depth:"1–100 m",status:"STABLE",description:"An electric-blue reef icon whose pattern transforms completely with age."},
  {name:"Copperband butterflyfish",latin:"Chelmon rostratus",image:"copperband-butterflyfish.png",depth:"1–25 m",status:"STABLE",description:"Its long snout reaches small prey hidden deep inside coral crevices."},
  {name:"Regal angelfish",latin:"Pygoplites diacanthus",image:"regal-angelfish.png",depth:"1–80 m",status:"STABLE",description:"A shy coral specialist dressed in precise orange, white, and blue bands."},
  {name:"Blue-green chromis",latin:"Chromis viridis",image:"blue-green-chromis.png",depth:"1–12 m",status:"STABLE",description:"A luminous schooling fish that gathers above branching Pacific corals."},
  {name:"Picasso triggerfish",latin:"Rhinecanthus aculeatus",image:"picasso-triggerfish.png",depth:"1–50 m",status:"STABLE",description:"A territorial lagoon resident marked with vivid geometric lines."},
  {name:"Foxface rabbitfish",latin:"Siganus vulpinus",image:"foxface-rabbitfish.png",depth:"1–30 m",status:"STABLE",description:"A peaceful algae grazer protected by venomous dorsal fin spines."},
  {name:"Achilles tang",latin:"Acanthurus achilles",image:"achilles-tang.png",depth:"0–20 m",status:"STABLE",description:"A powerful swimmer found where Pacific surge breaks across the reef."},
  {name:"Red lionfish",latin:"Pterois volitans",image:"red-lionfish.png",depth:"1–55 m",status:"EXPANDING",description:"A patient ambush predator carrying venom in its ornate dorsal spines."},
  {name:"Pajama cardinalfish",latin:"Sphaeramia nematoptera",image:"pajama-cardinalfish.png",depth:"1–14 m",status:"STABLE",description:"A nocturnal coral dweller with red eyes and a constellation of spots."}
]
const spriteImages=species.map(item=>{const image=new Image();image.src=`assets/${item.image}`;return image})
let width=0
let height=0
let current=1
let pointer={x:0,y:0,active:false}
let wakes=[]
let speciesIndex=0
let audio
let audioGain
let soundEnabled=true
const fish=Array.from({length:26},(_,i)=>({x:Math.random(),y:.15+Math.random()*.67,size:24+Math.random()*34,speed:.00012+Math.random()*.00032,direction:Math.random()>.18?1:-1,sprite:i%species.length,phase:Math.random()*6.28,alpha:.38+Math.random()*.42}))
const bubbles=Array.from({length:38},()=>({x:Math.random(),y:Math.random(),size:1+Math.random()*3,speed:.0001+Math.random()*.0003}))
function resize(){width=canvas.width=innerWidth*devicePixelRatio;height=canvas.height=innerHeight*devicePixelRatio;ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);width=innerWidth;height=innerHeight}
function drawFish(f,t){const image=spriteImages[f.sprite];if(!image.complete)return;const x=f.x*width,y=f.y*height+Math.sin(t*.001+f.phase)*7,s=f.size;ctx.save();ctx.translate(x,y);ctx.scale(f.direction,1);ctx.globalAlpha=f.alpha;ctx.filter=`blur(${Math.max(0,(1-f.alpha)*1.3)}px) drop-shadow(0 8px 9px rgba(0,12,25,.28))`;ctx.drawImage(image,-s*1.6,-s*1.6,s*3.2,s*3.2);ctx.restore()}
function render(t){ctx.clearRect(0,0,width,height);bubbles.forEach(b=>{b.y-=b.speed*(current?1:.35);if(b.y<-.02){b.y=1.02;b.x=Math.random()}ctx.strokeStyle="rgba(220,255,250,.32)";ctx.lineWidth=.6;ctx.beginPath();ctx.arc(b.x*width,b.y*height,b.size,0,Math.PI*2);ctx.stroke()});wakes=wakes.filter(w=>w.life>0);wakes.forEach(w=>{w.life-=.025;w.radius+=1.8;ctx.strokeStyle=`rgba(190,255,248,${w.life*.42})`;ctx.lineWidth=1;ctx.beginPath();ctx.ellipse(w.x,w.y,w.radius,w.radius*.42,0,0,Math.PI*2);ctx.stroke()});fish.forEach(f=>{f.x+=f.speed*f.direction*(current?1:.2);if(f.x>1.12)f.x=-.12;if(f.x<-.12)f.x=1.12;if(pointer.active){const dx=f.x*width-pointer.x,dy=f.y*height-pointer.y,dist=Math.hypot(dx,dy);if(dist<180&&dist>0){const force=(180-dist)/180;f.x+=dx/dist*force*.006;f.y+=dy/dist*force*.004;f.direction=dx>0?1:-1}}drawFish(f,t)});requestAnimationFrame(render)}
function showSpecies(index){speciesIndex=(index+species.length)%species.length;const item=species[speciesIndex];document.querySelector("#species-number").textContent=`${String(speciesIndex+1).padStart(2,"0")} / ${species.length}`;const image=document.querySelector("#species-image");image.src=`assets/${item.image}`;image.alt=`${item.name} fish`;document.querySelector("#species-latin").textContent=item.latin;document.querySelector("#species-name").textContent=item.name;document.querySelector("#species-description").textContent=item.description;document.querySelector("#species-depth").textContent=item.depth;document.querySelector("#species-status").textContent=item.status}
function descend(){if(scrollY+innerHeight>document.documentElement.scrollHeight-innerHeight*.5)document.body.style.height=`${document.documentElement.scrollHeight+innerHeight*3}px`;const progress=Math.min(1,scrollY/(innerHeight*3)),depth=12.4+scrollY*.08,aquarium=document.querySelector("#aquarium"),backdrop=document.querySelector(".backdrop"),hero=document.querySelector(".hero"),card=document.querySelector("#species-card"),zone=document.querySelector(".coordinates span"),nextIndex=Math.floor(scrollY/(innerHeight*.55))%species.length;document.querySelector("#depth-value").textContent=depth<1000?`${depth.toFixed(1)} M`:`${(depth/1000).toFixed(2)} KM`;aquarium.style.setProperty("--descent",String(progress));backdrop.style.transform=`scale(${1+progress*.55}) translateY(${-progress*18}%)`;backdrop.style.filter=`brightness(${1-progress*.68}) saturate(${1-progress*.25})`;hero.style.opacity=String(Math.max(0,1-progress*2.2));hero.style.transform=`translateY(${-progress*140}px)`;document.querySelector(".caustics").style.opacity=String(Math.max(0,.14-progress*.18));card.classList.toggle("open",scrollY>innerHeight*.28);if(nextIndex!==speciesIndex)showSpecies(nextIndex);zone.textContent=progress<.25?"SUNLIT REEF · LIVE CURRENT":progress<.6?"TWILIGHT WATER · DESCENDING":"DEEP PACIFIC · CONTINUING"}
function startOcean(){if(!soundEnabled)return;if(audio){audio.resume();audioGain.gain.setTargetAtTime(.018,audio.currentTime,.25);return}audio=new AudioContext();audioGain=audio.createGain();audioGain.gain.value=.018;const filter=audio.createBiquadFilter();filter.type="lowpass";filter.frequency.value=380;const buffer=audio.createBuffer(1,audio.sampleRate*2,audio.sampleRate),data=buffer.getChannelData(0);for(let i=0;i<data.length;i++)data[i]=Math.random()*2-1;const source=audio.createBufferSource();source.buffer=buffer;source.loop=true;source.connect(filter).connect(audioGain).connect(audio.destination);source.start()}
addEventListener("resize",()=>{resize();descend()})
addEventListener("scroll",descend,{passive:true})
addEventListener("pointermove",e=>{startOcean();pointer={x:e.clientX,y:e.clientY,active:true};if(!wakes.length||Math.hypot(e.clientX-wakes[wakes.length-1].x,e.clientY-wakes[wakes.length-1].y)>28)wakes.push({x:e.clientX,y:e.clientY,radius:4,life:1})},{passive:true})
addEventListener("pointerleave",()=>pointer.active=false)
addEventListener("wheel",startOcean,{once:true,passive:true})
addEventListener("touchstart",startOcean,{once:true,passive:true})
document.querySelector("#current").addEventListener("click",e=>{current=current?0:1;e.currentTarget.setAttribute("aria-pressed",String(Boolean(current)));e.currentTarget.querySelector("b").textContent=current?"GENTLE":"STILL"})
document.querySelector("#dive").addEventListener("click",()=>scrollTo({top:innerHeight*1.25,behavior:"smooth"}))
document.querySelector("#close-card").addEventListener("click",()=>document.querySelector("#species-card").classList.remove("open"))
document.querySelector("#previous-species").addEventListener("click",()=>showSpecies(speciesIndex-1))
document.querySelector("#next-species").addEventListener("click",()=>showSpecies(speciesIndex+1))
document.querySelector("#sound").addEventListener("click",e=>{soundEnabled=!soundEnabled;e.currentTarget.classList.toggle("on",soundEnabled);e.currentTarget.setAttribute("aria-pressed",String(soundEnabled));e.currentTarget.setAttribute("aria-label",soundEnabled?"Turn ocean sound off":"Turn ocean sound on");e.currentTarget.lastElementChild.textContent=soundEnabled?"SOUND ON":"SOUND OFF";if(soundEnabled)startOcean();else if(audio)audioGain.gain.setTargetAtTime(0,audio.currentTime,.25)})
resize()
showSpecies(0)
descend()
requestAnimationFrame(render)
if(location.hash==="#species")document.querySelector("#species-card").classList.add("open")
