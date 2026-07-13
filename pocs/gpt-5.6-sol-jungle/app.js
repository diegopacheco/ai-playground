const jungle=document.querySelector("#jungle")
const backdrops=[...document.querySelectorAll(".backdrop")]
const backgrounds=["amazon-river.png","jungle-02-lianas.png","jungle-03-ferns.png","jungle-04-woods.png","jungle-05-after-rain.png","jungle-06-roots.png","jungle-07-golden.png"]
const hero=document.querySelector(".hero")
const story=document.querySelector("#story")
const rain=document.querySelector("#rain")
const rainContext=rain.getContext("2d")
const soundButton=document.querySelector("#sound")
const chapters=[
  {title:"I · IGAPÓ",text:"Where the river enters the forest, roots learn the rhythm of the water.",zone:"FLOODED FOREST · MORNING",creature:"snake",tone:"dawn"},
  {title:"II · TERRA FIRME",text:"Under the high canopy, an unseen path belongs to the onça-pintada.",zone:"HIGH FOREST · AFTERNOON",creature:"jaguar",tone:"day"},
  {title:"III · SAMAMBAIA",text:"Fern banks gather the river's cool breath beneath a thousand green fronds.",zone:"FERN BANK · MIDDAY",creature:"",tone:"day"},
  {title:"IV · THE RAIN",text:"Warm rain returns everything it borrows to leaf, river, and root.",zone:"OLD WOODS · RAINFALL",creature:"",tone:"rain"},
  {title:"V · ANCESTRAL GROUND",text:"The forest is home, memory, medicine, and living relation.",zone:"WET FOREST · DUSK",creature:"guardian",tone:"dusk"},
  {title:"VI · FLECHEIRA",text:"She reads distance in moving leaves and guards the quiet river path.",zone:"BUTTRESS ROOTS · LATE LIGHT",creature:"archer",tone:"dusk"},
  {title:"VII · NOCTURNE",text:"When the light leaves, the river carries the last gold beneath the canopy.",zone:"DEEP CANOPY · NIGHT",creature:"",tone:"night"}
]
let journey=0
let previousScroll=0
let paused=false
let raining=false
let automaticRain=false
let audio
let master
let rainGain
let waterGain
let birdTimer
let noticeTimer
let drops=[]
let backgroundIndex=0
let backdropLayer=0
function makeDrop(){const depth=.25+Math.random()*.75;return{x:Math.random()*innerWidth,y:Math.random()*innerHeight,length:8+depth*20,speed:7+depth*16,depth}}
function resize(){const ratio=Math.min(devicePixelRatio,2);rain.width=innerWidth*ratio;rain.height=innerHeight*ratio;rainContext.setTransform(ratio,0,0,ratio,0,0);drops=Array.from({length:Math.floor(innerWidth*.17)},makeDrop)}
function drawRain(){rainContext.clearRect(0,0,innerWidth,innerHeight);if(!raining)return;rainContext.lineCap="round";drops.forEach(drop=>{drop.y+=drop.speed;drop.x-=drop.speed*.14;if(drop.y>innerHeight+30||drop.x<0){drop.y=-drop.length;drop.x=Math.random()*innerWidth+40}const fade=.09+drop.depth*.23;const gradient=rainContext.createLinearGradient(drop.x,drop.y,drop.x+3,drop.y+drop.length);gradient.addColorStop(0,"rgba(224,235,229,0)");gradient.addColorStop(1,`rgba(224,235,229,${fade})`);rainContext.strokeStyle=gradient;rainContext.lineWidth=.45+drop.depth*.7;rainContext.beginPath();rainContext.moveTo(drop.x,drop.y);rainContext.lineTo(drop.x-3,drop.y+drop.length);rainContext.stroke()});rainContext.fillStyle="rgba(204,218,211,.025)";rainContext.fillRect(0,innerHeight*.68,innerWidth,innerHeight*.32)}
function render(){if(!paused)drawRain();requestAnimationFrame(render)}
function setRain(value,source="automatic"){raining=value;automaticRain=source==="automatic"&&value;rain.classList.toggle("active",raining);document.querySelector("#weather").setAttribute("aria-pressed",String(raining));document.querySelector("#weather b").textContent=raining?"RAINING":"HUMID";if(audio&&rainGain)rainGain.gain.setTargetAtTime(raining?.045:0,audio.currentTime,.8)}
function showNotice(text){const notice=document.querySelector("#notice");notice.textContent=text;notice.classList.add("show");clearTimeout(noticeTimer);noticeTimer=setTimeout(()=>notice.classList.remove("show"),2400)}
function updateScene(){const span=innerHeight*.82;const raw=Math.max(0,journey/span);const index=Math.floor(raw)%chapters.length;const local=raw-Math.floor(raw);const chapter=chapters[index];if(index!==backgroundIndex){const nextLayer=1-backdropLayer;backdrops[nextLayer].style.backgroundImage=`url("assets/${backgrounds[index]}")`;backdrops[nextLayer].style.opacity="1";backdrops[backdropLayer].style.opacity="0";backdropLayer=nextLayer;backgroundIndex=index}const night=chapter.tone==="night"?.48:chapter.tone==="dusk"?.18:0;const brightness=chapter.tone==="night"?.48:chapter.tone==="dusk"?.72:chapter.tone==="rain"?.68:.82;const heroFade=Math.min(1,journey/(innerHeight*.7));hero.style.opacity=String(1-heroFade);hero.style.transform=`translateY(${-heroFade*90}px)`;story.classList.toggle("visible",journey>innerHeight*.55);document.querySelector("#chapter").textContent=chapter.title;document.querySelector("#story-text").textContent=chapter.text;document.querySelector("#zone").textContent=chapter.zone;document.querySelector("#distance").textContent=`${String(Math.floor(journey*.18)).padStart(3,"0")} M`;jungle.style.setProperty("--scale",String(1.03+local*.08));jungle.style.setProperty("--drift-x",`${Math.sin(raw*2.1)*1.2}%`);jungle.style.setProperty("--drift-y",`${-local*1.2}%`);jungle.style.setProperty("--brightness",String(brightness));jungle.style.setProperty("--saturation",chapter.tone==="rain"?".86":"1.04");jungle.style.setProperty("--hue",chapter.tone==="night"?"12deg":"0deg");jungle.style.setProperty("--night",String(night));jungle.style.setProperty("--journey",`${(local*140)-15}px`);["snake","jaguar","guardian","archer"].forEach(id=>document.querySelector(`#${id}`).classList.toggle("visible",chapter.creature===id));if(chapter.tone==="rain"&&!raining)setRain(true);if(chapter.tone!=="rain"&&automaticRain)setRain(false)}
function recenter(){const center=innerHeight*2;scrollTo(0,center);previousScroll=center}
function onScroll(){if(paused)return;const center=innerHeight*2;const delta=scrollY-previousScroll;journey=Math.max(0,journey+delta);previousScroll=scrollY;if(scrollY<innerHeight*.7||scrollY>innerHeight*3.3)recenter();updateScene()}
function noiseSource(context,seconds=2){const buffer=context.createBuffer(1,context.sampleRate*seconds,context.sampleRate);const data=buffer.getChannelData(0);let brown=0;for(let i=0;i<data.length;i++){brown=(brown+(Math.random()*2-1)*.04)/1.02;data[i]=brown*3.5}const source=context.createBufferSource();source.buffer=buffer;source.loop=true;return source}
function spatialNode(pan){if(!audio.createStereoPanner)return master;const node=audio.createStereoPanner();node.pan.value=pan;node.connect(master);return node}
function forestTone(frequency,start,duration,volume,pan,curve=1.12){const oscillator=audio.createOscillator();const gain=audio.createGain();oscillator.type="sine";oscillator.frequency.setValueAtTime(frequency,start);oscillator.frequency.exponentialRampToValueAtTime(frequency*curve,start+duration*.72);gain.gain.setValueAtTime(.0001,start);gain.gain.exponentialRampToValueAtTime(volume,start+duration*.16);gain.gain.exponentialRampToValueAtTime(.0001,start+duration);oscillator.connect(gain).connect(spatialNode(pan));oscillator.start(start);oscillator.stop(start+duration+.02)}
function callBird(){if(!audio||audio.state!=="running"||paused)return;const now=audio.currentTime+.04;const pan=Math.random()*1.6-.8;const call=Math.floor(Math.random()*3);if(call===0){const base=1850+Math.random()*480;[0,.16,.34].forEach((offset,index)=>forestTone(base+index*210,now+offset,.22,.012,pan,1.28))}else if(call===1){const base=2550+Math.random()*650;[0,.1,.2,.42].forEach((offset,index)=>forestTone(base-index*130,now+offset,.16,.009,pan,.86))}else{const base=1150+Math.random()*300;forestTone(base,now,.7,.014,pan,1.9);forestTone(base*1.45,now+.18,.48,.008,pan,1.18)}}
function callFrog(){if(!audio||audio.state!=="running"||paused)return;const now=audio.currentTime+.04;const pan=Math.random()*1.4-.7;[0,.22].forEach(offset=>{const oscillator=audio.createOscillator();const gain=audio.createGain();oscillator.type="triangle";oscillator.frequency.setValueAtTime(138+Math.random()*22,now+offset);oscillator.frequency.exponentialRampToValueAtTime(92,now+offset+.28);gain.gain.setValueAtTime(.0001,now+offset);gain.gain.exponentialRampToValueAtTime(.009,now+offset+.05);gain.gain.exponentialRampToValueAtTime(.0001,now+offset+.31);oscillator.connect(gain).connect(spatialNode(pan));oscillator.start(now+offset);oscillator.stop(now+offset+.34)})}
function rustleLeaves(){if(!audio||audio.state!=="running"||paused)return;const duration=.8+Math.random()*.7;const buffer=audio.createBuffer(1,Math.floor(audio.sampleRate*duration),audio.sampleRate);const data=buffer.getChannelData(0);for(let i=0;i<data.length;i++){const position=i/data.length;data[i]=(Math.random()*2-1)*Math.sin(Math.PI*position)*(.4+Math.random()*.6)}const source=audio.createBufferSource();const filter=audio.createBiquadFilter();const gain=audio.createGain();const now=audio.currentTime;source.buffer=buffer;filter.type="bandpass";filter.frequency.value=2100+Math.random()*1200;filter.Q.value=.7;gain.gain.setValueAtTime(.0001,now);gain.gain.exponentialRampToValueAtTime(.006,now+duration*.3);gain.gain.exponentialRampToValueAtTime(.0001,now+duration);source.connect(filter).connect(gain).connect(spatialNode(Math.random()*1.8-.9));source.start(now);source.stop(now+duration)}
function knockWood(){if(!audio||audio.state!=="running"||paused)return;const now=audio.currentTime+.03;const pan=Math.random()*1.6-.8;[0,.13,.29].forEach((offset,index)=>forestTone(520-index*72,now+offset,.055,.006,pan,.72))}
function scheduleBird(){clearTimeout(birdTimer);birdTimer=setTimeout(()=>{callBird();scheduleBird()},2400+Math.random()*4200)}
function startSound(){if(audio){audio.resume();master.gain.setTargetAtTime(.72,audio.currentTime,.4);return}audio=new AudioContext();master=audio.createGain();master.gain.value=.72;master.connect(audio.destination);waterGain=audio.createGain();waterGain.gain.value=.0035;const waterFilter=audio.createBiquadFilter();waterFilter.type="lowpass";waterFilter.frequency.value=360;rainGain=audio.createGain();rainGain.gain.value=raining?.045:0;const rainFilter=audio.createBiquadFilter();rainFilter.type="highpass";rainFilter.frequency.value=1500;const insectsGain=audio.createGain();insectsGain.gain.value=.0015;const insectsFilter=audio.createBiquadFilter();insectsFilter.type="bandpass";insectsFilter.frequency.value=5100;insectsFilter.Q.value=9;const pulse=audio.createOscillator();const pulseDepth=audio.createGain();pulse.type="sine";pulse.frequency.value=11.7;pulseDepth.gain.value=.0012;pulse.connect(pulseDepth).connect(insectsGain.gain);pulse.start();const sources=[noiseSource(audio),noiseSource(audio),noiseSource(audio)];sources[0].connect(waterFilter).connect(waterGain).connect(master);sources[1].connect(rainFilter).connect(rainGain).connect(master);sources[2].connect(insectsFilter).connect(insectsGain).connect(master);sources.forEach(source=>source.start());scheduleBird();setInterval(callFrog,13000);setInterval(rustleLeaves,5200);setInterval(knockWood,17000)}
function toggleSound(){const enabled=!soundButton.classList.contains("on");soundButton.classList.toggle("on",enabled);soundButton.setAttribute("aria-pressed",String(enabled));soundButton.setAttribute("aria-label",enabled?"Turn forest sound off":"Turn forest sound on");soundButton.lastElementChild.textContent=enabled?"SOUND ON":"SOUND OFF";if(enabled)startSound();else if(audio)master.gain.setTargetAtTime(0,audio.currentTime,.35)}
function interact(id,className,message){const element=document.querySelector(`#${id}`);element.classList.remove(className);requestAnimationFrame(()=>element.classList.add(className));showNotice(message);if(!soundButton.classList.contains("on"))toggleSound();callBird()}
addEventListener("resize",()=>{resize();recenter();updateScene()})
addEventListener("scroll",onScroll,{passive:true})
document.querySelector("#begin").addEventListener("click",()=>{if(!soundButton.classList.contains("on"))toggleSound();journey+=innerHeight*.9;updateScene();showNotice("The forest is listening")})
soundButton.addEventListener("click",toggleSound)
document.querySelector("#fullscreen").addEventListener("click",async()=>{if(!document.fullscreenElement)await document.documentElement.requestFullscreen();else await document.exitFullscreen()})
document.addEventListener("fullscreenchange",()=>{document.querySelector("#fullscreen b").textContent=document.fullscreenElement?"EXIT FULL SCREEN":"FULL SCREEN"})
document.querySelector("#weather").addEventListener("click",()=>setRain(!raining,"manual"))
document.querySelector("#pause").addEventListener("click",event=>{paused=!paused;jungle.classList.toggle("still",paused);event.currentTarget.classList.toggle("paused",paused);event.currentTarget.setAttribute("aria-pressed",String(paused));event.currentTarget.querySelector("b").textContent=paused?"START":"STOP";if(audio&&soundButton.classList.contains("on"))master.gain.setTargetAtTime(paused?0:.72,audio.currentTime,.35);showNotice(paused?"The forest is resting":"The forest breathes again")})
document.querySelector("#jaguar").addEventListener("click",()=>interact("jaguar","awake","A low call travels beneath the canopy"))
document.querySelector("#snake").addEventListener("click",()=>interact("snake","slither","Bright bands vanish beneath the leaves"))
document.querySelector("#guardian").addEventListener("click",()=>interact("guardian","greet","Walk gently. Everything here is alive."))
document.querySelector("#archer").addEventListener("click",()=>interact("archer","aim","The arrow remains still. The forest path is clear."))
setInterval(()=>{if(!paused&&!raining){setRain(true);showNotice("A warm rain crosses the river");setTimeout(()=>{if(automaticRain)setRain(false)},18000)}},52000)
backgrounds.forEach(file=>{const image=new Image();image.src=`assets/${file}`})
resize()
recenter()
updateScene()
requestAnimationFrame(render)
