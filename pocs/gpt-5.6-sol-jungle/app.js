const canvas=document.querySelector("#atmosphere")
const gl=canvas.getContext("webgl",{alpha:true,antialias:false})
const jungle=document.querySelector("#jungle")
const hero=document.querySelector(".hero")
const story=document.querySelector("#story")
const rain=document.querySelector("#rain")
const soundButton=document.querySelector("#sound")
const chapters=[
  {title:"I · IGAPÓ",text:"Where the river enters the forest, roots learn the rhythm of the water.",zone:"FLOODED FOREST · MORNING",creature:"snake",tone:"dawn"},
  {title:"II · TERRA FIRME",text:"Under the high canopy, an unseen path belongs to the onça-pintada.",zone:"HIGH FOREST · AFTERNOON",creature:"jaguar",tone:"day"},
  {title:"III · THE RAIN",text:"Warm rain returns everything it borrows to leaf, river, and root.",zone:"RIVER MIST · RAINFALL",creature:"",tone:"rain"},
  {title:"IV · ANCESTRAL GROUND",text:"The forest is home, memory, medicine, and living relation.",zone:"ANCESTRAL FOREST · DUSK",creature:"guardian",tone:"dusk"},
  {title:"V · NOCTURNE",text:"When the light leaves, a thousand smaller constellations wake.",zone:"CANOPY EDGE · NIGHT",creature:"",tone:"night"}
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
let lastTime=0
let shaderTime=0
function shader(type,source){const item=gl.createShader(type);gl.shaderSource(item,source);gl.compileShader(item);return item}
let program
let timeLocation
let resolutionLocation
function setupWebGL(){if(!gl)return;const vertex=shader(gl.VERTEX_SHADER,"attribute vec2 p;void main(){gl_Position=vec4(p,0.,1.);}");const fragment=shader(gl.FRAGMENT_SHADER,"precision mediump float;uniform vec2 r;uniform float t;float h(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}void main(){vec2 u=gl_FragCoord.xy/r;float n=h(floor(u*vec2(34.,20.)));float fire=step(.992,n)*(.45+.55*sin(t*2.+n*19.))*smoothstep(.05,.8,u.y);float mist=(sin(u.x*8.+t*.08)+sin(u.y*7.-t*.05))*0.025;gl_FragColor=vec4(.65,.9,.5,fire*.65+mist*.12);}");program=gl.createProgram();gl.attachShader(program,vertex);gl.attachShader(program,fragment);gl.linkProgram(program);gl.useProgram(program);const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),gl.STATIC_DRAW);const position=gl.getAttribLocation(program,"p");gl.enableVertexAttribArray(position);gl.vertexAttribPointer(position,2,gl.FLOAT,false,0,0);timeLocation=gl.getUniformLocation(program,"t");resolutionLocation=gl.getUniformLocation(program,"r")}
function resize(){const ratio=Math.min(devicePixelRatio,2);canvas.width=innerWidth*ratio;canvas.height=innerHeight*ratio;if(gl)gl.viewport(0,0,canvas.width,canvas.height)}
function render(time){const delta=Math.min(32,time-lastTime);lastTime=time;if(!paused)shaderTime+=delta*.001;if(gl&&program){gl.uniform1f(timeLocation,shaderTime);gl.uniform2f(resolutionLocation,canvas.width,canvas.height);gl.drawArrays(gl.TRIANGLES,0,6)}requestAnimationFrame(render)}
function setRain(value,source="automatic"){raining=value;automaticRain=source==="automatic"&&value;rain.classList.toggle("active",raining);document.querySelector("#weather").setAttribute("aria-pressed",String(raining));document.querySelector("#weather b").textContent=raining?"RAINING":"HUMID";if(audio&&rainGain)rainGain.gain.setTargetAtTime(raining?.09:0,audio.currentTime,.8)}
function showNotice(text){const notice=document.querySelector("#notice");notice.textContent=text;notice.classList.add("show");clearTimeout(noticeTimer);noticeTimer=setTimeout(()=>notice.classList.remove("show"),2400)}
function updateScene(){const span=innerHeight*.82;const raw=Math.max(0,journey/span);const index=Math.floor(raw)%chapters.length;const local=raw-Math.floor(raw);const chapter=chapters[index];const night=chapter.tone==="night"?.58:chapter.tone==="dusk"?.28:0;const brightness=chapter.tone==="night"?.38:chapter.tone==="dusk"?.58:chapter.tone==="rain"?.55:.78;const heroFade=Math.min(1,journey/(innerHeight*.7));hero.style.opacity=String(1-heroFade);hero.style.transform=`translateY(${-heroFade*90}px)`;story.classList.toggle("visible",journey>innerHeight*.55);document.querySelector("#chapter").textContent=chapter.title;document.querySelector("#story-text").textContent=chapter.text;document.querySelector("#zone").textContent=chapter.zone;document.querySelector("#distance").textContent=`${String(Math.floor(journey*.18)).padStart(3,"0")} M`;jungle.style.setProperty("--scale",String(1.03+local*.12));jungle.style.setProperty("--drift-x",`${Math.sin(raw*2.1)*2}%`);jungle.style.setProperty("--drift-y",`${-local*2}%`);jungle.style.setProperty("--brightness",String(brightness));jungle.style.setProperty("--saturation",chapter.tone==="rain"?".74":"1.1");jungle.style.setProperty("--hue",chapter.tone==="night"?"20deg":"0deg");jungle.style.setProperty("--night",String(night));jungle.style.setProperty("--light-opacity",String(1-night*1.4));jungle.style.setProperty("--journey",`${(local*140)-15}px`);["snake","jaguar","guardian"].forEach(id=>document.querySelector(`#${id}`).classList.toggle("visible",chapter.creature===id));if(chapter.tone==="rain"&&!raining)setRain(true);if(chapter.tone!=="rain"&&automaticRain)setRain(false)}
function recenter(){const center=innerHeight*2;scrollTo(0,center);previousScroll=center}
function onScroll(){if(paused)return;const center=innerHeight*2;const delta=scrollY-previousScroll;journey=Math.max(0,journey+delta);previousScroll=scrollY;if(scrollY<innerHeight*.7||scrollY>innerHeight*3.3)recenter();updateScene()}
function noiseSource(context,seconds=2){const buffer=context.createBuffer(1,context.sampleRate*seconds,context.sampleRate);const data=buffer.getChannelData(0);let brown=0;for(let i=0;i<data.length;i++){brown=(brown+(Math.random()*2-1)*.04)/1.02;data[i]=brown*3.5}const source=context.createBufferSource();source.buffer=buffer;source.loop=true;return source}
function callBird(){if(!audio||audio.state!=="running")return;const now=audio.currentTime;const oscillator=audio.createOscillator();const gain=audio.createGain();oscillator.type="sine";oscillator.frequency.setValueAtTime(1400+Math.random()*600,now);oscillator.frequency.exponentialRampToValueAtTime(2100+Math.random()*800,now+.14);oscillator.frequency.exponentialRampToValueAtTime(1200,now+.38);gain.gain.setValueAtTime(0,now);gain.gain.linearRampToValueAtTime(.018,now+.04);gain.gain.exponentialRampToValueAtTime(.001,now+.4);oscillator.connect(gain).connect(master);oscillator.start(now);oscillator.stop(now+.42)}
function startSound(){if(audio){audio.resume();master.gain.setTargetAtTime(.6,audio.currentTime,.4);return}audio=new AudioContext();master=audio.createGain();master.gain.value=.6;master.connect(audio.destination);waterGain=audio.createGain();waterGain.gain.value=.035;const waterFilter=audio.createBiquadFilter();waterFilter.type="lowpass";waterFilter.frequency.value=420;rainGain=audio.createGain();rainGain.gain.value=raining?.09:0;const rainFilter=audio.createBiquadFilter();rainFilter.type="highpass";rainFilter.frequency.value=1100;const drone=audio.createOscillator();const droneGain=audio.createGain();drone.type="sine";drone.frequency.value=73;droneGain.gain.value=.008;drone.connect(droneGain).connect(master);drone.start();const sources=[noiseSource(audio),noiseSource(audio)];sources[0].connect(waterFilter).connect(waterGain).connect(master);sources[1].connect(rainFilter).connect(rainGain).connect(master);sources.forEach(source=>source.start());birdTimer=setInterval(callBird,4200)}
function toggleSound(){const enabled=!soundButton.classList.contains("on");soundButton.classList.toggle("on",enabled);soundButton.setAttribute("aria-pressed",String(enabled));soundButton.setAttribute("aria-label",enabled?"Turn forest sound off":"Turn forest sound on");soundButton.lastElementChild.textContent=enabled?"SOUND ON":"SOUND OFF";if(enabled)startSound();else if(audio)master.gain.setTargetAtTime(0,audio.currentTime,.35)}
function interact(id,className,message){const element=document.querySelector(`#${id}`);element.classList.remove(className);requestAnimationFrame(()=>element.classList.add(className));showNotice(message);if(!soundButton.classList.contains("on"))toggleSound();callBird()}
addEventListener("resize",()=>{resize();recenter();updateScene()})
addEventListener("scroll",onScroll,{passive:true})
addEventListener("pointermove",event=>{jungle.style.setProperty("--light-x",`${(event.clientX/innerWidth-.5)*4}%`)},{passive:true})
document.querySelector("#begin").addEventListener("click",()=>{if(!soundButton.classList.contains("on"))toggleSound();journey+=innerHeight*.9;updateScene();showNotice("The forest is listening")})
soundButton.addEventListener("click",toggleSound)
document.querySelector("#fullscreen").addEventListener("click",async()=>{if(!document.fullscreenElement)await document.documentElement.requestFullscreen();else await document.exitFullscreen()})
document.addEventListener("fullscreenchange",()=>{document.querySelector("#fullscreen b").textContent=document.fullscreenElement?"EXIT FULL SCREEN":"FULL SCREEN"})
document.querySelector("#weather").addEventListener("click",()=>setRain(!raining,"manual"))
document.querySelector("#pause").addEventListener("click",event=>{paused=!paused;jungle.classList.toggle("still",paused);event.currentTarget.classList.toggle("paused",paused);event.currentTarget.setAttribute("aria-pressed",String(paused));event.currentTarget.querySelector("b").textContent=paused?"START":"STOP";if(audio&&soundButton.classList.contains("on"))master.gain.setTargetAtTime(paused?0:.6,audio.currentTime,.35);showNotice(paused?"The forest is resting":"The forest breathes again")})
document.querySelector("#jaguar").addEventListener("click",()=>interact("jaguar","awake","A low call travels beneath the canopy"))
document.querySelector("#snake").addEventListener("click",()=>interact("snake","slither","Bright bands vanish beneath the leaves"))
document.querySelector("#guardian").addEventListener("click",()=>interact("guardian","greet","Walk gently. Everything here is alive."))
setInterval(()=>{if(!paused&&!raining){setRain(true);showNotice("A warm rain crosses the river");setTimeout(()=>{if(automaticRain)setRain(false)},18000)}},52000)
setupWebGL()
resize()
recenter()
updateScene()
requestAnimationFrame(render)
