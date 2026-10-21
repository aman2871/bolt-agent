import * as THREE from 'three';
import { buildModels } from './model.js';
import { damp, turn, resolveDirection, validateState } from './motion.js';
import { pawTrajectory,solveLeg,gallopCycle } from './gait.js';

// --- Bolt dashboard patch (re-apply when this drop is updated) -------------
// A parked cat sits; the drop only ever has it standing to attention. Blended
// in over the rest pose so the gallop is untouched, and exported so the numbers
// can be tuned from the page while they are being dialled in.
export const SIT = { bodyDrop: .10, bodyTilt: -.15, backTilt: 1.0, backShift: .12,
                     backDrop: .35, headTilt: .05, frontPawZ: .06,
                     rearHip: 1.25, rearKnee: -2.0,
                     // The tail hangs off the rear, which the sit rotates down
                     // and under. tailBase lifts it back out of the body and
                     // tailSide lays it alongside, where the body is not in
                     // front of it — head on, a centred tail is never seen.
                     tailBase: -.45, tailCurl: .18, tailSide: 1.15 };
// --------------------------------------------------------------------------

/** Mount one procedural Bolt pet into a dedicated container. Call dispose on unmount. */
export function createBoltPet(container, options={}) {
  if (!(container instanceof HTMLElement)) throw new TypeError('Expected an HTML container.');
  for(const key of ['maxDpr','turnSpeed'])if(options[key]!==undefined&&(!Number.isFinite(options[key])||options[key]<=0))throw new TypeError(`${key} must be positive and finite.`);
  const state={mode:validateState(options.state??'idle'),yaw:resolveDirection(options.direction??'front'),targetYaw:resolveDirection(options.direction??'front'),speed:0,phase:0,time:0,x:0,targetX:0,roam:options.roam??false,paused:options.autoStart===false,disposed:false,visible:true,lost:false};
  const scene=new THREE.Scene();
  const camera=new THREE.OrthographicCamera(-2,2,2,-2,.1,50);
  camera.position.set(0,2.65,8);camera.lookAt(0,1.3,0);
  const renderer=new THREE.WebGLRenderer({alpha:true,antialias:true,powerPreference:'low-power'});
  renderer.setClearColor(0,0);renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,options.maxDpr??1.5));
  renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.04;
  renderer.domElement.style.cssText='display:block;width:100%;height:100%;';
  renderer.domElement.setAttribute('role','img');container.append(renderer.domElement);
  scene.add(new THREE.HemisphereLight('#fff7ea','#778391',1.65));
  const key=new THREE.DirectionalLight('#fff5e3',2.3);key.position.set(-3,5,5);scene.add(key);
  const rim=new THREE.DirectionalLight('#9fe7f1',1.3);rim.position.set(3,3,-3);scene.add(rim);
  const models=buildModels();scene.add(models.cat,models.complete);
  const backRestY=models.back.position.y;     // Bolt patch: the sit's reference
  const pawTarget=new THREE.Vector3();
  const media=window.matchMedia('(prefers-reduced-motion: reduce)');
  let reduced=options.reducedMotion??media.matches;
  let raf=0,last=0,aspect=1,limit=0,nextHeading=0;
  const label=()=>renderer.domElement.setAttribute('aria-label',`Bolt ${state.mode==='working'?'is working':state.mode==='complete'?'has finished':state.mode==='error'?'needs attention':'is idle'}`);
  function resize(){
    if(state.disposed)return;
    const {width,height}=container.getBoundingClientRect();
    const hairs=options.fur==='off'?0:options.fur==='high'?7500:Math.min(width,height)<96?2200:Math.min(width,height)<160?4200:7500;
    models.coat.geometry.setDrawRange(0,hairs*9);models.coat.visible=hairs>0;
    renderer.setSize(Math.max(1,width),Math.max(1,height),false);aspect=Math.max(1,width)/Math.max(1,height);
    const halfHeight=Math.max(1.8,1.8/aspect);camera.left=-halfHeight*aspect;camera.right=halfHeight*aspect;camera.top=halfHeight;camera.bottom=-halfHeight;camera.updateProjectionMatrix();
    limit=Math.max(0,halfHeight*aspect-1.8);state.targetX=THREE.MathUtils.clamp(state.targetX,-limit,limit);state.x=THREE.MathUtils.clamp(state.x,-limit,limit);
    render();
  }
  function pose(dt){
    const moving=state.mode==='working'&&!reduced;
    state.speed=damp(state.speed,moving?1:0,7,dt);
    // Sits only when parked and still; any run unfolds it again.
    state.sit=damp(state.sit??0,(state.mode==='idle'&&!moving)?1:0,6,dt);
    state.phase+=dt*(moving?10.8:2);state.time+=dt;
    state.yaw=turn(state.yaw,state.targetYaw,options.turnSpeed??7,dt);
    if(state.roam&&moving&&limit>0){
      if(state.time>nextHeading){
        state.targetX=state.targetX>=0?-limit:limit;
        state.targetYaw=state.targetX>state.x?Math.PI/2:-Math.PI/2;
        nextHeading=state.time+3.5;
      }
      // Translation follows actual facing, and slows while turning around.
      const facing=Math.sin(state.yaw),difference=state.targetX-state.x;
      if(Math.abs(difference)>.06&&Math.sign(facing)===Math.sign(difference))state.x+=Math.sign(difference)*Math.min(Math.abs(difference),Math.abs(facing)*dt*1.05);
    }else state.x=damp(state.x,state.targetX,2.7,dt);
    const p=state.phase,run=state.speed,cycle=gallopCycle(p);
    const flight=Math.exp(-(((cycle-.38)/.055)**2))+Math.exp(-(((cycle-.95)/.05)**2));
    models.cat.rotation.y=state.yaw;models.cat.position.set(state.x,flight*.09*run,0);
    models.torso.position.y=1.06-.12*run;
    models.torso.rotation.x=Math.sin(p-.8)*.035*run;
    models.back.rotation.x=Math.sin(p+.8)*.17*run;
    models.back.position.z=-.4+Math.sin(p+.8)*.075*run;
    models.head.rotation.x=-Math.sin(p-.8)*.025*run;
    models.head.rotation.y=-Math.sin(state.yaw)*.38;
    models.head.rotation.z=!reduced&&state.mode==='idle'?Math.sin(state.time*.7)*.035:state.mode==='error'?.14:0;
    // --- Bolt dashboard patch: the sit pose, blended over the rest pose. ---
    const sit=state.sit;
    models.back.position.y=backRestY;
    if(sit>.002){
      models.torso.position.y-=SIT.bodyDrop*sit;
      models.torso.rotation.x+=SIT.bodyTilt*sit;
      models.back.rotation.x+=SIT.backTilt*sit;
      models.back.position.z-=SIT.backShift*sit;
      models.back.position.y=backRestY-SIT.backDrop*sit;   // the rear is what sinks
      models.head.rotation.x+=SIT.headTilt*sit;
    }
    models.cat.updateMatrixWorld(true);
    for(const leg of models.legs){
      const paw=pawTrajectory(cycle,leg.front,leg.side);
      pawTarget.set(leg.side*.32,.27+paw.lift-models.cat.position.y,leg.z+paw.z);
      models.cat.localToWorld(pawTarget);leg.hip.parent.worldToLocal(pawTarget);pawTarget.sub(leg.hip.position);
      const angles=solveLeg(pawTarget.y,pawTarget.z,leg.front);
      let hip=angles.hip*run,knee=angles.knee*run;
      if(sit>.002){
        if(leg.front){
          // Solve again against the lowered body so the forelegs hold the chest
          // up rather than sinking with it — that is what reads as sitting.
          pawTarget.set(leg.side*.32,.27-models.cat.position.y,leg.z+SIT.frontPawZ);
          models.cat.localToWorld(pawTarget);leg.hip.parent.worldToLocal(pawTarget);pawTarget.sub(leg.hip.position);
          const planted=solveLeg(pawTarget.y,pawTarget.z,true);
          hip=hip*(1-sit)+planted.hip*sit;knee=knee*(1-sit)+planted.knee*sit;
        }else{
          hip=hip*(1-sit)+SIT.rearHip*sit;knee=knee*(1-sit)+SIT.rearKnee*sit;
        }
      }
      leg.hip.rotation.x=hip;leg.knee.rotation.x=knee;
      const parentTilt=models.torso.rotation.x+(leg.front?0:models.back.rotation.x);
      leg.ankle.rotation.x=-(hip+knee+parentTilt)*.92;
    }
    models.tail.forEach((joint,i)=>{
      joint.rotation.z=reduced?0:Math.sin(state.time*2.1-i*.5)*.025;
      joint.rotation.x=reduced?0:(i===0?-.25*run:0)+Math.sin(p*.6-i*.4)*.035*run;
      if(sit>.002){
        joint.rotation.x+=(i===0?SIT.tailBase:SIT.tailCurl*(i<3?1:.5))*sit;
        // Z, not Y: the chain points up, so this is what leans it out to the
        // side where the body is not in front of it.
        joint.rotation.z+=SIT.tailSide*sit*(i===0?1:.6);
      }
    });
    const blinkPhase=state.time%4.8;
    const blink=!reduced&&blinkPhase>4.55?Math.max(.07,Math.abs(blinkPhase-4.675)/.125):1;
    models.eyes.forEach(eye=>eye.scale.y=blink);
    const done=state.mode==='complete';
    // Brief geometry scale transition, with no timers that can race state changes.
    models.cat.scale.setScalar(damp(models.cat.scale.x,done?.001:1,12,dt));
    models.complete.scale.setScalar(damp(models.complete.scale.x,done?1:.001,12,dt));
    models.cat.visible=models.cat.scale.x>.01;models.complete.visible=models.complete.scale.x>.01;
    models.complete.rotation.y=reduced?0:Math.sin(state.time*.85)*.2;
    models.logo.rotation.z=reduced?0:Math.sin(state.time*1.5)*.045;
    models.shineUniform.value=reduced?3:2.8-((state.time%2.4)/2.4)*5.6;
    models.logoMaterial.emissiveIntensity=reduced?.14:.15+.1*(1+Math.sin(state.time*2.6));
    models.sparkles.forEach((s,i)=>{s.scale.setScalar(reduced?.45:Math.max(.05,Math.pow(Math.max(0,Math.sin(state.time*3-i*1.7)),4)));});
  }
  function render(){if(!state.disposed&&!state.lost)renderer.render(scene,camera);}
  function canAnimate(){return !state.disposed&&!state.paused&&!state.lost&&!document.hidden&&state.visible;}
  function tick(now){
    raf=0;if(!canAnimate())return;
    const dt=last?Math.min((now-last)/1000,.1):1/60;last=now;
    pose(dt);render();
    if(!reduced)raf=requestAnimationFrame(tick);
  }
  function schedule(){if(!raf&&canAnimate()){last=0;raf=requestAnimationFrame(tick);}}
  function sync(){if(!canAnimate()&&raf){cancelAnimationFrame(raf);raf=0;}else schedule();}
  function onMedia(){if(options.reducedMotion!==undefined)return;reduced=media.matches;if(reduced)snapPose();sync();}
  function snapPose(){
    models.cat.scale.setScalar(state.mode==='complete'?.001:1);models.complete.scale.setScalar(state.mode==='complete'?1:.001);
    state.yaw=state.targetYaw;pose(0);render();
  }
  function contextLost(e){e.preventDefault();state.lost=true;sync();options.onContextLost?.();}
  function contextRestored(){state.lost=false;render();schedule();}
  const observer=new ResizeObserver(resize);observer.observe(container);
  const visibilityObserver=new IntersectionObserver(entries=>{state.visible=entries[0]?.isIntersecting??true;sync();});visibilityObserver.observe(container);
  document.addEventListener('visibilitychange',sync);media.addEventListener('change',onMedia);
  renderer.domElement.addEventListener('webglcontextlost',contextLost);renderer.domElement.addEventListener('webglcontextrestored',contextRestored);
  function alive(){if(state.disposed)throw new Error('This Bolt pet has been disposed.');}
  const api={
    setSize(width,height=width){alive();if(!Number.isFinite(width)||width<=0||!Number.isFinite(height)||height<=0)throw new TypeError('Size must be positive finite CSS pixels.');container.style.width=`${width}px`;container.style.height=`${height}px`;resize();},
    setState(value){alive();validateState(value);if(value===state.mode)return;state.mode=value;label();if(reduced)snapPose();schedule();options.onStateChange?.(value);},
    setDirection(value){alive();state.targetYaw=resolveDirection(value);if(reduced)snapPose();schedule();},
    setRoaming(value){alive();state.roam=Boolean(value);nextHeading=0;if(!state.roam)state.targetX=0;schedule();},
    moveTo(x){alive();if(!Number.isFinite(x))throw new TypeError('x must be finite world units.');state.roam=false;state.targetX=THREE.MathUtils.clamp(x,-limit,limit);if(Math.abs(state.targetX-state.x)>.01)state.targetYaw=state.targetX>state.x?Math.PI/2:-Math.PI/2;if(reduced){state.x=state.targetX;snapPose();}schedule();},
    pause(){alive();state.paused=true;sync();},
    resume(){alive();state.paused=false;sync();},
    step(seconds){alive();if(!state.paused)throw new Error('Pause Bolt before calling step().');if(!Number.isFinite(seconds)||seconds<0||seconds>.25)throw new TypeError('Step must be between 0 and 0.25 seconds.');pose(seconds);render();},
    resize,
    getState(){return {state:state.mode,direction:state.targetYaw,yaw:state.yaw,x:state.x,roam:state.roam,paused:state.paused,reducedMotion:reduced};},
    dispose(){
      if(state.disposed)return;state.disposed=true;cancelAnimationFrame(raf);observer.disconnect();visibilityObserver.disconnect();
      document.removeEventListener('visibilitychange',sync);media.removeEventListener('change',onMedia);
      renderer.domElement.removeEventListener('webglcontextlost',contextLost);renderer.domElement.removeEventListener('webglcontextrestored',contextRestored);
      const geometries=new Set(),materials=new Set();scene.traverse(o=>{if(o.geometry)geometries.add(o.geometry);if(o.material)(Array.isArray(o.material)?o.material:[o.material]).forEach(m=>materials.add(m));});
      geometries.forEach(g=>g.dispose());materials.forEach(m=>m.dispose());models.skeleton?.dispose();renderer.dispose();renderer.forceContextLoss();renderer.domElement.remove();
    }
  };
  label();snapPose();resize();schedule();return api;
}
