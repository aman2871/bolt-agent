const TAU=Math.PI*2;
const clamp=(x,a,b)=>Math.min(b,Math.max(a,x));
export function pawTrajectory(cycle,front,side){
  // Hind-hind-fore-fore contacts, not a diagonal trot. Each forepaw has its
  // own contact interval, with suspension between the two limb pairs.
  const touchdown=front?(side<0?.43:.68):(side<0?0:.11);
  const phase=((cycle-touchdown)%1+1)%1,duty=.22;
  const reach=front?.4:.35,push=front?.37:.43;
  if(phase<duty)return {z:reach-(reach+push)*phase/duty,lift:0,contact:true};
  const u=(phase-duty)/(1-duty),s=u*u*(3-2*u);
  const taper=u>.9?1-((u-.9)/.1)**2:1;
  const stretch=front?Math.exp(-(((u-.82)/.14)**2))*Math.max(0,taper):0;
  return {z:-push+(reach+push)*s+stretch*.3,lift:Math.max(Math.sin(Math.PI*u)*(front?.24:.31),stretch*.32),contact:false};
}
export function solveLeg(dy,dz,front){
  const upper=.42,lower=Math.hypot(.36,.07),bias=Math.atan2(-.07,.36);
  const distance=clamp(Math.hypot(dy,dz),.12,upper+lower-.002);
  const bend=(front?-1:1)*Math.acos(clamp((distance*distance-upper*upper-lower*lower)/(2*upper*lower),-1,1));
  const direction=Math.atan2(-dz,-dy);
  return {hip:direction-Math.atan2(lower*Math.sin(bend),upper+lower*Math.cos(bend)),knee:bend-bias};
}
export function gallopCycle(phase){return ((phase/TAU)%1+1)%1;}
