import * as THREE from 'three';

const smooth=(a,b,k)=>{const h=Math.max(k-Math.abs(a-b),0)/k;return Math.min(a,b)-h*h*k*.25;};
const clamp=(x,a=0,b=1)=>Math.min(b,Math.max(a,x));
const ease=(a,b,x)=>{const t=clamp((x-a)/(b-a));return t*t*(3-2*t);};
const tailPoints=[[0,1.16,-.94],[0,1.35,-1.23],[.04,1.63,-1.42],[.1,1.9,-1.44],[.1,2.08,-1.27]];
const tailSamples=new THREE.CatmullRomCurve3(tailPoints.map(p=>new THREE.Vector3(...p))).getPoints(20).map(p=>[p.x,p.y,p.z]);

// --- Bolt dashboard patch (re-apply when this drop is updated) -------------
// The coat is baked into the shader at compile time, so recolouring means a
// rebuild of the pet. Keeping the palette here is what makes that possible.
export const furPalette = { cream:'#f9e7cd', orange:'#d98324', ivory:'#fff7e8',
                            pink:'#e7a0a2', gold:'#f7b927', iris:'#c68519' };
export function setFurPalette(next){ Object.assign(furPalette, next); }
const paletteKey = () => Object.values(furPalette).join('');
// --------------------------------------------------------------------------

export function createFurMaterial(hair=false){
  const material=hair?new THREE.MeshBasicMaterial():new THREE.MeshStandardMaterial({roughness:.92});
  const linear=hex=>{const c=new THREE.Color(hex);return `vec3(${c.r},${c.g},${c.b})`;};
  material.onBeforeCompile=shader=>{
    shader.vertexShader=(hair?'attribute vec3 furRoot; attribute float furVariation; varying float vFurVariation;\n':'')+'varying vec3 vFurPosition;\n'+shader.vertexShader;
    shader.vertexShader=shader.vertexShader.replace('#include <begin_vertex>','#include <begin_vertex>\n'+(hair?'vFurPosition = furRoot; vFurVariation = furVariation;':'vFurPosition = position;'));
    shader.fragmentShader=(hair?'varying float vFurVariation;\n':'')+`varying vec3 vFurPosition;
float furPatch(vec3 p,vec3 c,vec3 r){vec3 d=(p-c)/r;float wav=.10*sin(p.x*9.0+p.z*12.0)*sin(p.y*9.0+p.z*3.0);return 1.0-smoothstep(.92,1.08,dot(d,d)+wav);}
vec3 boltFur(vec3 p){
 vec3 cream=${linear(furPalette.cream)}, orange=${linear(furPalette.orange)}, ivory=${linear(furPalette.ivory)}, pink=${linear(furPalette.pink)}, gold=${linear(furPalette.gold)};
 float pigment=max(furPatch(p,vec3(0.0,1.36,-.46),vec3(.62,.44,.60)),furPatch(p,vec3(-.43,1.11,-.13),vec3(.21,.4,.33)));
 float face=smoothstep(1.52,1.63,p.y)*smoothstep(.13,.23,abs(p.x))*smoothstep(.83,1.02,p.z);
 pigment=max(pigment,face);pigment=max(pigment,smoothstep(1.90,1.99,p.y)*smoothstep(.23,.34,abs(p.x)));pigment=max(pigment,(1.0-smoothstep(-1.12,-1.02,p.z))*.97);
 vec3 c=mix(cream,orange,pigment);c=mix(c,ivory,max(1.0-smoothstep(.3,.43,p.y),smoothstep(1.13,1.28,p.z)*(1.0-smoothstep(1.47,1.6,p.y))));c=mix(c,ivory,smoothstep(1.85,2.08,p.y)*(1.0-smoothstep(-1.1,-1.0,p.z)));
 float taper=1.0-clamp((p.y-1.87)/.52,0.0,1.0);float ex=abs(p.x)-(.425+.12*clamp((p.y-1.87)/.52,0.0,1.0));
 float inner=(1.0-smoothstep(.09*taper,.17*taper+.001,abs(ex)))*smoothstep(1.97,2.06,p.y)*smoothstep(.70,.74,p.z);c=mix(c,pink,inner*.85);
 vec2 q=vec2(p.x,p.y-1.87);vec2 pts[6];pts[0]=vec2(.045,.21);pts[1]=vec2(-.12,-.03);pts[2]=vec2(-.015,-.03);pts[3]=vec2(-.055,-.22);pts[4]=vec2(.14,.055);pts[5]=vec2(.025,.055);
 bool inside=false;float edge=10.0;for(int i=0;i<6;i++){vec2 a=pts[i],b=pts[(i+1)%6];vec2 d=b-a;float t=clamp(dot(q-a,d)/dot(d,d),0.0,1.0);edge=min(edge,length(q-a-t*d));if((a.y>q.y)!=(b.y>q.y)){if(q.x<(b.x-a.x)*(q.y-a.y)/(b.y-a.y)+a.x)inside=!inside;}}
 float mark=smoothstep(-.004,.004,inside?edge:-edge)*smoothstep(1.0,1.05,p.z);return mix(c,gold,mark);
}
`+shader.fragmentShader;
    shader.fragmentShader=shader.fragmentShader.replace('#include <color_fragment>','#include <color_fragment>\ndiffuseColor.rgb = boltFur(vFurPosition);'+(hair?'diffuseColor.rgb *= vFurVariation;':''));
  };
  material.customProgramCacheKey=()=>(hair?'bolt-hair-v1-':'bolt-skin-v1-')+paletteKey();
  return material;
}

// Sculpt one connected rest-pose surface, then deform it with a skeleton.
// Orange markings are painted by the fur shader, never separate meshes.
export function sculptCat(material) {
  const ellipsoids=[
    [0,1.06,-.28,.48,.47,.75], [0,1.17,.24,.43,.48,.43],
    [0,1.3,.5,.42,.35,.39], [0,1.59,.71,.7,.59,.57],
    [-.43,1.42,.88,.32,.28,.29],[.43,1.42,.88,.32,.28,.29],
    [-.17,1.37,1.19,.23,.17,.17],[.17,1.37,1.19,.23,.17,.17],
  ];
  for(const front of [true,false])for(const side of [-1,1]){
    const z=front?.43:-.73;
    ellipsoids.push([side*.32,.88,z,.19,.33,.23],[side*.32,.49,z+.015,.14,.3,.15],[side*.32,.2,z+.13,.19,.14,.26]);
  }
  function field(x,y,z){
    let d=10;
    for(const [cx,cy,cz,rx,ry,rz] of ellipsoids){
      const dx=(x-cx)/rx,dy=(y-cy)/ry,dz=(z-cz)/rz;
      const k0=Math.sqrt(dx*dx+dy*dy+dz*dz);
      const k1=Math.sqrt(dx*dx/(rx*rx)+dy*dy/(ry*ry)+dz*dz/(rz*rz));
      const q=k1<1e-8?-Math.min(rx,ry,rz):k0*(k0-1)/k1;
      d=smooth(d,q,.14);
    }
    for(const side of [-1,1]){
      const ey=y-1.87,ex=x-side*(.425+.12*clamp(ey/.52));
      const taper=1-clamp(ey/.52);
      const constraints=[(Math.abs(ex)-.23*taper)*.85,(Math.abs(z-.66)-.19*taper)*.85,-ey,ey-.52];
      let ear=constraints[0];for(let i=1;i<constraints.length;i++)ear=-smooth(-ear,-constraints[i],.075);ear-=.018;
      d=smooth(d,ear,.16);
    }
    for(let i=0;i<tailSamples.length-1;i++){
      const a=tailSamples[i],b=tailSamples[i+1];const vx=b[0]-a[0],vy=b[1]-a[1],vz=b[2]-a[2];
      const t=clamp(((x-a[0])*vx+(y-a[1])*vy+(z-a[2])*vz)/(vx*vx+vy*vy+vz*vz));
      const dx=x-a[0]-t*vx,dy=y-a[1]-t*vy,dz=z-a[2]-t*vz;
      const radius=.235-.045*(i+t)/(tailSamples.length-1);
      d=smooth(d,Math.sqrt(dx*dx+dy*dy+dz*dz)-radius,.035);
    }
    return d;
  }
  const root=new THREE.Bone(),spine=new THREE.Bone(),head=new THREE.Bone();
  root.name='root';spine.name='spine';head.name='head';root.add(spine);spine.position.set(0,1.06,-.28);spine.add(head);head.position.set(0,.36,.91);
  const back=new THREE.Bone();back.name='lumbar';back.position.set(0,0,-.4);spine.add(back);
  const bones=[root,spine,head,back],legs=[];
  for(const front of [true,false])for(const side of [-1,1]){
    const hip=new THREE.Bone(),knee=new THREE.Bone(),ankle=new THREE.Bone();
    hip.name=`${front?'front':'rear'}-${side}-hip`;knee.name=hip.name+'-knee';ankle.name=hip.name+'-paw';
    hip.position.set(side*.32,0,front?.71:-.05);knee.position.set(0,-.42,0);ankle.position.set(0,-.36,.07);
    (front?spine:back).add(hip);hip.add(knee);knee.add(ankle);
    const indices=[bones.length,bones.length+1,bones.length+2];bones.push(hip,knee,ankle);legs.push({hip,knee,ankle,front,side,indices,z:front?.43:-.73});
  }
  const tail=[];let parent=back,previous=[0,1.06,-.68];
  for(const p of tailPoints){const b=new THREE.Bone();b.position.set(p[0]-previous[0],p[1]-previous[1],p[2]-previous[2]);parent.add(b);tail.push(b);bones.push(b);parent=b;previous=p;}
  const tailStart=bones.length-tail.length;
  const cream=new THREE.Color(furPalette.cream),orange=new THREE.Color(furPalette.orange),ivory=new THREE.Color(furPalette.ivory),pink=new THREE.Color(furPalette.pink),gold=new THREE.Color(furPalette.gold);
  const color=new THREE.Color();
  function colour(x,y,z){
    const noise=.025*Math.sin(x*24+z*12)*Math.sin(y*19-z*9);
    const pigment=(cx,cy,cz,rx,ry,rz)=>1-ease(.92,1.06,((x-cx)/rx)**2+((y-cy)/ry)**2+((z-cz)/rz)**2+noise);
    let mark=Math.max(pigment(.28,1.38,-.55,.52,.31,.46),pigment(-.43,1.11,-.13,.21,.4,.33));
    const headmark=ease(1.8,1.91,y)*ease(.13,.23,Math.abs(x)+noise)*ease(.65,.84,z);
    mark=Math.max(mark,headmark,y>2.22?ease(.23,.34,Math.abs(x)):0,z<-1.05?.97:0);
    color.copy(cream).lerp(orange,mark);
    color.lerp(ivory,Math.max(1-ease(.3,.43,y),ease(.95,1.1,z)*(1-ease(1.75,1.88,y))));
    if(y>2.27&&z>.52){const xx=Math.abs(x)-(.405+.17*clamp((y-2.16)/.59));const taper=1-clamp((y-2.16)/.59);const mask=(1-ease(.09*taper,.17*taper+.001,Math.abs(xx)))*ease(2.27,2.34,y)*ease(.52,.56,z);color.lerp(pink,mask*.85);}
    // Flat painted forehead lightning, following the curved head surface.
    const bx=x,by=y-2.15;const poly=[[.045,.21],[-.12,-.03],[-.015,-.03],[-.055,-.22],[.14,.055],[.025,.055]];
    let inside=false;for(let i=0,j=poly.length-1;i<poly.length;j=i++){const [xi,yi]=poly[i],[xj,yj]=poly[j];if((yi>by)!==(yj>by)&&bx<(xj-xi)*(by-yi)/(yj-yi)+xi)inside=!inside;}
    if(inside&&z>.86)color.copy(gold);
    return [color.r,color.g,color.b];
  }
  function weights(x,y,z){
    if(z<-1.03&&y>1.14){
      let best=1e9,seg=0,tBest=0;
      for(let i=0;i<tailPoints.length-1;i++){
        const a=tailPoints[i],b=tailPoints[i+1],v=b.map((v,k)=>v-a[k]);const t=clamp(((x-a[0])*v[0]+(y-a[1])*v[1]+(z-a[2])*v[2])/(v[0]**2+v[1]**2+v[2]**2));
        const dist=(x-a[0]-t*v[0])**2+(y-a[1]-t*v[1])**2+(z-a[2]-t*v[2])**2;if(dist<best){best=dist;seg=i;tBest=t;}
      }
      if(seg===0){const blend=ease(.03,.5,tBest);return [[3,tailStart,tailStart+1,0],[1-blend,blend*(1-tBest),blend*tBest,0]];}
      return [[tailStart+seg,tailStart+seg+1,3,0],[1-tBest,tBest,0,0]];
    }
    const h=ease(1.10,1.49,y)*ease(.25,.5,z);
    if(h>.001)return [[1,2,0,0],[1-h,h,0,0]];
    let leg=legs[0],dist=1e9;
    for(const l of legs){const d=((x-l.side*.32)/.23)**2+((z-l.z)/.35)**2;if(d<dist){dist=d;leg=l;}}
    const amount=(1-ease(.6,.94,y))*(1-ease(.9,2.0,dist));
    const upper=ease(.47,.83,y),paw=1-ease(.22,.43,y);
    if(amount<.001){const rear=1-ease(-.7,-.12,z);return [[1,3,0,0],[1-rear,rear,0,0]];}
    return [[leg.front?1:3,...leg.indices],[1-amount,amount*upper,amount*(1-upper)*(1-paw),amount*(1-upper)*paw]];
  }
  const min=[-.91,.015,-1.75],max=[.96,2.65,1.5],nx=40,ny=64,nz=68;
  const sx=(max[0]-min[0])/nx,sy=(max[1]-min[1])/ny,sz=(max[2]-min[2])/nz;
  const index=(x,y,z)=>x+(nx+1)*(y+(ny+1)*z);
  const values=new Float32Array((nx+1)*(ny+1)*(nz+1));
  for(let z=0;z<=nz;z++)for(let y=0;y<=ny;y++)for(let x=0;x<=nx;x++)values[index(x,y,z)]=field(min[0]+x*sx,min[1]+y*sy,min[2]+z*sz);
  const pos=[],norm=[],colors=[],skinIndices=[],skinWeights=[],triangles=[],cache=new Map();
  const corners=[[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],tets=[[0,5,1,6],[0,1,2,6],[0,2,3,6],[0,3,7,6],[0,7,4,6],[0,4,5,6]];
  const ids=new Array(8),coords=new Array(8),vals=new Array(8);
  function vertex(a,b){
    const ka=ids[a],kb=ids[b],key=ka<kb?ka*values.length+kb:kb*values.length+ka;
    if(cache.has(key))return cache.get(key);
    const t=vals[a]/(vals[a]-vals[b]);const p=coords[a].map((v,k)=>v+(coords[b][k]-v)*t);const [x,y,z]=p,e=.0025;
    const n=new THREE.Vector3(field(x+e,y,z)-field(x-e,y,z),field(x,y+e,z)-field(x,y-e,z),field(x,y,z+e)-field(x,y,z-e)).normalize();
    const at=pos.length/3;pos.push(...p);norm.push(n.x,n.y,n.z);colors.push(...colour(x,y,z));const [bi,bw]=weights(x,y,z);skinIndices.push(...bi);skinWeights.push(...bw);cache.set(key,at);return at;
  }
  function triangle(a,b,c){
    const i=a*3,j=b*3,k=c*3;
    const ux=pos[j]-pos[i],uy=pos[j+1]-pos[i+1],uz=pos[j+2]-pos[i+2],vx=pos[k]-pos[i],vy=pos[k+1]-pos[i+1],vz=pos[k+2]-pos[i+2];
    const dot=(uy*vz-uz*vy)*norm[i]+(uz*vx-ux*vz)*norm[i+1]+(ux*vy-uy*vx)*norm[i+2];
    if(dot<0)triangles.push(a,c,b);else triangles.push(a,b,c);
  }
  for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
    let count=0;
    for(let c=0;c<8;c++){const [dx,dy,dz]=corners[c];ids[c]=index(x+dx,y+dy,z+dz);vals[c]=values[ids[c]];if(vals[c]<0)count++;coords[c]=[min[0]+(x+dx)*sx,min[1]+(y+dy)*sy,min[2]+(z+dz)*sz];}
    if(count===0||count===8)continue;
    for(const tet of tets){const inside=tet.filter(c=>vals[c]<0),outside=tet.filter(c=>vals[c]>=0);
      if(inside.length===1)triangle(...outside.map(b=>vertex(inside[0],b)));
      else if(inside.length===3)triangle(...inside.map(b=>vertex(outside[0],b)));
      else if(inside.length===2){const a=vertex(inside[0],outside[0]),b=vertex(inside[0],outside[1]),c=vertex(inside[1],outside[0]),d=vertex(inside[1],outside[1]);triangle(a,b,c);triangle(b,d,c);}
    }
  }
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));geometry.setAttribute('normal',new THREE.Float32BufferAttribute(norm,3));geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));geometry.setAttribute('skinIndex',new THREE.Uint16BufferAttribute(skinIndices,4));geometry.setAttribute('skinWeight',new THREE.Float32BufferAttribute(skinWeights,4));geometry.setIndex(triangles);
  const mesh=new THREE.SkinnedMesh(geometry,material);mesh.frustumCulled=false;const skeleton=new THREE.Skeleton(bones);mesh.add(root);mesh.updateMatrixWorld(true);mesh.bind(skeleton);
  return {mesh,spine,back,head,legs,tail,skeleton};
}
