import * as THREE from 'three';

/** Short tapered hair geometry. It shares the cat's actual skeleton and coat shader. */
export function buildFur(skin,material,count=7500){
  const source=skin.geometry,p=source.attributes.position,n=source.attributes.normal,si=source.attributes.skinIndex,sw=source.attributes.skinWeight,idx=source.index.array;
  const cumulative=[],a=new THREE.Vector3(),b=new THREE.Vector3(),c=new THREE.Vector3(),cross=new THREE.Vector3();let area=0;
  for(let i=0;i<idx.length;i+=3){a.fromBufferAttribute(p,idx[i]);b.fromBufferAttribute(p,idx[i+1]);c.fromBufferAttribute(p,idx[i+2]);area+=cross.subVectors(b,a).cross(c.clone().sub(a)).length()*.5;cumulative.push(area);}
  let seed=8317;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)|0;return (seed>>>0)/4294967296;};
  const positions=[],normals=[],roots=[],boneIndices=[],boneWeights=[],variations=[];
  const root=new THREE.Vector3(),normal=new THREE.Vector3(),tangent=new THREE.Vector3(),bitangent=new THREE.Vector3(),tip=new THREE.Vector3();
  for(let hair=0;hair<count;){
    const target=random()*area;let lo=0,hi=cumulative.length-1;while(lo<hi){const mid=(lo+hi)>>1;if(cumulative[mid]<target)lo=mid+1;else hi=mid;}
    const ids=[idx[lo*3],idx[lo*3+1],idx[lo*3+2]],u=Math.sqrt(random()),v=random(),bary=[1-u,u*(1-v),u*v];
    root.set(0,0,0);normal.set(0,0,0);
    for(let k=0;k<3;k++){a.fromBufferAttribute(p,ids[k]);root.addScaledVector(a,bary[k]);a.fromBufferAttribute(n,ids[k]);normal.addScaledVector(a,bary[k]);}
    normal.normalize();const {x,y,z}=root;
    // The eye sockets, nose and inner ears stay clear; cheek edges keep long hair.
    if((z>1.02&&y>1.18&&y<1.98&&Math.abs(x)<.5)||(y>2.0&&z>.78)||y<.12)continue;
    const tail=z<-1.04,cheek=y>1.15&&y<1.9&&z>.45&&Math.abs(x)>.4;
    let length=(tail?.11:cheek?.085:y<.55?.035:y>1.9?.035:.055)*(.72+random()*.48);
    if(z>1.12)length*=.3;
    const radius=.0015+random()*.0018;
    tangent.crossVectors(normal,Math.abs(normal.y)>.9?new THREE.Vector3(1,0,0):new THREE.Vector3(0,1,0)).normalize();bitangent.crossVectors(normal,tangent).normalize();
    const comb=new THREE.Vector3(0,tail?1:-1,tail?.1:-.35);comb.addScaledVector(normal,-comb.dot(normal));
    if(comb.lengthSq()>.001)comb.normalize();
    tip.copy(root).addScaledVector(normal,length*.45).addScaledVector(comb,length*.85);
    const weightMap=new Map();
    for(let k=0;k<3;k++)for(let q=0;q<4;q++){const bi=si.array[ids[k]*4+q],bw=sw.array[ids[k]*4+q]*bary[k];weightMap.set(bi,(weightMap.get(bi)??0)+bw);}
    const weights=[...weightMap].sort((a,b)=>b[1]-a[1]).slice(0,4);while(weights.length<4)weights.push([0,0]);const sum=weights.reduce((s,w)=>s+w[1],0);
    const shade=.91+random()*.16;
    for(let side=0;side<3;side++){
      const corners=[side,(side+1)%3];
      for(let vtx=0;vtx<3;vtx++){
        if(vtx===2){a.copy(tip);b.copy(normal);}else{const theta=corners[vtx]/3*Math.PI*2;b.copy(tangent).multiplyScalar(Math.cos(theta)).addScaledVector(bitangent,Math.sin(theta));a.copy(root).addScaledVector(b,radius).addScaledVector(normal,-.003);}
        positions.push(a.x,a.y,a.z);normals.push(b.x,b.y,b.z);roots.push(root.x,root.y,root.z);variations.push(shade);
        for(const [bi,bw] of weights){boneIndices.push(bi);boneWeights.push(bw/sum);}
      }
    }
    hair++;
  }
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setAttribute('normal',new THREE.Float32BufferAttribute(normals,3));geometry.setAttribute('furRoot',new THREE.Float32BufferAttribute(roots,3));geometry.setAttribute('furVariation',new THREE.Float32BufferAttribute(variations,1));geometry.setAttribute('skinIndex',new THREE.Uint16BufferAttribute(boneIndices,4));geometry.setAttribute('skinWeight',new THREE.Float32BufferAttribute(boneWeights,4));
  const fur=new THREE.SkinnedMesh(geometry,material);fur.bind(skin.skeleton,skin.bindMatrix);fur.frustumCulled=false;fur.name='BoltFur';return fur;
}
