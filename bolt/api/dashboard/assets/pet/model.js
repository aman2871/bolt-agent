import * as THREE from 'three';
import { sculptCat, createFurMaterial, furPalette } from './sculpt.js';
import { buildFur } from './fur.js';

// Bolt faces local +Z. Every visible part is geometry, not a billboard or image.
export function buildModels() {
  const white = new THREE.MeshStandardMaterial({ color: '#fffaf0', roughness: .8 });
  const pink = new THREE.MeshStandardMaterial({ color: '#e9949e', roughness: .72 });
  const dark = new THREE.MeshStandardMaterial({ color: '#241c22', roughness: .45 });
  const gold = new THREE.MeshStandardMaterial({ color: '#ffc52e', metalness: .38, roughness: .3, emissive: '#ffb600', emissiveIntensity: .12 });
  const iris = new THREE.MeshStandardMaterial({ color: furPalette.iris, roughness: .28 });
  const ball = new THREE.SphereGeometry(1, 24, 16);
  function ellipsoid(parent, material, pos, scale) {
    const mesh = new THREE.Mesh(ball, material);
    mesh.position.set(...pos); mesh.scale.set(...scale); parent.add(mesh); return mesh;
  }
  function boltGeometry(size=1, depth=.09) {
    const shape=new THREE.Shape();
    [[.4,.74],[-.35,.56],[-.72,-.16],[-.1,-.06],[-.37,-.87],[.74,.22],[.12,.14]].forEach(([x,y],i)=>i?shape.lineTo(x*size,y*size):shape.moveTo(x*size,y*size));
    shape.closePath();
    const g=new THREE.ExtrudeGeometry(shape,{depth,bevelEnabled:true,bevelSegments:2,steps:1,bevelSize:.025*size,bevelThickness:.02*size});
    g.translate(0,0,-depth/2); return g;
  }
  const cat=new THREE.Group(); cat.name='BoltCat';
  const fur=createFurMaterial();
  const sculpt=sculptCat(fur);cat.add(sculpt.mesh);
  const coat=buildFur(sculpt.mesh,createFurMaterial(true));cat.add(coat);
  const torso=sculpt.spine,head=sculpt.head,legs=sculpt.legs,tail=sculpt.tail;
  const eyes=[];
  for(const side of [-1,1]){
    const eye=new THREE.Group();eye.position.set(side*.262,.19,.60);eye.rotation.y=side*.16;eye.rotation.z=side*.12;head.add(eye);eyes.push(eye);
    ellipsoid(eye,dark,[0,0,0],[.208,.159,.048]);
    ellipsoid(eye,iris,[0,0,.019],[.194,.144,.042]);
    ellipsoid(eye,dark,[.005,-.005,.057],[.069,.12,.023]);
    ellipsoid(eye,white,[-.047,.052,.082],[.034,.039,.012]);
    ellipsoid(eye,white,[.04,-.05,.08],[.015,.017,.008]);
    for(let i=0;i<3;i++){
      const curve=new THREE.CatmullRomCurve3([new THREE.Vector3(side*.28,-.03,.67),new THREE.Vector3(side*.49,-.02+(i-1)*.055,.69),new THREE.Vector3(side*.71,-.02+(i-1)*.11,.62)]);
      head.add(new THREE.Mesh(new THREE.TubeGeometry(curve,8,.0045,4,false),white));
    }
  }
  ellipsoid(head,pink,[0,-.002,.747],[.061,.038,.03]);
  for(const side of [-1,1]){
    const curve=new THREE.QuadraticBezierCurve3(new THREE.Vector3(0,-.03,.75),new THREE.Vector3(side*.065,-.12,.744),new THREE.Vector3(side*.13,-.073,.728));
    head.add(new THREE.Mesh(new THREE.TubeGeometry(curve,14,.007,5,false),dark));
  }
  const collar=new THREE.Mesh(new THREE.TorusGeometry(.355,.025,8,40),dark);collar.rotation.x=Math.PI/2;collar.position.set(0,.14,.83);torso.add(collar);
  const pendant=new THREE.Mesh(boltGeometry(.15,.025),gold);pendant.position.set(0,-.02,1.24);torso.add(pendant);
  // Completed-state logo is a thick extruded mesh, with a travelling sheen shader.
  const complete=new THREE.Group();complete.name='BoltComplete';
  const shineUniform={value:0};
  const logoMaterial=gold.clone();logoMaterial.roughness=.23;logoMaterial.metalness=.18;
  logoMaterial.onBeforeCompile=shader=>{
    shader.uniforms.uBoltShine=shineUniform;
    shader.vertexShader='varying vec3 vBoltLocal;\n'+shader.vertexShader;
    shader.vertexShader=shader.vertexShader.replace('#include <begin_vertex>','#include <begin_vertex>\nvBoltLocal = position;');
    shader.fragmentShader='uniform float uBoltShine; varying vec3 vBoltLocal;\n'+shader.fragmentShader;
    shader.fragmentShader=shader.fragmentShader.replace('#include <color_fragment>','#include <color_fragment>\nfloat sweep = 1.0-smoothstep(0.025,0.22,abs(vBoltLocal.x + vBoltLocal.y - uBoltShine));\ndiffuseColor.rgb = mix(diffuseColor.rgb, vec3(1.0, 0.98, 0.8), sweep * 0.9);');
  };
  const logoSide=new THREE.MeshStandardMaterial({color:'#ee920c',metalness:.12,roughness:.38});
  const logo=new THREE.Mesh(boltGeometry(1.6,.25),[logoMaterial,logoSide]);logo.position.y=1.35;complete.add(logo);
  for(const side of [-1,1]) {
    const x=-.17+side*.19;
    ellipsoid(logo,dark,[x,.26,.174],[.075,.106,.025]);
    ellipsoid(logo,white,[x-.02,.301,.2],[.021,.027,.009]);
    const brow=new THREE.QuadraticBezierCurve3(new THREE.Vector3(x-.07,.409,.178),new THREE.Vector3(x,.454,.178),new THREE.Vector3(x+.07,.409,.178));
    logo.add(new THREE.Mesh(new THREE.TubeGeometry(brow,12,.014,5,false),dark));
  }
  const grin=new THREE.QuadraticBezierCurve3(new THREE.Vector3(-.34,.06,.18),new THREE.Vector3(-.16,-.15,.18),new THREE.Vector3(.025,.08,.18));
  logo.add(new THREE.Mesh(new THREE.TubeGeometry(grin,28,.024,8,false),dark));
  const teeth=new THREE.QuadraticBezierCurve3(new THREE.Vector3(-.285,.036,.204),new THREE.Vector3(-.16,-.055,.204),new THREE.Vector3(-.035,.051,.204));
  logo.add(new THREE.Mesh(new THREE.TubeGeometry(teeth,22,.008,6,false),white));
  const sparkles=[];
  for(let i=0;i<3;i++){
    const s=new THREE.Group();s.position.set([.65,-.63,.4][i],[2.35,.63,1.15][i],.23);complete.add(s);
    const mat=new THREE.MeshBasicMaterial({color:i===2?'#a3f3ff':'#fff3b3'});
    ellipsoid(s,mat,[0,0,0],[.025,.14,.025]);ellipsoid(s,mat,[0,0,0],[.11,.025,.025]);sparkles.push(s);
  }
  return {cat,coat,torso,back:sculpt.back,head,eyes,legs,tail,skeleton:sculpt.skeleton,complete,logo,logoMaterial,shineUniform,sparkles};
}
