// The grid, as glass. Alpha is silica thickness and lifts the surface;
// pigment is the colour. Nothing here is procedural: whatever the cells
// are, that is what stands on the plate.

import * as THREE from "../node_modules/three/build/three.module.js";

export { mount };

const VERT = /* glsl */ `
  uniform sampler2D body;
  uniform float lift;
  varying vec2 vUv;
  varying vec3 vPos;
  varying float vAlpha;
  varying vec3 vColor;
  void main() {
    vUv = uv;
    vec4 cell = texture2D(body, uv);
    float a = clamp(cell.a, 0.0, 1.0);
    vAlpha = a;
    vColor = clamp(cell.rgb, 0.0, 1.0);
    vec3 p = position;
    p.z += a * lift;
    vPos = p;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }
`;

const FRAG = /* glsl */ `
  uniform sampler2D body;
  uniform float lift;
  uniform float texel;
  uniform vec3 lightDir;
  uniform vec3 rimColor;
  uniform float time;
  varying vec2 vUv;
  varying vec3 vPos;
  varying float vAlpha;
  varying vec3 vColor;
  float height(vec2 uv) { return clamp(texture2D(body, uv).a, 0.0, 1.0) * lift; }
  void main() {
    if (vAlpha < 0.1) discard;
    float hx = height(vUv + vec2(texel, 0.0)) - height(vUv - vec2(texel, 0.0));
    float hy = height(vUv + vec2(0.0, texel)) - height(vUv - vec2(0.0, texel));
    vec3 n = normalize(vec3(-hx * 8.0, -hy * 8.0, 1.0));
    float diff = max(dot(n, normalize(lightDir)), 0.0);
    vec3 viewDir = normalize(cameraPosition - vPos);
    float spec = pow(max(dot(reflect(-normalize(lightDir), n), viewDir), 0.0), 26.0);
    float fres = pow(1.0 - max(dot(n, viewDir), 0.0), 3.0);
    float edge = smoothstep(0.1, 0.35, vAlpha);
    vec3 col = vColor * (0.32 + 0.78 * diff) + spec * vec3(0.9, 0.85, 0.7) + fres * rimColor * 0.7;
    col = mix(rimColor * 0.8, col, edge);
    gl_FragColor = vec4(col, 1.0);
  }
`;

function mount(canvas, body) {
  const size = body.size;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.05, 40);
  camera.position.set(0, 1.35, 2.6);
  camera.lookAt(0, 0, 0);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setClearColor(0x070b10, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const texture = new THREE.DataTexture(
    new Float32Array(size * size * 4), size, size, THREE.RGBAFormat, THREE.FloatType
  );
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearFilter;
  texture.needsUpdate = true;

  const material = new THREE.ShaderMaterial({
    vertexShader: VERT,
    fragmentShader: FRAG,
    side: THREE.DoubleSide,
    uniforms: {
      body: { value: texture },
      lift: { value: 0.22 },
      texel: { value: 1 / size },
      lightDir: { value: new THREE.Vector3(0.6, 0.5, 1.0) },
      rimColor: { value: new THREE.Color(0x58c4dc) },
      time: { value: 0 },
    },
  });
  const geo = new THREE.PlaneGeometry(2.2, 2.2, size * 2, size * 2);
  const plate = new THREE.Mesh(geo, material);
  plate.rotation.x = -Math.PI / 2;
  const group = new THREE.Group();
  group.add(plate);
  scene.add(group);

  const dish = new THREE.Mesh(
    new THREE.CircleGeometry(1.5, 96),
    new THREE.MeshBasicMaterial({ color: 0x0b1219, transparent: true, opacity: 0.9, depthWrite: false })
  );
  dish.rotation.x = -Math.PI / 2;
  dish.position.y = -0.02;
  scene.add(dish);
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(1.49, 1.53, 96),
    new THREE.MeshBasicMaterial({ color: 0x243244, side: THREE.DoubleSide })
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = -0.015;
  scene.add(ring);

  const dustCount = 120;
  const dustPos = new Float32Array(dustCount * 3);
  for (let i = 0; i < dustCount; i++) {
    dustPos[i * 3] = (Math.random() - 0.5) * 6;
    dustPos[i * 3 + 1] = Math.random() * 3 - 0.5;
    dustPos[i * 3 + 2] = (Math.random() - 0.5) * 6;
  }
  const dustGeo = new THREE.BufferGeometry();
  dustGeo.setAttribute("position", new THREE.BufferAttribute(dustPos, 3));
  scene.add(
    new THREE.Points(dustGeo, new THREE.PointsMaterial({ color: 0x9fb4c6, size: 0.014, transparent: true, opacity: 0.5 }))
  );

  let yaw = 0;
  let pitch = 0;
  let drag = null;
  let spin = 0.0025;
  canvas.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX, y: e.clientY, yaw, pitch };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    yaw = drag.yaw + (e.clientX - drag.x) * 0.005;
    pitch = Math.max(-0.5, Math.min(0.5, drag.pitch + (e.clientY - drag.y) * 0.004));
  });
  canvas.addEventListener("pointerup", () => {
    drag = null;
  });

  function upload() {
    const n = size * size;
    const st = body.state;
    const data = texture.image.data;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        // Texture rows run bottom-up; the grid runs top-down.
        const src = (size - 1 - y) * size + x;
        const dst = (y * size + x) * 4;
        data[dst] = st[src];
        data[dst + 1] = st[n + src];
        data[dst + 2] = st[2 * n + src];
        data[dst + 3] = st[3 * n + src];
      }
    }
    texture.needsUpdate = true;
  }

  function resize() {
    const w = canvas.clientWidth || canvas.parentElement.clientWidth || 800;
    const h = canvas.clientHeight || canvas.parentElement.clientHeight || 600;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(1, h);
    camera.updateProjectionMatrix();
  }

  let t = 0;
  function frame() {
    t += 1;
    upload();
    if (!drag) yaw += spin;
    group.rotation.y = yaw;
    group.rotation.x = pitch;
    dish.rotation.z = yaw;
    material.uniforms.time.value = t;
    const pos = dustGeo.attributes.position;
    for (let i = 0; i < dustCount; i++) {
      pos.array[i * 3 + 1] += 0.002;
      if (pos.array[i * 3 + 1] > 2.5) pos.array[i * 3 + 1] = -0.5;
    }
    pos.needsUpdate = true;
    renderer.render(scene, camera);
  }

  function loop() {
    resize();
    frame();
    requestAnimationFrame(loop);
  }
  resize();
  requestAnimationFrame(loop);

  return {
    resize,
    setSpin(v) {
      spin = v;
    },
    setLift(v) {
      material.uniforms.lift.value = v;
    },
    setRim(hex) {
      material.uniforms.rimColor.value.set(hex);
    },
  };
}
