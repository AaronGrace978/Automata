// Procedural centric frustule. Two valves, a girdle, ribs, and areolae.
// The pose comes from DiatomMorph; this file only wears it as glass.

import * as THREE from "../node_modules/three/build/three.module.js";

export { mount };

const CONTINUOUS = [
    "girdle", "dome", "pores", "ribs", "asymmetry", "spin", "damage", "hue", "bloom", "fire", "energy",
  ];

  function hsl(h, s, l) {
    const a = s * Math.min(l, 1 - l);
    const f = (n) => {
      const k = (n + h * 12) % 12;
      return l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
    };
    return [f(0), f(8), f(4)];
  }

  function mix(a, b, t) {
    return [
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t,
      a[2] + (b[2] - a[2]) * t,
    ];
  }

  function mount(canvas) {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(32, 1, 0.05, 40);
    camera.position.set(2.15, 1.15, 2.65);
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    renderer.setClearColor(0x070b10, 1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    scene.add(new THREE.HemisphereLight(0xd5e4f2, 0x1a1208, 0.7));
    const key = new THREE.DirectionalLight(0xffe2b0, 1.45);
    key.position.set(2.6, 3.4, 1.6);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x7fd0ff, 0.9);
    rim.position.set(-2.6, -0.8, -1.8);
    scene.add(rim);

    const group = new THREE.Group();
    scene.add(group);
    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      metalness: 0.72,
      roughness: 0.28,
      side: THREE.DoubleSide,
    });
    let mesh = null;

    const shadow = new THREE.Mesh(
      new THREE.CircleGeometry(1.15, 64),
      new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.28, depthWrite: false })
    );
    shadow.rotation.x = -Math.PI / 2;
    shadow.position.y = -0.62;
    scene.add(shadow);

    const dustCount = 90;
    const dustPos = new Float32Array(dustCount * 3);
    for (let i = 0; i < dustCount; i++) {
      dustPos[i * 3] = (Math.random() - 0.5) * 7;
      dustPos[i * 3 + 1] = (Math.random() - 0.5) * 4.2;
      dustPos[i * 3 + 2] = (Math.random() - 0.5) * 5;
    }
    const dustGeo = new THREE.BufferGeometry();
    dustGeo.setAttribute("position", new THREE.BufferAttribute(dustPos, 3));
    const dust = new THREE.Points(
      dustGeo,
      new THREE.PointsMaterial({ color: 0x9fb4c6, size: 0.015, transparent: true, opacity: 0.55 })
    );
    scene.add(dust);

    let target = null;
    let shown = null;
    let yaw = 0.55;
    let pitch = 0.32;
    let drag = null;
    let frameCount = 0;

    canvas.addEventListener("pointerdown", (event) => {
      drag = { x: event.clientX, y: event.clientY, yaw, pitch };
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!drag) return;
      yaw = drag.yaw + (event.clientX - drag.x) * 0.005;
      pitch = Math.max(-0.6, Math.min(1.1, drag.pitch + (event.clientY - drag.y) * 0.005));
    });
    canvas.addEventListener("pointerup", () => {
      drag = null;
    });

    function approach(pose) {
      if (!shown) {
        shown = Object.assign({}, pose);
        return;
      }
      if (shown.folds !== pose.folds) {
        shown.ribs *= 0.9;
        if (shown.ribs < 0.12) shown.folds = pose.folds;
      }
      for (const key of CONTINUOUS) {
        if (key === "ribs" && shown.folds !== pose.folds) continue;
        shown[key] += (pose[key] - shown[key]) * 0.08;
      }
    }

    function surfacePoint(rv, theta, sign) {
      const folds = shown.folds | 0;
      const ribAmp = shown.ribs * Math.pow(0.5 + 0.5 * Math.cos(folds * theta), 2);
      let pore = 0;
      const rings = [0.22, 0.4, 0.58, 0.76];
      for (const ring of rings) {
        const dr = (rv - ring) / 0.032;
        const along = Math.pow(0.5 + 0.5 * Math.cos(folds * theta), 0.55);
        pore = Math.max(pore, Math.exp(-dr * dr) * along);
      }
      pore *= shown.pores;
      const shoulder = Math.pow(rv, 3);
      const zWall = shown.girdle * 0.5 * shoulder;
      const zDome = shown.dome * Math.pow(Math.max(0, 1 - rv * rv), 1.1);
      const zRib = ribAmp * 0.07 * Math.sin(Math.PI * rv);
      const zRose = Math.exp(-Math.pow(rv / 0.1, 2)) * (0.09 + 0.06 * shown.bloom);
      let z = sign * (zWall + zDome + zRib + zRose - pore * 0.08);
      let rad = rv * 1.05;
      const ax = 1 + shown.asymmetry * 0.95;
      const ay = 1 - shown.asymmetry * 0.28;
      const facing = Math.max(0, Math.cos(theta));
      const dmg = shown.damage * facing;
      rad *= 1 - 0.82 * dmg;
      z *= 1 - 0.72 * dmg;
      const x = Math.cos(theta) * rad * ax;
      const y = z;
      const zz = Math.sin(theta) * rad * ay;
      const light = 0.48 + 0.25 * ribAmp + 0.16 * (1 - rv) * shown.bloom - 0.32 * pore;
      let rgb = hsl(shown.hue, 0.58, Math.max(0.16, Math.min(0.78, light)));
      rgb = mix(rgb, hsl(0.52, 0.42, 0.46), dmg * 0.85);
      if (rv > 0.9) rgb = mix(rgb, [0.96, 0.9, 0.72], ((rv - 0.9) / 0.1) * 0.65);
      return { xyz: [x, y, zz], rgb };
    }

    function pushVertex(positions, colors, xyz, rgb) {
      positions.push(xyz[0], xyz[1], xyz[2]);
      colors.push(rgb[0], rgb[1], rgb[2]);
    }

    function addGrid(positions, colors, indices, segR, segT, sampler) {
      const base = positions.length / 3;
      for (let i = 0; i <= segR; i++) {
        for (let j = 0; j <= segT; j++) {
          const sample = sampler(i / segR, (j / segT) * Math.PI * 2);
          pushVertex(positions, colors, sample.xyz, sample.rgb);
        }
      }
      const row = segT + 1;
      for (let i = 0; i < segR; i++) {
        for (let j = 0; j < segT; j++) {
          const a = base + i * row + j;
          const b = a + 1;
          const c = a + row;
          const d = c + 1;
          indices.push(a, c, b, b, c, d);
        }
      }
    }

    function build() {
      const positions = [];
      const colors = [];
      const indices = [];
      const segT = Math.max(96, (shown.folds | 0) * 12);
      const segR = 36;
      addGrid(positions, colors, indices, segR, segT, (rv, theta) => surfacePoint(rv, theta, 1));
      addGrid(positions, colors, indices, segR, segT, (rv, theta) => surfacePoint(rv, theta, -1));
      const girdleRows = 6;
      addGrid(positions, colors, indices, girdleRows, segT, (rv, theta) => {
        const folds = shown.folds | 0;
        const stripe = 0.5 + 0.5 * Math.cos(folds * 2 * theta);
        const facing = Math.max(0, Math.cos(theta));
        const dmg = shown.damage * facing;
        const rad = 1.055 * (1 - 0.82 * dmg);
        const z = (rv - 0.5) * shown.girdle * (1 - 0.72 * dmg);
        const ax = 1 + shown.asymmetry * 0.95;
        const ay = 1 - shown.asymmetry * 0.28;
        const xyz = [Math.cos(theta) * rad * ax, z, Math.sin(theta) * rad * ay];
        let rgb = hsl(shown.hue, 0.45, 0.32 + 0.18 * stripe);
        rgb = mix(rgb, hsl(0.52, 0.4, 0.4), dmg * 0.8);
        return { xyz, rgb };
      });

      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
      geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
      geo.setIndex(indices);
      geo.computeVertexNormals();
      if (mesh) {
        group.remove(mesh);
        mesh.geometry.dispose();
      }
      mesh = new THREE.Mesh(geo, material);
      group.add(mesh);
    }

    function resize() {
      const width = canvas.clientWidth || canvas.parentElement.clientWidth || 800;
      const height = canvas.clientHeight || canvas.parentElement.clientHeight || 600;
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(width, height, false);
      camera.aspect = width / Math.max(1, height);
      camera.updateProjectionMatrix();
    }

    function frame() {
      frameCount += 1;
      if (target) approach(target);
      if (shown) build();
      if (!drag && shown) yaw += 0.0024 * (0.35 + shown.spin);
      group.rotation.y = yaw;
      group.rotation.x = pitch;
      const breathe = shown ? 1 + 0.025 * Math.sin(frameCount * 0.04) * (0.45 + shown.bloom) : 1;
      group.scale.setScalar(breathe);
      shadow.scale.set(0.85 + (shown ? shown.asymmetry : 0), 1, 0.85);
      const pos = dustGeo.attributes.position;
      for (let i = 0; i < dustCount; i++) {
        pos.array[i * 3 + 1] += 0.0025;
        if (pos.array[i * 3 + 1] > 2.1) pos.array[i * 3 + 1] = -2.1;
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
      setTarget(pose) {
        target = pose;
      },
      shown() {
        return shown;
      },
      resize,
    };
  }
