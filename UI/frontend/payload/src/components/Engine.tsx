import React, { useState, useRef, useEffect } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import planet from '../models/planet.ts'
import star from '../models/star.ts'
/** 
 *
 *  Engine 
 *
 *  The following contains all the logic to render the Scene
 *
 */

import React, { useState, useRef, useEffect } from 'react';

interface EngineProps {
// Add all the stuff you need to pass into the Engine here.
}

const Engine: React.FC<EngineProps> = () => {

  // -------------------- Initialize THREEJS Variables -----------------------

  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene>(new THREE.Scene());
  const rendererRef = useRef<THREE.WebGLRenderer>(new THREE.WebGLRenderer({ antialias: true }));
  const cameraRef = useRef<THREE.PerspectiveCamera>(
    new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000)
  );
  const planetRef = useRef<planet | null>(null);
  const starRef = useRef<star | null>(null);
  const controlsRef = useRef<any | null>(null);
  const initialisedScene = useRef<boolean>(false);


  // First-time initialization of all Three.js entities
  const initialiseScene = (mount: HTMLDivElement) => {

    // Initialize renderer
    rendererRef.current.setSize(window.innerWidth, window.innerHeight);
    mount.appendChild(rendererRef.current.domElement);

    // Initialize camera
    cameraRef.current.position.z = 15;

    // Initialize controls
    controlsRef.current = new OrbitControls(cameraRef.current, rendererRef.current.domElement);
    controlsRef.current.enableZoom = true;
  };

  // Handle window resize
  const handleResize = () => {
    if (rendererRef.current && cameraRef.current) {
      const width = window.innerWidth;
      const height = window.innerHeight;
      rendererRef.current.setSize(width, height);
      cameraRef.current.aspect = width / height;
      cameraRef.current.updateProjectionMatrix();
    }
  };

//=============================== MAIN ============================ 
  useEffect(() => { 

    const mount = mountRef.current;
    if (!mount) return;
     
    initialiseScene(mount);
    window.addEventListener('resize', handleResize);
     

    return () => {
      if (mount && mount.firstChild) {
        mount.removeChild(mount.firstChild);
      }
      rendererRef.current.dispose();
      controlsRef.current?.dispose();
    };

    console.log("Running Main function."); 

    }, []); 

  return <> 
    <div ref={mountRef} style={{ width: '100vw', height: '100vh' }} />
  </>;
};

export default Engine;
