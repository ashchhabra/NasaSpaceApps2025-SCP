"use client";

import React, { useState, useRef, useEffect } from 'react';
import * as THREE from 'three';
import planet from '../models/planet';
import star from '../models/star';
/**
 *
 *  Engine
 *
 *  The following contains all the logic to render the Scene
 *
 */

interface EngineProps {
// Add all the stuff you need to pass into the Engine here.
}

const Engine: React.FC<EngineProps> = () => {

  // -------------------- Initialize THREEJS Variables -----------------------
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const planetRef = useRef<planet | null>(null);
  const starRef = useRef<star | null>(null);
  const controlsRef = useRef<any | null>(null);
  const initialisedScene = useRef<boolean>(false);

  // First-time initialization of all Three.js entities
  const initialiseScene = (mount: HTMLDivElement) => {
    // Initialize scene
    sceneRef.current = new THREE.Scene();

    // Initialize camera
    cameraRef.current = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    cameraRef.current.position.z = 15;

    // Initialize renderer - only in browser environment
    rendererRef.current = new THREE.WebGLRenderer({ antialias: true });
    rendererRef.current.setSize(window.innerWidth, window.innerHeight);
    mount.appendChild(rendererRef.current.domElement);

    // Initialize controls - dynamically import OrbitControls
    import('three/examples/jsm/controls/OrbitControls').then(({ OrbitControls }) => {
      if (cameraRef.current && rendererRef.current) {
        controlsRef.current = new OrbitControls(cameraRef.current, rendererRef.current.domElement);
        controlsRef.current.enableZoom = true;
      }
    });
  };

  // Initialize the world
  const initialiseWorld = () => {
    planetRef.current = new planet();
    starRef.current = new star();
    sceneRef.current.add(planetRef.current.group);
    sceneRef.current.add(starRef.current.group);
  }

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

  // Animation loop
  const animate = () => {
    requestAnimationFrame(animate);

    rendererRef.current.render(sceneRef.current, cameraRef.current);

    // Update controls
    if (controlsRef.current) {
      controlsRef.current.update();
    }
  };

  //=============================== MAIN ============================
  useEffect(() => {
    // Only run in browser environment
    if (typeof window === 'undefined') return;

    const mount = mountRef.current;
    if (!mount) return;

    console.log("Running Main function.");
    initialiseScene(mount);
    initialiseWorld();
    window.addEventListener('resize', handleResize);
    animate(); 

    console.log("Exiting Main function!");

    return () => {
      if (mount && mount.firstChild) {
        mount.removeChild(mount.firstChild);
      }

      if (rendererRef.current) {
        rendererRef.current.dispose();
      }

      if (controlsRef.current) {
        controlsRef.current.dispose();
      }

      window.removeEventListener('resize', handleResize);
    };
  }, []);

  return <>
    <div ref={mountRef} style={{ width: '100vw', height: '100vh' }} />
  </>;
};

export default Engine;
