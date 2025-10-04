
/** 
 *
 *  Engine 
 *
 *  The following contains all the logic to render the Scene
 *
 */

import React, { useState, useRef, useEffect } from 'react';


const Engine: React.FC<EngineProps> = () => {


  // -------------------------- MAIN ------------------------------------

  useEffect(() => {
    console.log("Running Main function.");
    
  }, []);

  return <>
    <div>
    Engine
    </>
  </>;
};

export default Engine;
