
import * as THREE from 'three';

export default class Planet {

  private loadedTexture: boolean = false;
  private radius : number;

  constructor() {
     this.radius = 5; // Random value for now  
     // initialiseTexture(this.radius);
     // initialiseShaders();
  }

  /*
   * @brief The following function loads a specific texture from a jpeg file,
   * this is just an internal function.
   *
   * @param src import jpg like the following: import something from "../assets/something.jpg"
   * and then pass the string as something.src to provide the content of the function to threeJS
   */
  private loadTexture(src: string) : THREE.Texture {

  }

  /**
   * @brief Initialises the Shaders 
   */
  private initialiseTexture(radius : number) {

  }

  /**
   * @brief Initialises the Shaders 
   */
  private initialiseShaders(vertexShader : string, fragmentShader : string) {

  }
}
