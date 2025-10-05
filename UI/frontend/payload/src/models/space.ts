
import * as THREE from 'three';

export default class star {

  private loadedTexture: boolean = false;

  constructor() {
     // initialiseTexture(this.radius);
     initialiseGridLines(10);
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
    const texture = this.textureLoaderRef.load(src);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.minFilter = THREE.LinearMipMapLinearFilter;
    texture.magFilter = THREE.LinearFilter;
    return texture;
  }

  /**
   * @brief Initialises the Shaders 
   */
  private initialiseTexture(radius : number) {
    const coordmesh = new THREE.MeshStandardMaterial({ map: this.loadTexture(globe1.src) });
    const geometry = new THREE.SphereGeometry(radius, 36, 36);
    
    this.coordRef = new THREE.Mesh(geometry, coordmesh);
    this.coordRef.name = 'cloudRef';

    // Create a Group Reference
    this.group.add(this.coordRef);
    
    // Add grid lines
    this.initialiseGridLines();

    this.loadedTexture = true;
  }
  
  /**
   * @brief Initialises the grid lines
   */
  private initialiseGridLines(radius) {
    const geometry = new THREE.SphereGeometry(5, 36, 36);
    
    // Create wireframe material for longitude and latitude lines
    const gridMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.3 });
    
    // Create a wireframe geometry from the sphere
    const wireframeGeometry = new THREE.WireframeGeometry(geometry);
    
    // Create the line segments for the grid
    const gridLines = new THREE.LineSegments(wireframeGeometry, gridMaterial);
    
    // Add to group
    this.group.add(gridLines);
  }

  /**
   * @brief Initialises the Shaders 
   */
  private initialiseShaders(vertexShader : string, fragmentShader : string) { }
}
