import * as THREE from 'three';

export default class BaseSphereModel {
  geometry : THREE.SphereGeometry;
  group: THREE.Group;

  constructor(radius : number) {
     this.geometry = new THREE.SphereGeometry(radius, 36, 36);
     this.group = new THREE.Group();
     // Remove call to initialiseGridLines from here as it should be called by child classes
  }

  /**
   * @brief Initialises grid lines - base implementation
   * This method can be overridden by child classes
   */
  protected initialiseGridLines() {
    // Base implementation - can be overridden by child classes
  }

  /**
   * @brief Initialises the Shaders
   */
  initialiseShaders(vertexShader: string, fragmentShader: string) {
    // Implementation to be provided by child classes or base implementation
  }
}

