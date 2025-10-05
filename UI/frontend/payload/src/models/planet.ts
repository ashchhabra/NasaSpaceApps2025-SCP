import * as THREE from 'three';
import BaseSphereModel from './BaseSphereModel'

export default class planet extends BaseSphereModel {

  private loadedTexture: boolean = false;
  private radius : number;

  constructor(radius: number = 5) {
    super(radius);
    this.radius = radius;
    this.initialiseGridLines();
  }

  /**
   * @brief Initialises the grid lines
   */
  initialiseGridLines() {
    const geometry = new THREE.SphereGeometry(this.radius, 36, 36);
    const gridMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.3 });
    const wireframeGeometry = new THREE.WireframeGeometry(geometry);
    const gridLines = new THREE.LineSegments(wireframeGeometry, gridMaterial);
    this.group.add(gridLines);
  }

  /**
   * @brief Initialises the Shaders
   */
  initialiseShaders(vertexShader : string, fragmentShader : string) {
    super.initialiseShaders(vertexShader, fragmentShader);
  }
}
