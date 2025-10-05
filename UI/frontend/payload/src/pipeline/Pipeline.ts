import IOperation from './OperationInterface'

export class Pipeline<T> implements IOperation<T> {
     private readonly operations: IOperation<T>[] = [];
 
     // add operation at the end of the pipeline
     public Register(operation: IOperation<T>): void {
         this.operations.push(operation);
     }
 
     // invoke every operations
     public Invoke(data: T): void {
         for (const operation of this.operations) {
             operation.Invoke(data);
         }
     }
 }

