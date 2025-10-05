export default interface IOperation<T> {
   Invoke(data: T) : void;
}
