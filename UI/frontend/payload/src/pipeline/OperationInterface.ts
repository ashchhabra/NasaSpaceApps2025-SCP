export default interface IOperation<T> {
   invoke(data: T) : void;
}
