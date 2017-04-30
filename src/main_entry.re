type example = {inputs: list float, output: float};

type actFunc = {func: float => float, deriv: float => float};

let alpha: float = 0.1;

let examples = [|
  {inputs: [1.0, 2.1], output: 1.4},
  {inputs: [(-2.2), 0.8], output: (-1.7)},
  {inputs: [0.4, 1.4], output: (-0.3)}
|];

let feedForward (activation: actFunc) (w: list float) (i: list float) => {
  let value = List.fold_right2 (fun x y c => c +. x *. y) w [1.0, ...i] 0.0;
  activation.func value
};

let calcError activation weights example => {
  let res = feedForward activation weights example.inputs;
  res -. example.output;
};

let sumSqError activation weights examples => {
  Array.fold_right
    (fun example error => error +. (calcError activation weights example) ** 2.0)
    examples
    0.0;
};

let linear = {func: fun x => x, deriv: fun x => 1.0};

let runEpoch examples weights => {
  let processExample = fun example weights => {
    let error = calcError linear weights example;
    List.map (fun weight: float => weight *. (1.0 -. error *. alpha)) weights;
  };

  Array.fold_right processExample examples weights;
};

Random.self_init ();
let seed () => (Random.float 2.0) -. 1.0;

let weights = runEpoch examples [seed (), seed (), seed ()];
let error = sumSqError linear weights examples;
print_float error;
print_newline ();

let weights = runEpoch examples weights;
let error = sumSqError linear weights examples;
print_float error;
print_newline ();

let weights = runEpoch examples weights;
let error = sumSqError linear weights examples;
print_float error;
print_newline ();
