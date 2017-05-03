type example = {inputs: list float, output: float};

type actFunc = {func: float => float, deriv: float => float};

let constructList (x: int) (init: float) => {
  let rec constructor (listSoFar: list float) => {
    if (List.length listSoFar < x) {
      constructor (List.append listSoFar [init]);
    } else {
      listSoFar;
    }
  };
  constructor [init];
};

let alpha: float = 0.1;

let examples = [|
  {inputs: [1.0, 2.1], output: 1.4},
  {inputs: [(-2.2), 0.8], output: (-1.7)},
  {inputs: [(-1.3), -0.2], output: (0.4)}
|];

let feedForward (activation: actFunc) (w: list float) (i: list float) => {
  let value = List.fold_right2 (fun x y c => c +. x *. y) w [1.0, ...i] 0.0;
  activation.func value
};

let linearSum (w: list float) (i: list float) => {
  List.fold_right2 (fun x y c => c +. x *. y) w [1.0, ...i] 0.0;
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

let runEpoch activation origWeights examples => {
  let processExample = fun example weightUpdates => {
    let sum = linearSum origWeights example.inputs;
    let act = activation.func sum;
    let actDeriv = activation.deriv sum;
    let diff = example.output -. act;
    let factor = diff *. actDeriv *.alpha;
    let newUpdates = List.map
      (fun (input: float) => input *. factor)
      [1.0, ...example.inputs];
    List.map2
      (fun (update: float) (newUpdate: float) => update +. newUpdate)
      weightUpdates
      newUpdates;
  };

  List.map2
    (fun (origWeight: float) (update: float) => origWeight +. update)
    origWeights
    (Array.fold_right processExample examples (constructList (List.length origWeights) 0.0));
};

Random.self_init ();
let seed () => (Random.float 2.0) -. 1.0;

type statusType = {
  mutable weights: list float,
  mutable error: float,
  mutable epoch: int
};
let status = {
  weights: [seed (), seed (), seed ()],
  error: 1000.0,
  epoch: 0
};

let printError = fun (status: statusType) => {
  print_string "Epoch ";
  print_int status.epoch;
  print_string ", error ";
  print_float status.error;
  print_newline ();
};

while (status.error > 0.01 && status.epoch < 1000 && status.error !== infinity) {
  status.weights = (runEpoch linear status.weights examples);
  status.error = sumSqError linear status.weights examples;
  status.epoch = status.epoch + 1;
  printError status;
};
