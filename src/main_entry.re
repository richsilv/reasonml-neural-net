type example = {inputs: list float, outputs: list float};

type actFunc = {func: float => float, deriv: float => float};

let constructList initFunc (x: int) => {
  let rec constructor (listSoFar: list float) => {
    if (List.length listSoFar < x) {
      constructor (List.append listSoFar [initFunc()]);
    } else {
      listSoFar;
    }
  };
  constructor [initFunc()];
};

let zeroMatrix (exampleMatrix) => {
  List.map
    (fun l => {
      constructList (fun () => 0.0) (List.length l);
    })
    exampleMatrix;
};

let makeMatrix rowSizes genFunc => {
  List.map
    (constructList(genFunc))
    rowSizes;
};

let alpha: float = 0.1;

let examples = [|
  {inputs: [1.0, 2.1], outputs: [0.7, 1.4]},
  {inputs: [(-2.2), 0.8], outputs: [3.6, (-1.7)]},
  {inputs: [(-1.3), -0.2], outputs: [(0.4), 2.0]}
|];

let feedForward (activation: actFunc) (weightsArray: list (list float)) (inputs: list float) => {
  List.map 
    (fun weights => {
      let value = List.fold_right2 (fun x y c => c +. x *. y) weights [1.0, ...inputs] 0.0;
      activation.func value
    })
    weightsArray;
};

let linearSum (w: list float) (i: list float) => {
  List.fold_right2 (fun x y c => c +. x *. y) w [1.0, ...i] 0.0;
};

let sumSqError activation weightsArray example => {
  let actOutputs = feedForward activation weightsArray example.inputs;
  List.fold_right2
    (fun actOutput expOutput error => error +. (actOutput -. expOutput) ** 2.0)
    actOutputs
    example.outputs
    0.0
};

let multiSumSqError activation weightsArray examples => {
  Array.fold_right
    (fun example error => error +. (sumSqError activation weightsArray example))
    examples
    0.0;
};

let linear = {func: fun x => x, deriv: fun x => 1.0};

let processNode = fun activation inputs weights output => {
  let sum = linearSum weights inputs;
  let act = activation.func sum;
  let actDeriv = activation.deriv sum;
  let factor = (output -. act) *. actDeriv *.alpha;

  List.map
    (fun (input: float) => input *. factor)
    [1.0, ...inputs];
};

let processExample = fun activation weightsArray example => {
  List.map2 (processNode activation example.inputs) weightsArray example.outputs;
};

let updatedWeightsFromExample = fun activation weightsArray example => {
  let updatesArray = processExample activation weightsArray example;

  List.map2
    (fun weights updates => {
      List.map2 (fun weight update => weight +. update) weights updates;
    })
    weightsArray
    updatesArray;
};

let runEpoch activation origWeights examples => {
  Array.fold_right
    (fun example weights => updatedWeightsFromExample activation weights example)
    examples
    origWeights;
};

Random.self_init ();
let seed () => (Random.float 2.0) -. 1.0;

type statusType = {
  mutable weights: list (list float),
  mutable error: float,
  mutable epoch: int
};
let status = {
  weights: (makeMatrix [3, 3] seed),
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
  status.error = multiSumSqError linear status.weights examples;
  status.epoch = status.epoch + 1;
  printError status;
};
