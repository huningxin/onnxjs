const Models = {
  'MobileNetV2': {
    name: 'mobilenetv2-1.0.onnx',
    input: 'data',
    output: 'mobilenetv20_output_flatten0_reshape0',
    postprocess: true,
  },
  'ResNet50V2': {
    name: 'resnet50-v2-7.onnx',
    input: 'data',
    output: 'resnetv24_dense0_fwd',
    postprocess: true,
  }
};

const ExecutionProviders = {
  'wasm_webnn': ['wasm', 'webnn'],
  'wasm': ['wasm'],
  'webgl': ['webgl'],
};

async function runExample() {
  const sessionOptions = {executionProviders: ExecutionProviders[document.getElementById('eps').value]};
  const model = Models[document.getElementById('model').value];
  const numRuns = parseInt(document.getElementById('numRuns').value);
  // Create an ONNX inference session with Wasm backend.
  const session = await ort.InferenceSession.create(model.name, sessionOptions);

  // Load image.
  const imageLoader = new ImageLoader(imageSize, imageSize);
  const imageData = await imageLoader.getImageData('./resnet-cat.jpg');

  // Preprocess the image data to match input dimension requirement, which is 1*3*224*224.
  const width = imageSize;
  const height = imageSize;
  const preprocessedData = preprocess(imageData.data, width, height);

  const inputTensor = new ort.Tensor('float32', preprocessedData, [1, 3, width, height]);
  let feeds = {};
  feeds[model.input] = inputTensor;
  let outputMap = await session.run(feeds);
  // Run model with Tensor inputs and get the result.
  let total = 0;
  for (i = 0; i < numRuns; i++) {
    const startTime = performance.now();
    outputMap = await session.run(feeds);
    const elapsedTime = performance.now() - startTime;
    total += elapsedTime;
  }
  let outputData = outputMap[model.output].data;
  if (model.postprocess) {
    outputData = postprocess(outputData);
  }
  // Render the output result in html.
  printMatches(outputData);

  const average = total/numRuns;
  const inferenceTime = document.getElementById('inferenceTime');
  inferenceTime.innerHTML = `<br>Model: ${model.name}<br>Execution Providers: ${sessionOptions.executionProviders}<br>The average of ${numRuns} inferences took ${average.toFixed(2)} ms`;
}

/**
 * Preprocess raw image data to match Resnet50 requirement.
 */
function preprocess(data, width, height) {
  const dataFromImage = ndarray(new Float32Array(data), [width, height, 4]);
  const dataProcessed = ndarray(new Float32Array(width * height * 3), [1, 3, height, width]);

  // Normalize 0-255 to (-1)-1
  ndarray.ops.divseq(dataFromImage, 128.0);
  ndarray.ops.subseq(dataFromImage, 1.0);

  // Realign imageData from [224*224*4] to the correct dimension [1*3*224*224].
  ndarray.ops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 2));
  ndarray.ops.assign(dataProcessed.pick(0, 1, null, null), dataFromImage.pick(null, null, 1));
  ndarray.ops.assign(dataProcessed.pick(0, 2, null, null), dataFromImage.pick(null, null, 0));

  return dataProcessed.data;
}

function postprocess(arr) {
  // softmax
  const C = Math.max(...arr);
  const d = arr.map((y) => Math.exp(y - C)).reduce((a, b) => a + b);
  return arr.map((value, index) => {
    return Math.exp(value - C) / d;
  });
}

/**
 * Utility function to post-process Resnet50 output. Find top k ImageNet classes with highest probability.
 */
function imagenetClassesTopK(classProbabilities, k) {
  if (!k) { k = 5; }
  const probs = Array.from(classProbabilities);
  const probsIndices = probs.map(
    function (prob, index) {
      return [prob, index];
    }
  );
  const sorted = probsIndices.sort(
    function (a, b) {
      if (a[0] < b[0]) {
        return -1;
      }
      if (a[0] > b[0]) {
        return 1;
      }
      return 0;
    }
  ).reverse();
  const topK = sorted.slice(0, k).map(function (probIndex) {
    const iClass = imagenetClasses[probIndex[1]];
    return {
      id: iClass[0],
      index: parseInt(probIndex[1], 10),
      name: iClass[1].replace(/_/g, ' '),
      probability: probIndex[0]
    };
  });
  return topK;
}

/**
 * Render Resnet50 output to Html.
 */
function printMatches(data) {
  let outputClasses = [];
  if (!data || data.length === 0) {
    const empty = [];
    for (let i = 0; i < 5; i++) {
      empty.push({ name: '-', probability: 0, index: 0 });
    }
    outputClasses = empty;
  } else {
    outputClasses = imagenetClassesTopK(data, 5);
  }
  const predictions = document.getElementById('predictions');
  predictions.innerHTML = '';
  const results = [];
  for (let i of [0, 1, 2, 3, 4]) {
    results.push(`${outputClasses[i].name}: ${Math.round(100 * outputClasses[i].probability)}%`);
  }
  predictions.innerHTML = results.join('<br/>');
}

