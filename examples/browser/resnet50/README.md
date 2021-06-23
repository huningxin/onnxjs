# ONNXRuntime Web Image Classification Example with WebNN acceleration

This example shows:
- How to create an InferenceSession
- Load ResNet50V2 and MobileNetV2 models
- Use an image as input Tensor
- Run the inference using the input
- Get output Tensor back
- Access raw data in the Tensor
- Match the results with the predefined ImageNet vector

## How to run
1. Download model files [resnet50-v2-7.onnx](https://github.com/onnx/models/blob/master/vision/classification/resnet/model/resnet50-v2-7.tar.gz) and [mobilenetv2-1.0.onnx](https://github.com/onnx/models/blob/cbda9ebd037241c6c6a0826971741d5532af8fa4/vision/classification/mobilenet/model/mobilenetv2-7.tar.gz), uncompress and put them in the current folder.

1. Build [ONNXRuntime Web](https://github.com/microsoft/onnxruntime/tree/master/js#onnxruntime-web) and copy the outputs to `examples/browser/dist` folder.

1. Build [WebNN-polyfill](https://github.com/webmachinelearning/webnn-polyfill) and copy the outputs to `examples/browser/webnn-poyfill` folder.

1. Build [WebNN-native](https://github.com/webmachinelearning/webnn-native).

### Browser
1. Start an http server in this folder. You can install [`http-server`](https://github.com/indexzero/http-server) via
    ```
    npm install http-server -g
    ```
    Then start an http server by running
    ```
    http-server .. -c-1 -p 3000
    ```

    This will start the local http server with disabled cache and listens on port 3000

1. Open up the browser and access this URL:
http://localhost:3000/resnet50/

1. Click on Run button to see results of the inference run. If you choose Execution Providers as "wasm, webnn", it will use WebNN-polyfill.

## Electron.js
1. Install electron.js and others via
    ```
    npm install
    ```

1. Install webnn-native node.js binding via
    ```
    npm install <path_to_webnn_native_out_dir> --webnn_native_lib_path=<relative_path_webnn_native_out_dir_to_node_binding_dir>
    ```

1. Set the system environment variables so the webnn-native libraries could be found.

1. Start the app via
    ```
    npm start
    ```

1. Click on Run button to see results of the inference run. If you choose Execution Providers as "wasm, webnn", it will use WebNN-native.

## Files in folder
- **index.html**

    The HTML file to render the UI in browser

- **index.js**

    The main .js file that holds all `ONNX.js` logic of how to load and execute the model.

- **resnet-cat.jpg**

    A sample image chosen from one of the 1000 categories. Could be replaced with any image of your choice.

