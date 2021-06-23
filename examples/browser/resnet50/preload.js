const webnn = require('webnn-node');
global.navigator.ml = webnn.ml;
global.MLGraphBuilder = webnn.MLGraphBuilder;