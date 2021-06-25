// Modules to control application life and create native browser window
const {app, BrowserWindow, protocol} = require('electron')
const path = require('path')

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow

function createWindow() {
  // Create the browser window.
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 840,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      enableBlinkFeatures: "WebAssemblySimd",
      preload: path.join(app.getAppPath(), 'preload.js')
    }
  });

  // Workaround the wasm module is not found.
  const PROTOCOL = 'file';
  protocol.interceptFileProtocol(PROTOCOL, (request, callback) => {
    let url;
    if (request.url.endsWith('.wasm/')) {
      let wasmPath = request.url.substr(PROTOCOL.length + 3);
      wasmPath = wasmPath.substr(0, wasmPath.length - 1);
      url = path.join(__dirname, '..', 'onnxruntime', 'js', 'web', 'dist', wasmPath);
      url = path.normalize(url);
      console.log('redirect ' + wasmPath + ' to ' + url);
    } else {
      url = request.url.substr(PROTOCOL.length + 3);
    }
    callback({path: url});
  });


  mainWindow.loadFile('index.html')
  // Emitted when the window is closed.
  mainWindow.on('closed', function() {
    // Dereference the window object, usually you would store windows
    // in an array if your app supports multi windows, this is the time
    // when you should delete the corresponding element.
    mainWindow = null
  })
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on('ready', createWindow)

// Quit when all windows are closed.
app.on('window-all-closed', function() {
  // On macOS it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform !== 'darwin') app.quit()
})

app.on(
    'activate',
    function() {
      // On macOS it's common to re-create a window in the app when the
      // dock icon is clicked and there are no other windows open.
      if (mainWindow === null) createWindow()
    })

    // In this file you can include the rest of your app's specific main process
    // code. You can also put them in separate files and require them here.
