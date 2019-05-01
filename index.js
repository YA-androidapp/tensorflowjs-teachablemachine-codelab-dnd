const canvasPrefix = 'canvas-'
const classifier = knnClassifier.create();
const classPrefix = 'class-'
const img_height = 224;
const img_width = 224;
const splitStr = ' / ';
const webcamElement = document.getElementById('webcam');
let net;
var classLabels = new Array();

async function setupWebcam() {
  return new Promise((resolve, reject) => {
    const navigatorAny = navigator;
    navigator.getUserMedia = navigator.getUserMedia ||
      navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
      navigatorAny.msGetUserMedia;
    if (navigator.getUserMedia) {
      navigator.getUserMedia({
          video: true
        },
        stream => {
          webcamElement.srcObject = stream;
          webcamElement.addEventListener('loadeddata', () => resolve(), false);
        },
        error => reject());
    } else {
      reject();
    }
  });
}

// Reads an image from the webcam and associates it with a specific class
// index.
function addExampleVideo(_classLabel) {
  // console.log('classLabel: ' + _classLabel);

  // Get the intermediate activation of MobileNet 'conv_preds' and pass that
  // to the KNN classifier.
  const activation = net.infer(webcamElement, 'conv_preds');

  // Pass the intermediate activation to the classifier.
  classifier.addExample(activation, _classLabel);

  let buttonElement = document.getElementById(classPrefix + _classLabel);
  let spt = buttonElement.innerText.split(splitStr)
  buttonElement.innerText = spt[0] + splitStr + ((spt[1]) == undefined ? 1 : parseInt((spt[1]).trim()) + 1);
}

function addExampleImage(_classLabel) {
  // console.log('classLabel: ' + _classLabel);

  var canvasId = canvasPrefix + _classLabel
  var canvas = document.getElementById(canvasId);

  // Get the intermediate activation of MobileNet 'conv_preds' and pass that
  // to the KNN classifier.
  const activation = net.infer(canvas, 'conv_preds');

  // Pass the intermediate activation to the classifier.
  classifier.addExample(activation, _classLabel);

  let buttonElement = document.getElementById(classPrefix + _classLabel);
  let spt = buttonElement.innerText.split(splitStr)
  buttonElement.innerText = spt[0] + splitStr + ((spt[1]) == undefined ? 1 : parseInt((spt[1]).trim()) + 1);
}

async function app() {
  document.getElementById('message').innerText = 'Loading mobilenet..';

  // Load the model.
  net = await mobilenet.load();
  document.getElementById('message').innerText = 'Sucessfully loaded model';

  await setupWebcam();

  // When clicking a button, add a class.
  document.getElementById('add-class-button').addEventListener('click', () => {
    let classLabel = document.getElementById('class-name').value;
    document.getElementById('buttons').innerHTML += ' <button id="' + classPrefix + classLabel + '" onclick="addExampleVideo(\'' + classLabel + '\')">Add ' + classLabel + '</button>';
    document.getElementById('canvases').innerHTML += ' <div style="border:#000 1px solid;float:left;width:256px;height:256px;"><canvas id="' + canvasPrefix + classLabel + '" style="height:' + img_height + 'px;width:' + img_width + 'px;" title="' + classLabel + '"></canvas></div>';
    classLabels.push(classLabel);
    initCanvasTrain(classLabel);

    document.getElementById('class-name').value = '';
  });

  while (true) {
    if (classifier.getNumClasses() > 0) {
      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(webcamElement, 'conv_preds');
      // Get the most likely class and confidences from the classifier module.
      const result = await classifier.predictClass(activation);

      // console.log(result);
      document.getElementById('console').innerText = `
        prediction: ${result.label}\t
        probability: ${result.confidences[result.label]}
      `;
    }

    await tf.nextFrame();
  }
}

function initCanvasTrain(_classLabel) {
  var canvasId = canvasPrefix + _classLabel
  var canvas = document.getElementById(canvasId);
  var ctx = canvas.getContext('2d');
  var render = function(image) {
    canvas.height = img_height;
    canvas.width = img_width;
    ctx.drawImage(image, 0, 0, image.width, image.height, 0, 0, img_width, img_height);
    console.log(image)

    addExampleImage(_classLabel)
  };
  canvas.addEventListener("drop", function(e) {
    e.preventDefault();

    var file = e.dataTransfer.files[0];
    var image = new Image();
    image.onload = function() {
      render(this);
    };

    var reader = new FileReader();
    reader.onload = function(e) {
      image.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }, false);
}

window.onload = function() {
  var cancelEvent = function(e) {
    e.preventDefault();
    e.stopPropagation();
    return false;
  };

  document.addEventListener("drop", cancelEvent, false);
  document.addEventListener("dragover", cancelEvent, false);
  document.addEventListener("dragenter", cancelEvent, false);

  app();
};