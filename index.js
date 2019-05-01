const classifier = knnClassifier.create();
const classPrefix = 'class-'
const webcamElement = document.getElementById('webcam');
let net;
var classLabels = new Array();

async function app() {
  document.getElementById('message').innerText = 'Loading mobilenet..';

  // Load the model.
  net = await mobilenet.load();
  document.getElementById('message').innerText = 'Sucessfully loaded model';

  await setupWebcam();

  // Reads an image from the webcam and associates it with a specific class
  // index.
  const splitStr = ' / '
  const addExample = classId => {
    console.log('classId: ' + classId)

    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.
    const activation = net.infer(webcamElement, 'conv_preds');

    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);

    let buttonElement = document.getElementById(classPrefix + classLabels[classId]);
    let spt = buttonElement.innerText.split(splitStr)
    buttonElement.innerText = spt[0] + splitStr + ((spt[1]) == undefined ? 1 : parseInt((spt[1]).trim()) + 1);
  };

  // When clicking a button, add a class.
  document.getElementById('add-class-button').addEventListener('click', () => {
    let lab = document.getElementById('class-name').value;
    document.getElementById('buttons').innerHTML += ' <button id="' + classPrefix + +lab + '">Add ' + lab + '</button>';
    classLabels.push(lab);
    document.getElementById('class-' + lab).addEventListener('click', () => addExample(classLabels.length - 1));
  });

  while (true) {
    if (classifier.getNumClasses() > 0) {
      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(webcamElement, 'conv_preds');
      // Get the most likely class and confidences from the classifier module.
      const result = await classifier.predictClass(activation);

      document.getElementById('console').innerText = `
        prediction: ${classLabels[result.classIndex]}\t
        probability: ${result.confidences[result.classIndex]}
      `;
    }

    await tf.nextFrame();
  }
}

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

app();