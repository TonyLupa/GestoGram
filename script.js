let model;

const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();

const imgEl = document.getElementById("img");
const descEl = document.getElementById("descripcion_imagen");
var count = 0;
var net;
var webcam;

async function app() {
  console.log("Cargando modelo de identificacion de imagenes");
  net = await mobilenet.load();
  console.log("Carga terminada")
  const result = await net.classify(imgEl);
  console.log(result);
  descEl.innerHTML = JSON.stringify(result);
  webcam = await tf.data.webcam(webcamElement);

  while (true) {
    const img = await webcam.capture();
    const result = await net.classify(img);
    const activation = net.infer(img, 'conv_preds');
    var result2;
    try {
      result2 = await classifier.predictClass(activation);
    } catch (error) {
      result2 = {};
    }

    const classes = ["Untrained", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


    document.getElementById('console').innerHTML = `
  <p class="prediction-text">Prediction: ${result[0].className}</p>
  <p class="probability-text">Probability: ${result[0].probability.toFixed(4)}</p>
`;

    try {
      document.getElementById("console2").innerHTML = `
    <p class="prediction-text">Prediction: ${classes[result2.label]}</p>
    <p class="probability-text">Probability: ${result2.confidences[result2.label].toFixed(4)}</p>
  `;
    } catch (error) {
      document.getElementById("console2").innerHTML = "<p class='untrained-text'>Untrained</p>";
    }
    img.dispose();
    await tf.nextFrame();
  }
}

img.onload = async function () {

  try {
    result = await net.classify(img);
    descEl.innerHTML = JSON.stringify(result);
  } catch (error) {

  }
}

async function cambiarImagen() {
  count = count + 1;
  imgEl.src = "https://picsum.photos/200/300?random=" + count;
  descEl.innerHTM = "";
}


async function addExample(classId) {
  const img = await webcam.capture();
  const activation = net.infer(img, true);
  classifier.addExample(activation, classId);
  img.dispose()
}

const saveKnn = async () => {
  let strClassifier = JSON.stringify(Object.entries(classifier.getClassifierDataset()).map(([label, data]) => [label, Array.from(data.dataSync()), data.shape]));
  const storageKey = "knnClassifier";
  localStorage.setItem(storageKey, strClassifier);
};


const loadKnn = async () => {
  const storageKey = "knnClassifier";
  let datasetJson = localStorage.getItem(storageKey);
  classifier.setClassifierDataset(Object.fromEntries(JSON.parse(datasetJson).map(([label, data, shape]) => [label, tf.tensor(data, shape)])));
};


app()